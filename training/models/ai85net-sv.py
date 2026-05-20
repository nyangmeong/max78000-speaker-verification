###################################################################################################
#
# Speaker Verification for MAX78000 — ThinResNet (VGGNet), INT8 전용
#
# ~/project/dvector/model.py 구조를 MAX78000 제약에 맞게 이식.
#
# 입력 : [B, 1, 80, 128]
# 출력 : [B, dr]  L2-normalized d-vector
#
# MAX78000 제약 적용:
#   - stride=2 conv 미지원 → 외부 MaxPool2d 대체
#   - AdaptiveAvgPool2d 미지원 → AvgPool2d((10, 16)) 고정
#   - 잔차 합산: ai8x.Add() + FusedConv2dBNReLU 1×1 (eltwise 후 1×1 conv 포함)
#   - BatchNorm: Conv 레이어 → FusedConv2dBN* (QAT 시 자동 folding)
#
# 채널 흐름 (model.py 대응):
#   stem      : Conv(1→16) + MaxPool(2,2)   → [16, 40, 64]   ← model.py stem
#   stage1    : BasicBlock(16→16, s=1)      → [16, 40, 64]   ← model.py stage1
#   pool1     : MaxPool(2,2)                → [16, 20, 32]   ← model.py stage2 stride=2 대체
#   stage2    : BasicBlock(16→32)           → [32, 20, 32]   ← model.py stage2
#   pool2     : MaxPool(2,2)                → [32, 10, 16]   ← model.py stage3 stride=2 대체
#   stage3    : BasicBlock(32→64)           → [64, 10, 16]   ← model.py stage3
#   stage4    : BasicBlock(64→128)          → [128, 10, 16]  ← model.py stage4
#   gap       : AvgPool((10,16))            → [128,  1,  1]  ← model.py GAP
#   head      : Linear(128→dr)              → [dr]           ← model.py fc
#
# BasicBlock 구조 (stages 1–3 vs stage4):
#   stages 1–3 (split_add_relu=False):
#     conv1        : FusedConv2dBNReLU 3×3
#     conv2        : FusedConv2dBN 3×3     (ReLU는 add 이후)
#     shortcut     : FusedConv2dBN 1×1     (채널 불일치 시)
#     add_eltwise  : ai8x.Add()            (HW: eltwise 부분 of stage*.add layer)
#     add          : FusedConv2dBNReLU 1×1 (HW: stage*.add layer conv+relu 부분)
#     → YAML: stage*.add  op=Conv2d, eltwise=Add, activate=Relu
#
#   stage4 (split_add_relu=True, 128ch HW bug 우회):
#     conv1        : FusedConv2dBNReLU 3×3
#     conv2        : FusedConv2dBN 3×3
#     shortcut     : FusedConv2dBN 1×1
#     add          : ai8x.Add()            (HW: stage4.add, op=none, eltwise=Add)
#     add_relu     : FusedConv2dBNReLU 1×1 (HW: stage4.add_relu, op=Conv2d, activate=Relu)
#
# 가중치 메모리 (INT8, bias=False, dr=64):
#   stem        1→ 16  3×3:   1× 16×9 =     144 B
#   stage1.c1  16→ 16  3×3:  16× 16×9 =   2,304 B
#   stage1.c2  16→ 16  3×3:  16× 16×9 =   2,304 B
#   stage1.add 16→ 16  1×1:  16× 16×1 =     256 B
#   stage2.c1  16→ 32  3×3:  16× 32×9 =   4,608 B
#   stage2.c2  32→ 32  3×3:  32× 32×9 =   9,216 B
#   stage2.sc  16→ 32  1×1:  16× 32×1 =     512 B
#   stage2.add 32→ 32  1×1:  32× 32×1 =   1,024 B
#   stage3.c1  32→ 64  3×3:  32× 64×9 =  18,432 B
#   stage3.c2  64→ 64  3×3:  64× 64×9 =  36,864 B
#   stage3.sc  32→ 64  1×1:  32× 64×1 =   2,048 B
#   stage3.add 64→ 64  1×1:  64× 64×1 =   4,096 B
#   stage4.c1  64→128  3×3:  64×128×9 =  73,728 B
#   stage4.c2 128→128  3×3: 128×128×9 = 147,456 B
#   stage4.sc  64→128  1×1:  64×128×1 =   8,192 B
#   stage4.ar 128→128  1×1: 128×128×1 =  16,384 B  (add_relu)
#   linear    128→ 64  FC:  128× 64   =   8,192 B
#   ─────────────────────────────────────────────
#   합계                               = 335,760 B ≈ 328 KB
#
###################################################################################################
"""
Speaker Verification network for MAX78000 — ThinResNet structure (INT8-only).
"""

import torch.nn as nn
import torch.nn.functional as F

import ai8x


class BasicBlock(nn.Module):
    """
    MAX78000용 BasicBlock (~/project/dvector/model.py BasicBlock 이식).

    conv1        : FusedConv2dBNReLU (Conv + BN + ReLU)
    conv2        : FusedConv2dBN     (Conv + BN; add 이후 ReLU)
    shortcut     : FusedConv2dBN 1×1 (채널 불일치 시)

    split_add_relu=False (stages 1–3, default):
      add_eltwise  : ai8x.Add()            — eltwise sum (YAML: eltwise 부분)
      add          : FusedConv2dBNReLU 1×1 — 학습된 1×1 conv+ReLU (YAML: stage*.add)

    split_add_relu=True (stage4):
      add          : ai8x.Add()            — pure eltwise (YAML: stage4.add)
      add_relu     : FusedConv2dBNReLU 1×1 — 1×1 conv+ReLU (YAML: stage4.add_relu)
      (128ch 멀티패스 HW 버그 우회를 위해 분리)

    stride=2는 외부 MaxPool로 처리; 이 블록은 항상 stride=1.
    """

    def __init__(self, in_ch: int, out_ch: int, bias: bool = False,
                 split_add_relu: bool = False, **kwargs):
        super().__init__()
        self.conv1 = ai8x.FusedConv2dBNReLU(in_ch, out_ch, 3, padding=1,
                                             stride=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dBN(out_ch, out_ch, 3, padding=1,
                                         stride=1, bias=bias, **kwargs)
        self.shortcut = (
            ai8x.FusedConv2dBN(in_ch, out_ch, 1, padding=0, stride=1, bias=bias, **kwargs)
            if in_ch != out_ch else None
        )
        self.split_add_relu = split_add_relu
        if not split_add_relu:
            # stages 1–3: eltwise add → 1×1 conv → ReLU (single HW layer stage*.add)
            # YAML: op=Conv2d #fake, eltwise=Add, activate=Relu
            self.add_eltwise = ai8x.Add()
            self.add = ai8x.FusedConv2dBNReLU(out_ch, out_ch, 1, padding=0,
                                               bias=bias, **kwargs)
        else:
            # stage4: pure eltwise (HW: stage4.add) + separate 1×1 conv+relu (HW: stage4.add_relu)
            self.add = ai8x.Add()
            self.add_relu = ai8x.FusedConv2dBNReLU(out_ch, out_ch, 1, padding=0,
                                                    bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.conv2(self.conv1(x))
        sc = self.shortcut(x) if self.shortcut is not None else x
        if not self.split_add_relu:
            return self.add(self.add_eltwise(out, sc))
        else:
            return self.add_relu(self.add(out, sc))


class AI85SV(nn.Module):
    """
    Speaker Verification for MAX78000 — ThinResNet 구조.

    ~/project/dvector/model.py (VGGNet) 아키텍처를 MAX78000 제약에 맞게 이식.
    입력 [B,1,80,128]: 배포 시 FIFO 기반 HWC streaming 전제가 필요.
    """

    def __init__(
            self,
            num_classes=None,       # pylint: disable=unused-argument
            num_channels: int = 1,
            dimensions=(80, 128),   # pylint: disable=unused-argument
            dimensionality=None,    # pylint: disable=unused-argument
            dr: int = 64,
            bias: bool = False,
            **kwargs
    ):
        super().__init__()

        # ── Stem ────────────────────────────────
        self.stem      = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, padding=1,
                                                 stride=1, bias=bias, **kwargs)
        self.stem_pool = ai8x.MaxPool2d(2, 2, **kwargs)           # [1,80,128] → [16,40,64]

        # ── Stage 1: 16→16, stride=1 ────────────
        self.stage1    = BasicBlock(16, 16, bias=bias, **kwargs)  # [16, 40, 64]
        self.pool1     = ai8x.MaxPool2d(2, 2, **kwargs)           # [16, 20, 32]

        # ── Stage 2: 16→32 ──────────────────────
        self.stage2    = BasicBlock(16, 32, bias=bias, **kwargs)  # [32, 20, 32]
        self.pool2     = ai8x.MaxPool2d(2, 2, **kwargs)           # [32, 10, 16]

        # ── Stage 3: 32→64 ──────────────────────
        self.stage3    = BasicBlock(32, 64, bias=bias, **kwargs)  # [64, 10, 16]

        # ── Stage 4: 64→128, stride=1 ───────────
        # split_add_relu=True: 128ch HW 버그로 인해 stage4.add / stage4.add_relu 분리
        self.stage4    = BasicBlock(64, 128, bias=bias,
                                    split_add_relu=True, **kwargs)  # [128, 10, 16]

        # ── Head ────────────────────────────────
        self.gap    = ai8x.AvgPool2d((10, 16), stride=1)          # [128, 1, 1]
        self.fc     = ai8x.Linear(128, dr, bias=bias, **kwargs)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.stem_pool(self.stem(x))          # [16, 40, 64]
        x = self.pool1(self.stage1(x))             # [16, 20, 32]
        x = self.pool2(self.stage2(x))             # [32, 10, 16]
        x = self.stage4(self.stage3(x))            # [128, 10, 16]
        x = self.gap(x).view(x.size(0), -1)        # [128]
        x = self.fc(x)                             # [dr]
        return F.normalize(x, p=2, dim=1)


def ai85sv(pretrained=False, **kwargs):
    """
    AI85SV speaker verification — ThinResNet 구조.
    INT8 전용, 가중치 메모리 ≈ 328 KB.
    """
    assert not pretrained
    return AI85SV(**kwargs)


models = [
    {
        'name': 'ai85sv',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
]
