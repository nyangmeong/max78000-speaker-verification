"""
VoxCeleb1-O EER 평가 스크립트.

memmap 불필요 — WAV 파일에서 직접 FBank 추출.

사용법:
    # float 학습 또는 QAT 체크포인트 (기본, simulate=False)
    python scripts/eval_eer_vox1.py --checkpoint logs/.../best.pth.tar

    # quantize.py 출력 체크포인트 (simulate=True + [-128,+127] 입력)
    python scripts/eval_eer_vox1.py --checkpoint logs/.../best.pth.tar --simulate --act-mode-8bit

출력:
    seg 1v1  : enroll/test 모두 best-energy segment 1개
    utt vs 1 : enroll=발화 전체 segment 평균, test=best-energy segment 1개
"""

import argparse
import glob as _glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import ai8x  # noqa: E402

# ─────────────────────────────────────────────
# FBank 파라미터 (학습과 동일)
# ─────────────────────────────────────────────
SAMPLE_RATE  = 16_000
N_MELS       = 80
N_FFT        = 512
WIN_LENGTH   = 400
HOP_LENGTH   = 160
N_FRAMES     = 128
MIN_SAMPLES  = (N_FRAMES - 1) * HOP_LENGTH + WIN_LENGTH   # 20,720 (~1.3초)
SEG_HOP      = int(SAMPLE_RATE * 0.5)                     # 0.5초 hop

FBANK_CLIP   = 6.8

VOX1_WAV_ROOT = Path(os.path.expanduser('~/datasets/voxceleb1/test/wav'))
TRIAL_PATH    = Path(os.path.expanduser('~/datasets/voxceleb1/veri_test2.txt'))


# ─────────────────────────────────────────────
# FBank 추출
# ─────────────────────────────────────────────

def build_fbank(device: torch.device) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=20.0,
        f_max=7600.0,
        center=False,
        power=2.0,
    ).to(device)


def wav_to_segments(wav_path: Path, fbank: T.MelSpectrogram,
                    device: torch.device) -> torch.Tensor:
    """WAV → [S, 1, 80, 128] float32 (학습 전처리와 동일)"""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    wav = wav.to(torch.float32)

    n = wav.numel()
    if n <= MIN_SAMPLES:
        left = (MIN_SAMPLES - n) // 2
        wav = F.pad(wav, (left, MIN_SAMPLES - n - left))
        raw_segs = [wav]
    else:
        raw_segs = [wav[s:s + MIN_SAMPLES].clone()
                    for s in range(0, n - MIN_SAMPLES + 1, SEG_HOP)]
        tail = n - MIN_SAMPLES
        if not raw_segs or (tail % SEG_HOP) > SEG_HOP // 2:
            raw_segs.append(wav[tail:tail + MIN_SAMPLES].clone())

    feats = []
    with torch.no_grad():
        for seg in raw_segs:
            feat = torch.log(fbank(seg.unsqueeze(0).to(device)).clamp(min=1e-9))  # [1, 80, T]
            t = feat.shape[-1]
            if t < N_FRAMES:
                feat = F.pad(feat, (0, N_FRAMES - t))
            elif t > N_FRAMES:
                feat = feat[:, :, :N_FRAMES]
            feat = feat - feat.mean(dim=-1, keepdim=True)   # per-bin mean norm
            feats.append(feat)                               # [1, 80, 128]

    return torch.stack(feats)   # [S, 1, 80, 128]


# ─────────────────────────────────────────────
# 정규화 (학습과 동일)
# ─────────────────────────────────────────────

def normalize(feats: torch.Tensor, act_mode_8bit: bool) -> torch.Tensor:
    """[S, 1, 80, 128] → clamp + scale"""
    if act_mode_8bit:
        return feats.clamp(-FBANK_CLIP, FBANK_CLIP).mul(127.0 / FBANK_CLIP).round().clamp(-128, 127)
    return feats.clamp(-FBANK_CLIP, FBANK_CLIP).div(FBANK_CLIP)


# ─────────────────────────────────────────────
# 임베딩 추출
# ─────────────────────────────────────────────

@torch.no_grad()
def embed_best_segment(segs: torch.Tensor, model, device, act_mode_8bit: bool) -> torch.Tensor:
    """best-energy segment 1개 임베딩 → [1, D]"""
    energy = segs.exp().pow(2).mean(dim=(-2, -1)).squeeze(1)   # [S]
    best   = int(energy.argmax().item())
    x = normalize(segs[best:best + 1], act_mode_8bit).to(device)
    return F.normalize(model(x), dim=1)


@torch.no_grad()
def embed_utterance(segs: torch.Tensor, model, device, act_mode_8bit: bool) -> torch.Tensor:
    """전체 segment 평균 임베딩 → [1, D]"""
    x = normalize(segs, act_mode_8bit).to(device)              # [S, 1, 80, 128]
    return F.normalize(model(x).mean(dim=0, keepdim=True), dim=1)


# ─────────────────────────────────────────────
# EER
# ─────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


def eval_trials(trials, enroll_cache, test_cache):
    scores, labels, missing = [], [], 0
    for label, enroll_rel, test_rel in trials:
        e = enroll_cache.get(enroll_rel)
        t = test_cache.get(test_rel)
        if e is None or t is None:
            missing += 1
            continue
        scores.append(float(F.cosine_similarity(e, t)))
        labels.append(label)
    if missing:
        print(f'  (경고: {missing}개 trial 누락)')
    return compute_eer(np.array(scores), np.array(labels))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='VoxCeleb1-O EER evaluation')
    parser.add_argument('--checkpoint', required=True,
                        help='model checkpoint (.pth.tar)')
    parser.add_argument('--model', default='ai85sv',
                        help='model factory 함수명 (default: ai85sv)')
    parser.add_argument('--dr', type=int, default=64,
                        help='embedding dimension (default: 64)')
    parser.add_argument('--act-mode-8bit', action='store_true',
                        help='입력을 [-128,+127]로 정규화 (quantize.py 출력 평가 시)')
    parser.add_argument('--simulate', action='store_true',
                        help='simulate=True (quantize.py 출력 체크포인트 평가 시 사용)')
    parser.add_argument('--device', default='MAX78000')
    parser.add_argument('--vox1-root', default=str(VOX1_WAV_ROOT),
                        help='VoxCeleb1 test wav 루트 디렉토리')
    parser.add_argument('--trial-file', default=str(TRIAL_PATH),
                        help='trial list 파일 경로')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav_root = Path(args.vox1_root).expanduser()
    trial_f  = Path(args.trial_file).expanduser()

    print(f'device    : {device}')
    print(f'model     : {args.model}  dr={args.dr}')
    print(f'checkpoint: {args.checkpoint}')
    print(f'8bit mode : {args.act_mode_8bit}')

    for p in [wav_root, trial_f]:
        if not p.exists():
            print(f'ERROR: {p} 없음')
            sys.exit(1)

    # ── 모델 로드 ──────────────────────────────
    # 기본값: simulate=False  (float / QAT 체크포인트 — act_mode_8bit 없이 학습된 것)
    # --simulate: quantize.py 로 변환한 체크포인트 평가 시 (+ --act-mode-8bit 함께 사용)
    ai8x.set_device(device=85, simulate=args.simulate, round_avg=False)

    _models_dir = str(Path(__file__).parent.parent / 'models')
    model_fn = None
    for _path in sorted(_glob.glob(os.path.join(_models_dir, 'ai85net-*.py'))):
        _spec = importlib.util.spec_from_file_location(
            os.path.splitext(os.path.basename(_path))[0], _path)
        try:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
        except Exception:
            continue
        if hasattr(_mod, args.model):
            model_fn = getattr(_mod, args.model)
            break

    if model_fn is None:
        print(f'ERROR: factory 함수 "{args.model}"을 models/ 에서 찾지 못했습니다.')
        sys.exit(1)

    model = model_fn(pretrained=False, num_classes=1251, dr=args.dr, bias=True)
    model = model.to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    ai8x.update_model(model)   # quantize/clamp 함수 재초기화 (quantized 체크포인트 필수)
    model.eval()
    print('checkpoint loaded.\n')

    fbank = build_fbank(device)

    # ── 캐시 구축 ──────────────────────────────
    wav_files = sorted(wav_root.rglob('*.wav'))
    print(f'총 {len(wav_files):,}개 utterance 처리 중...')

    seg_cache: dict[str, torch.Tensor] = {}
    utt_cache: dict[str, torch.Tensor] = {}

    for wav_path in tqdm(wav_files, ncols=80):
        rel = str(wav_path.relative_to(wav_root))
        try:
            segs = wav_to_segments(wav_path, fbank, device)   # [S, 1, 80, 128]
        except Exception as e:
            tqdm.write(f'  skip {rel}: {e}')
            continue
        seg_cache[rel] = embed_best_segment(segs, model, device, args.act_mode_8bit).cpu()
        utt_cache[rel] = embed_utterance(segs, model, device, args.act_mode_8bit).cpu()

    print(f'{len(seg_cache):,}개 utterance 완료\n')

    # ── trial list 로드 ────────────────────────
    trials = []
    with open(trial_f) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                trials.append((int(parts[0]), parts[1], parts[2]))
    print(f'trials: {len(trials):,}\n')

    # ── EER 평가 ──────────────────────────────
    print(f'{"mode":<18}  {"EER":>7}  {"threshold":>10}')
    print('─' * 42)

    eer, thr = eval_trials(trials, seg_cache, seg_cache)
    print(f'{"seg 1v1":<18}  {eer * 100:>6.2f}%  {thr:>10.4f}')

    eer, thr = eval_trials(trials, utt_cache, seg_cache)
    print(f'{"utt vs 1":<18}  {eer * 100:>6.2f}%  {thr:>10.4f}')

    print('─' * 42)


if __name__ == '__main__':
    main()
