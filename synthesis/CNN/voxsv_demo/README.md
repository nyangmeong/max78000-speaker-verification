# VoxSV Speaker Verification Demo

Continuous VAD-driven speaker verification on MAX78000 FTHR_RevA.  
No button press required — the board auto-detects speech onset.

**Model:** ai85-voxsv-qat-q (ThinResNet) → 64-D L2-normalised d-vector (Q7)

---

## Hardware

| Item | Detail |
|------|--------|
| Board | MAX78000 FTHR_RevA |
| Microphone | On-board MEMS (ICS-43434, powered via MAX20303) |
| LED0 (red) | Recording active |
| LED1 (green) | Listening / accepted flash |
| SW1 | Hold at startup → DB Build mode |

---

## Quick Start

### 1. Build & Flash

```bash
make -j$(nproc)
# then flash via VS Code task: "flash & run"
```

---

### 2. Build the Speaker Database

The board computes embeddings using its own CNN — no PC-side model needed.

**Step 1 — Enter DB Build mode**

1. Flash the firmware
2. Watch the serial terminal for `[INIT] Done.`
3. **Hold SW1 within 3 seconds** — LEDs alternate red/green during the window
4. You should see:
   ```
   DB BUILD MODE
   Run: python db_gen/manage_db.py
   ```

**Step 2 — Run the PC script**

```bash
python db_gen/manage_db.py --port /dev/ttyACM0
# Windows: --port COM3
```

**Step 3 — Record speakers**

```
Speaker name (Enter to finish): Alice
  Utterances for "Alice" [3]: 3

  Collecting 3 utterance(s) for "Alice"
  Speak when green LED lights up (~1.3 s per utterance)

  [1/3] Waiting...        ← green LED on, speak now
  [1/3] OK  [12, -5, 20, ...]
  [2/3] Waiting...
  ...
  3/3 collected, averaged → stored

Speaker name (Enter to finish): Bob
  ...

Speaker name (Enter to finish):    ← Enter to finish
```

The script writes `include/speaker_db.h` automatically.

**Step 4 — Rebuild firmware with the new DB**

```bash
make -j$(nproc)
# flash again
```

---

### 3. Run the Demo

After flashing with a DB loaded, **do not press SW1** — the board enters identification mode automatically.

```
DB mode: 2 speaker(s) loaded
  [0] Alice
  [1] Bob

[VAD] Listening...  (speak!)
[REC] Capturing 1.3 s...
[CNN] 9643 us

  +------------------------------------+
  | Similarity : +0.8821               |
  | Result     : ACCEPTED  :)         |
  | Speaker    : Alice                 |
  +------------------------------------+
```

| Similarity | Meaning |
|------------|---------|
| ≥ 0.75 | Accepted — speaker identified |
| < 0.75 | Rejected — unknown speaker |

---

### 4. No DB — Continuous Compare Mode

If `speaker_db.h` has `DB_NUM_ENTRIES 0`, the board runs in compare mode:

- **Utterance #1** → stored as reference
- **Utterance #N** → compared with the previous utterance

Useful for tuning the threshold or quick testing without a DB.

---

## Recording Tips

- Speak naturally at normal volume (~1.3 s per utterance)
- Keep ~10 cm distance from the board
- Record at least **3 utterances per speaker** for a stable average
- Avoid recording in very noisy environments
- `peak` should be > 1000 on the serial output; if consistently below 500, speak louder

---

## Serial Output

| Message | Meaning |
|---------|---------|
| `[REC] done. peak=XXXX rms=XXXX` | Audio level after capture |
| `READY` | Board ready for next utterance (DB Build mode) |
| `EMBED:<vals>` | Embedding sent to PC (DB Build mode) |

Baud rate: **115200**
