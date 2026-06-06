# Testing StreamForge outputs (Spout & NDI) without Resolume

Both outputs are verified with **zero-install Python receivers** (a small live window). GUI
alternatives are listed too. Always start the **sender** first, then the **receiver**.

## Quick verify (test pattern, no AI)

Spout — two terminals:
```powershell
# terminal 1 (sender): broadcast SMPTE bars over Spout
.\.venv\Scripts\python.exe scripts\live.py --sink spout --test-pattern --fps 30 --seconds 120
# terminal 2 (viewer): live window — press q/ESC to quit
.\.venv\Scripts\python.exe scripts\spout_receiver.py
```

NDI — two terminals:
```powershell
.\.venv\Scripts\python.exe scripts\live.py --sink ndi --test-pattern --fps 30 --seconds 120
.\.venv\Scripts\python.exe scripts\ndi_receiver.py
```

Headless checks (no window — saves one received frame):
```powershell
.\.venv\Scripts\python.exe scripts\spout_receiver.py --save out\rx.png
.\.venv\Scripts\python.exe scripts\ndi_receiver.py   --save out\rx.png
# list available senders/sources:
.\.venv\Scripts\python.exe scripts\spout_receiver.py --list
.\.venv\Scripts\python.exe scripts\ndi_receiver.py   --list
```

## Live AI restyle to either output
```powershell
# Spout, fast img2img mode, restyling the test clip
.\.venv\Scripts\python.exe scripts\live.py --sink spout --source file `
  --clip "D:\StreamForge\TestFile\DriveVideo.mp4" --mode img2img --preset BALANCED `
  --prompt "vivid oil painting, thick impasto" --fps 30 --seconds 120
# ...then in another terminal:  scripts\spout_receiver.py
# For NDI, swap --sink ndi  and view with scripts\ndi_receiver.py
```
`--mode edit` = strongest structure-preserving restyle (slower AI); `--mode img2img` = ~3× faster.
The output clock is decoupled, so the receiver always sees a stable 30/50 fps even when the AI
is slower.

## GUI alternatives (free)
- **NDI**: install **NDI Tools** (ndi.video, free) → **NDI Studio Monitor** shows any NDI source;
  pick "… (StreamForge)". This is the standard NDI viewer and also installs the NDI runtime.
- **Both, in one app**: **OBS Studio** (free) +
  - **Spout2** plugin → add a "Spout2 Capture" source → select "StreamForge".
  - **DistroAV** (obs-ndi) plugin → add an "NDI Source" → select "… (StreamForge)".
- **Spout only**: the **Spout SDK** download includes `SpoutReceiver.exe`, a minimal viewer.

## Notes
- Spout is same-machine only (shared GPU texture); NDI works across the network/localhost.
- Spout sender flips vertically by default for Resolume (`SpoutSink(flip=True)`); the Python
  receiver flips back so it displays upright.
- NDI here uses `ndi-python` whose wheel bundles the NDI runtime — no NDI SDK install needed to
  send/receive from our scripts.
