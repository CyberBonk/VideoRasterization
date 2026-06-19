# VideoRasterization

VideoRasterization is a Python CLI pipeline for turning grayscale videos into colorized videos. It extracts video frames, runs one of the available AI colorization backends, optionally applies temporal smoothing, creates a preview report, and rebuilds the final video with source audio preserved.

## Current Features

- Interactive CLI entrypoint: `main.py`
- Frame extraction through bundled FFmpeg from `imageio-ffmpeg`
- Extraction modes:
  - Full quality PNG frames
  - Faster JPG frames
- AI colorization backends:
  - `colorize_chromanet_v3`
  - `instcolorization2025`
  - `colorize_zhang`
- ChromaNet v3 integration with CUDA when available
- ChromaNet style controls:
  - `realistic`
  - `cartoonish`
  - `brush art / grainy`
- ChromaNet color strength controls:
  - default / realistic
  - mild
  - vivid
  - max / experimental
  - custom sliders for confidence filter, color amount, and film grain
- Full-resolution ChromaNet output path:
  - model still predicts color at model input size
  - original full-resolution luminance is preserved
  - only predicted color channels are upscaled
- Global exact duplicate-frame skip:
  - works before all AI models
  - compares consecutive decoded RGB frames
  - skips the AI model only when frames are a 100% pixel match
  - restores duplicate output frames after colorization
- Optional temporal smoothing
- Preview report generation
- Final video rebuild with original audio copied back when available
- Colored console status:
  - success/progress in green
  - warnings in yellow
  - failures in red

## Repository Layout

```text
VideoRasterization/
  main.py                         # interactive pipeline entrypoint
  requirements.txt                # Python dependencies
  video_pipeline/                 # pipeline orchestration
  tools/
    AImodels/                     # model adapters
    FFmpeg/                       # extraction and rebuild helpers
    TemporalSmoothing/            # temporal smoothing helper
  ChromaNet_v3_complete/
    chromanet_v3/                 # ChromaNet training and inference code
      configs/default.yaml        # ChromaNet training config
      train.py                    # ChromaNet trainer
      checkpoints/                # local checkpoints, ignored by git
      data/                       # local training datasets, ignored by git
```

Generated folders such as `temp/`, `reports/`, datasets, checkpoints, logs, videos, zips, and MP4 outputs are ignored by git.

## Setup

Use Python 3.11 on Windows.

Install PyTorch with CUDA first:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
```

Check CUDA:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected on the current machine:

```text
True
NVIDIA GeForce RTX 4070 Laptop GPU
```

## Run The Video Pipeline

Start the interactive pipeline:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" .\main.py
```

The CLI asks for:

1. Input video path
2. Extraction mode
3. AI model backend
4. ChromaNet style/color options, if ChromaNet is selected
5. Optional temporal smoothing window

Example input video:

```text
C:\Users\bebo\Documents\GitHub\VideoRasterization\Videos\Spider-Noir   Official Trailer (Authentic Black & White).mp4
```

Recommended quick choices:

```text
Extraction mode: 2
AI backend: 1) colorize_chromanet_v3
Style: 0 realistic, or 1 cartoonish, or 2 brush art / grainy
Color strength: 2 vivid or 3 max / experimental for weak color checkpoints
Temporal smoothing: blank for off, 9 for smoother videos
```

Final output is written beside the input video:

```text
<input_name>_colorized.mp4
```

If temporal smoothing is enabled:

```text
<input_name>_smoothed.mp4
```

## Pipeline Flow

```text
input video
  -> extract frames to temp/<timestamp>/frames
  -> global duplicate-frame middleman
       - exact consecutive duplicates skipped before AI
       - unique frames passed to selected model
  -> AI colorization
  -> duplicate output frames restored
  -> optional temporal smoothing
  -> preview report
  -> FFmpeg rebuild with source audio copied back
  -> final MP4
```

Duplicate-frame skipping is exact. If one pixel is different, the frame is treated as new and is sent to the model.

## ChromaNet Checkpoints

The project adapter uses:

```text
ChromaNet_v3_complete/chromanet_v3/checkpoints/checkpoint_latest.pth
```

Checkpoints are ignored by git because they are large. To share a trained model with teammates, send the latest checkpoint file separately or place it in the same checkpoints folder.

Useful checkpoint notes:

- `checkpoint_latest.pth` is what the project uses by default.
- `checkpoint_epochXXX.pth` stores a specific epoch.
- `checkpoint_epochXXX_best.pth` stores best validation checkpoints when generated.

## ChromaNet Training

Dataset folders expected by the current config:

```text
ChromaNet_v3_complete/chromanet_v3/data/
  train2017/
  DIV2K_train_HR/
```

Training config:

```text
ChromaNet_v3_complete/chromanet_v3/configs/default.yaml
```

Current config targets 512x512 fine-tuning with small batch size:

```yaml
data:
  image_size: 512
  batch_size: 3

training:
  epochs: 16
  lr: 0.00001
  mixed_precision: true
  amp_dtype: "bfloat16"
```

Start or resume training:

```powershell
cd .\ChromaNet_v3_complete\chromanet_v3
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -u train.py --config configs/default.yaml --resume latest
```

Run for a time limit, finishing the current epoch before stopping:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -u train.py `
  --config configs/default.yaml `
  --resume latest `
  --max-hours 3 `
  2>&1 | Tee-Object -FilePath training_512.log
```

The trainer saves checkpoints in:

```text
ChromaNet_v3_complete/chromanet_v3/checkpoints/
```

## Performance Notes

- ChromaNet runs on CUDA when PyTorch detects the NVIDIA GPU.
- GPU usage may look spiky when CPU prep, image saving, or disk I/O cannot feed the GPU continuously.
- The global duplicate-frame middleman helps most on cartoons, still title cards, frozen frames, and low-motion clips with exact repeated frames.
- Full-resolution ChromaNet output preserves sharp grayscale detail, but color boundaries are still limited by the model input size and training quality.
- For weak checkpoints or black-and-white trailers, use stronger ChromaNet presets:
  - `vivid`
  - `max / experimental`
  - custom confidence filter near `0.00`

## Future Plans

### Queue-Based In-Memory Pipeline

The current pipeline writes extracted frames to disk, then reads them back for the duplicate-frame middleman and AI colorization. A future optimization is to replace most frame disk I/O with a queue-based in-memory pipeline.

Target design:

```text
FFmpeg frame producer
  -> bounded frame queue
  -> duplicate-frame middleman
  -> bounded AI input queue
  -> AI colorizer workers
  -> bounded output queue
  -> video encoder / final writer
```

Core idea:

- Do not store a full video worth of frames in memory.
- Keep only a bounded window of frames, such as 10 frames or another configurable queue size.
- Let each stage run independently when work is available.
- Let each stage wait when its input queue is empty or its output queue is full.
- Use backpressure instead of writing every frame to disk.

FFmpeg producer plan:

- Make FFmpeg output frames as a stream instead of writing all frames to `temp/`.
- Read only enough frames to fill the first bounded queue.
- When the first queue is full, stop reading from FFmpeg so the pipe naturally applies backpressure.
- When queue space opens, continue reading frames.
- Treat this as pausing/resuming extraction without trying to control FFmpeg through manual process pauses.

Duplicate-frame middleman plan:

- Keep at least the previous frame and the current frame available.
- Compare consecutive decoded RGB frames exactly.
- If the current frame is a 100% pixel match with the previous frame, do not enqueue it for AI inference.
- Store a lightweight mapping so the output stage can duplicate the previous colorized result for that frame index.
- If one pixel differs, enqueue the frame normally.

AI queue plan:

- Batch frames from the AI input queue when enough frames are available.
- Flush a smaller final batch when FFmpeg reaches end of video.
- Keep ChromaNet, InstColorization, and Zhang behind the same queue interface where possible.
- Preserve frame order through frame indexes, even if processing is parallel.

Output plan:

- Rebuild video from ordered output frames without needing all frames on disk.
- Prefer piping frames directly into FFmpeg encoder.
- Preserve source audio during final muxing.
- Keep a fallback disk mode for debugging and for models that are not yet stream-friendly.

Edge cases to handle before implementation:

- First frame has no previous frame, so it always goes to AI.
- Last frame must flush all queues even if a batch is not full.
- Long duplicate runs should not grow memory; store duplicate mappings, not frame copies.
- Any queue can block safely without dropping frames.
- AI failures must stop the pipeline cleanly and close FFmpeg pipes.
- Output ordering must stay exact, including duplicated frames.
- Temporal smoothing may still need a small frame window and should be adapted separately.

This is intentionally not implemented yet. It should be revisited after the AI checkpoint quality is good enough, because the rewrite changes the pipeline architecture more than the model behavior.

## Known Bugs / Quality Issues

Blurry or low-resolution-looking output:

- Some AI adapters still predict a `256x256` color image and then upscale the whole result back into the video frame.
- This can make the final video look like a stretched `256x256` image, even when the extracted source frames are full resolution.
- ChromaNet has a partial fix: it preserves the original full-resolution luminance channel and only upscales predicted color channels.
- InstColorization and older adapters may still need the same full-resolution luminance-preserving path.
- Test by comparing frame dimensions and sharpness before/after colorization. If dimensions match but details look smeared, the adapter is probably upscaling low-resolution RGB instead of recombining full-resolution luminance with low-resolution color.

## Troubleshooting

CUDA not available:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Out of VRAM while training:

```yaml
data:
  batch_size: 2
```

No color or very weak color:

- Confirm ChromaNet checkpoint exists.
- Try `color strength = 3`.
- Try custom confidence filter closer to `0.00`.
- Train for more epochs or use a better checkpoint.

Video output has no audio:

- Confirm source video has an audio stream.
- The rebuild step maps source audio with FFmpeg and copies it when available.

No images found:

- Check extracted frames exist under `temp/<timestamp>/frames`.
- Check dataset folders match the expected training structure.

## Git Notes

Ignored local data includes:

- `temp/`
- `reports/`
- `Videos/`
- `*.mp4`
- `*.zip`
- training logs
- ChromaNet datasets
- ChromaNet checkpoints
- ChromaNet previews

Do not commit large datasets, local videos, generated MP4 outputs, or model checkpoints unless the project explicitly changes its storage policy.
