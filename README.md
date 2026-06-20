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
  - `Enhanced Zhang (Bebo's Experiment)`
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
- Motion-compensated chroma stabilization mode:
  - computes optical flow from original grayscale frames
  - warps previous chroma into the current frame
  - blends only color channels
  - preserves current-frame luminance to avoid blur
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
  docs/
    checkpoints/                 # tracked checkpoint note files
    training_datasets.md         # dataset pros/cons and mixing notes
  TrainingData/                  # local dataset archive staging area
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
5. Optional temporal smoothing mode

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
Temporal smoothing:
- `1` flow_chroma for motion-aware color stabilization
- `2` legacy_average only if you want the old sliding-window blend

Terminal controls:

- `flow_chroma` asks for:
  - `flow memory` with recommended default `0.75`
  - `motion confidence strength` with recommended default `1.00`
- `legacy_average` asks for:
  - `window size` with recommended default `9`
  - larger window = more temporal averaging and higher smear risk
  - smaller window = less smear but weaker color consistency
```

Experimental Zhang choice:

```text
AI backend: Enhanced Zhang (Bebo's Experiment)
Base Zhang format: 2 siggraph17
```

Final output is written beside the input video:

```text
<input_name>_colorized.mp4
```

If temporal smoothing is enabled:

```text
<input_name>_smoothed.mp4
```

Practical read:

- `flow_chroma` is the recommended mode.
- `legacy_average` is faster, but it can smear moving edges because it blends whole frames across time.
- On simpler scenes, still-frame comparisons may look very similar even when video playback shows the real temporal difference.
- Evaluate smoothing by scrubbing video, not by looking at one exported frame only.

## Pipeline Flow

```text
input video
  -> extract frames to temp/<timestamp>/frames
  -> global duplicate-frame middleman
       - exact consecutive duplicates skipped before AI
       - unique frames passed to selected model
  -> AI colorization
  -> duplicate output frames restored
  -> optional temporal smoothing / flow-chroma stabilization
  -> preview report
  -> FFmpeg rebuild with source audio copied back
  -> final MP4
```

Duplicate-frame skipping is exact. If one pixel is different, the frame is treated as new and is sent to the model.

## Model Checkpoints

All local model weights live under one ignored folder:

```text
checkpoints/
  chromanet/
    checkpoint_latest.pth
  instcolorization/
    coco_full_256_train2017/
      latest_net_G.pth
  zhang/
    eccv16.pth
    siggraph17-df00044c.pth
```

Checkpoints are ignored by git because they are large. To share a trained model with teammates, send the needed `.pth` file separately and place it in the matching subfolder.

Useful checkpoint notes:

- `checkpoints/chromanet/checkpoint_latest.pth` is what ChromaNet uses by default.
- `checkpoints/instcolorization/coco_full_256_train2017/latest_net_G.pth` is what InstColorization uses by default.
- `checkpoints/zhang/*.pth` holds the Zhang/ECCV/SIGGRAPH base weights.
- `checkpoint_epochXXX.pth` stores a specific epoch.
- `checkpoint_epochXXX_best.pth` stores best validation checkpoints when generated.

Tracked note files:

- `docs/checkpoints/chromanet.md`
- `docs/checkpoints/instcolorization.md`
- `docs/checkpoints/zhang.md`

## ChromaNet Training

Dataset folders expected by the current config:

```text
ChromaNet_v3_complete/chromanet_v3/data/
  train2017/
  DIV2K_train_HR/
```

Local dataset archive staging:

```text
TrainingData/
  train2017.zip
```

Dataset comparison notes:

```text
docs/training_datasets.md
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
checkpoints/chromanet/
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

## External Research Findings

This section records findings from checking two external student/project repos during model-quality debugging.

### Why We Looked

Current local model behavior:

- ChromaNet checkpoint trained quickly on COCO-style natural images.
- Nature scenes can look acceptable.
- People, buildings, roads, sand, sky/object separation, and high-contrast black-and-white footage can fail.
- Common failures include beige/sepia cast, weak color, gray faces, wrong semantic colors, and average-looking colors.

Root cause:

- Grayscale-to-color is ambiguous. Many colors can share the same luminance.
- A small or weak custom checkpoint learns easy natural priors first, such as grass/trees/sky.
- Pixel regression losses tend to predict average chroma when unsure, which causes desaturated beige/gray output.
- More epochs help, but do not fully solve missing semantic priors quickly.

### Hanshu110/Image_and_Video-Colorization-using-Deep-Learning

Repo checked:

```text
https://github.com/Hanshu110/Image_and_Video-Colorization-using-Deep-Learning
```

What it uses:

- OpenCV DNN.
- Caffe colorization model:
  - `models_colorization_deploy_v2.prototxt`
  - `colorization_release_v2.caffemodel`
  - `pts_in_hull.npy`
- LAB pipeline:
  - keep original `L`
  - resize to model input
  - predict `ab`
  - upscale `ab`
  - combine full-resolution `L + ab`

Why it may look better than our weak custom checkpoint:

- It uses a large pretrained Zhang/ECCV-style model instead of our partially trained ChromaNet checkpoint.
- That pretrained model already has broader color priors.
- It includes simple post-processing controls:
  - color correction
  - saturation/intensity adjustment
  - luminance sharpening/detail enhancement

Important limitation:

- It is not a new training solution.
- It does not solve true semantic ambiguity perfectly.
- It can still miscolor unusual scenes.

Practical value for this project:

- High immediate value.
- Use it as a demo-quality fallback/backend.
- Borrow its post-processing controls for all backends.
- Best next implementation candidate: `zhang_caffe_opencv`.

Suggested integration steps:

1. Add a new backend folder/function for OpenCV Caffe Zhang.
2. Store Caffe weights under:

   ```text
   checkpoints/zhang/
     colorization_release_v2.caffemodel
     models_colorization_deploy_v2.prototxt
     pts_in_hull.npy
   ```

3. Implement single-frame smoke test first.
4. Implement folder colorization using the same project backend contract as ChromaNet, InstColorization, and Zhang PyTorch.
5. Preserve full-resolution luminance and upscale only predicted `ab`.
6. Add optional post-processing:
   - saturation gain
   - color cast correction
   - luminance-only sharpening
7. Compare against current PyTorch Zhang ECCV/SIGGRAPH on the same frames.
8. If better, make it the recommended demo backend while keeping ChromaNet as the custom trained model for project requirements.

### Enhanced Zhang (Bebo's Experiment)

This repo now includes a separate experimental backend that keeps the PyTorch Zhang weights but improves repo-side inference behavior:

- preserves full-resolution luminance
- defaults to `siggraph17` or allows `eccv16`
- adds mild chroma neutralization to reduce whole-frame beige/orange cast
- adds a small saturation lift
- adds a small contrast lift

This is not a new trained model family. It is a safer experiment layer on top of the existing Zhang checkpoints.

### amr-yasser226/video-colorization

Repo checked:

```text
https://github.com/amr-yasser226/video-colorization
```

What it uses:

- Custom ResNet34-UNet.
- Input/output:
  - input `L`
  - predict LAB `ab`
- Training:
  - 15,000-image COCO 2017 subset
  - category-aware sampling across COCO classes
  - 256x256 images
  - AdamW
  - L1 regression
  - mixed precision
  - about 12 epochs in its documented run
- Video:
  - frame extraction
  - frame-by-frame colorization
  - `ab` upsampling
  - EMA smoothing in chroma space

Why it does not immediately fix our issue:

- It is the same general model class as our custom ChromaNet direction: train a model to regress `ab` from `L`.
- It does not ship a ready local checkpoint in the repo.
- It documents the same known limitation: L1 regression tends to produce desaturated average colors on ambiguous scenes.
- It also documents domain-gap issues between COCO and old/noir footage.

Practical value for this project:

- Medium value for later training improvements.
- Low value for an immediate 2-day demo fix.

Ideas worth borrowing later:

- Category-aware COCO subset creation instead of uniform random image selection.
- A pretrained encoder backbone if we keep training custom models.
- EMA smoothing in `ab` space for reducing flicker.
- Clear report language about limitations, domain gap, and regression-to-mean color.
- Future training improvements:
  - perceptual loss
  - adversarial loss
  - classification-based quantized `ab` head
  - domain-specific fine-tuning

### Current Decision

For near-term demo quality:

1. Prefer pretrained Zhang/Hanshu-style Caffe backend.
2. Keep ChromaNet as the custom-trained model for the project/report.
3. Use post-processing controls to improve visual output without retraining.
4. Do not spend the remaining time only adding ChromaNet epochs unless a specific checkpoint test proves it improves people/objects.

For later model quality:

1. Build a more targeted training subset with more people/faces/urban scenes.
2. Add semantic/color-classification loss instead of pure regression.
3. Add perceptual or adversarial loss only after the baseline pipeline is stable.
4. Keep testing on fixed smoke frames:
   - face/person
   - beach/sand/water
   - urban/noir building
   - nature/mountains

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
- ChromaNet and InstColorization preserve the original full-resolution luminance channel and only upscale predicted color channels.
- Older adapters or future model integrations should follow the same full-resolution luminance-preserving path.
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
