# MLB Ball Final Location MVP

This starter scaffold is intentionally narrow. The first milestone is:

> Given a recorded pitch and the first 5-10 visible ball positions after release, predict the ball's final strike-zone crossing location in screen pixels and report the prediction error.

Everything here is offline and measurement-focused. It does not include controller automation, PCI movement, timing scripts, or live gameplay control.

## What this repo gives you

- A manual video-labeling tool for release frame, early ball points, and final crossing point.
- A JSON label schema for each pitch.
- A baseline predictor/evaluator using either:
  - trajectory extrapolation from the first N points, or
  - leave-one-out ridge regression across labeled pitches.
- A debug visualization script that draws observed points, predicted point, and actual crossing point.
- A synthetic dataset generator so you can test the pipeline before labeling real clips.

## Folder layout

```text
mlb_ball_location_mvp/
|
|-- capture/
|   `-- record_clip.py
|-- coords/
|   `-- calibration.py
|-- labeling/
|   `-- manual_label_pitch.py
|-- prediction/
|   `-- predict_location.py
|-- visualization/
|   `-- render_prediction.py
|-- scripts/
|   `-- make_synthetic_dataset.py
|-- data/
|   |-- raw/
|   |-- labels/
|   `-- predictions/
`-- tests/
    `-- test_prediction.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1: Test the pipeline with synthetic labels

```bash
python scripts/make_synthetic_dataset.py --out data/labels/synthetic --count 40
python prediction/predict_location.py --labels data/labels/synthetic --n-points 5 --method velocity --use-actual-cross-frame --out data/predictions/synthetic_velocity.json
python prediction/predict_location.py --labels data/labels/synthetic --n-points 5 --method ridge --out data/predictions/synthetic_ridge.json
```

You should see per-pitch pixel errors plus summary metrics. The velocity smoke test uses the labeled `cross_frame`; that verifies the trajectory math. Later, remove `--use-actual-cross-frame` to test the harder live-style timing estimate.

## Step 2: Record pitch clips

Use your capture-card device index if available. Device `0` is just the default webcam/capture source.

```bash
python capture/record_clip.py --device 0 --out data/raw/pitch_001.mp4 --seconds 8
```

You can also use OBS or your capture software. Just place clips in `data/raw/`.

## Step 3: Label one pitch

```bash
python labeling/manual_label_pitch.py --video data/raw/pitch_001.mp4 --out data/labels/pitch_001.json
```

Controls:

```text
n / p     next / previous frame
j / k     jump forward / backward 10 frames
r         set release_frame to current frame
e         early-point mode
t         target-crossing mode
left click add an early ball point or target point, depending on mode
u         undo last early point, or clear target in target mode
g         toggle grid overlay (25 px / 100 px spacing)
z         toggle strike-zone box (if zone is in the label)
c         toggle mouse coordinate readout
s         save JSON
h         toggle help
q         quit
```

The labeler shows frame number, mode, early-point count, target status, release/cross frames, and live mouse `x/y` in full-frame pixels.

Labeling target:

- Mark `release_frame` as the frame where the ball leaves the pitcher's hand.
- Add the first 5-10 visible ball center points after release.
- Mark the final crossing point at the hitting/strike-zone plane. For the MVP, this is a 2D screen coordinate: `cross_x`, `cross_y`.

## Coordinate calibration (first milestone)

Use the video frame itself as the coordinate grid. This is a **coordinate calibration layer**, not a classification grid.

```text
top-left     = (0, 0)
x increases  = right
y increases  = down
bottom-right = (frame_width - 1, frame_height - 1) for the last pixel index
```

For a 1920x1080 capture, center is approximately `(960, 540)`.

Every label stores exact `x/y` pixel coordinates in `full_frame_pixels` space. The model predicts continuous `cross_x` and `cross_y` values, not grid cells. Evaluation stays in pixel error (`median_error_px`).

Keep three coordinate spaces separate:

1. **Full-frame pixels** — used now for labels, detection, trajectory, and error.
2. **Cropped ROI coordinates** — convert back to full frame before saving: `full_x = crop_x + local_x`.
3. **Normalized strike-zone coordinates** — optional later via a `zone` rectangle in the label.

PCI/stick calibration is a separate later step (screen pixel → controller stick). Do not mix it with ball prediction.

## Label JSON schema

```json
{
  "frame_width": 1920,
  "frame_height": 1080,
  "coordinate_space": "full_frame_pixels",
  "origin": "top_left",
  "x_direction": "right",
  "y_direction": "down",
  "pitch_id": "pitch_001",
  "video": "data/raw/pitch_001.mp4",
  "fps": 60.0,
  "release_frame": 128,
  "early_points": [
    {"frame": 129, "x": 421.0, "y": 210.0},
    {"frame": 130, "x": 432.0, "y": 214.0},
    {"frame": 131, "x": 443.0, "y": 220.0},
    {"frame": 132, "x": 455.0, "y": 228.0},
    {"frame": 133, "x": 466.0, "y": 238.0}
  ],
  "target": {
    "cross_frame": 154,
    "cross_x": 560.0,
    "cross_y": 412.0
  }
}
```

Optional later fields:

```json
{
  "zone": {
    "left": 485.0,
    "top": 285.0,
    "right": 690.0,
    "bottom": 525.0
  },
  "target_normalized": {
    "zone_x": 0.37,
    "zone_y": 0.52
  }
}
```

## Step 4: Evaluate labeled pitches

Once you have at least a few labels:

```bash
python prediction/predict_location.py --labels data/labels --n-points 5 --method velocity --out data/predictions/eval_velocity.json
```

When you have 20+ labels, try ridge regression:

```bash
python prediction/predict_location.py --labels data/labels --n-points 5 --method ridge --out data/predictions/eval_ridge.json
```

## Step 5: Predict and render one pitch

```bash
python prediction/predict_location.py --label data/labels/pitch_001.json --n-points 5 --method velocity --use-actual-cross-frame --out data/predictions/pitch_001_prediction.json
python visualization/render_prediction.py --label data/labels/pitch_001.json --prediction data/predictions/pitch_001_prediction.json --out data/predictions/pitch_001_debug.png --grid --zone
```

The rendered image marks:

- raw video frame (or blank canvas if video is missing)
- optional 25 px / 100 px debug grid
- optional strike-zone box
- observed early ball points and path
- predicted final crossing point
- actual labeled final crossing point
- pixel error and coordinate readout

## Success metric

Start tracking these numbers:

```text
mean_error_px
median_error_px
p90_error_px
mean_abs_x_error_px
mean_abs_y_error_px
```

A reasonable initial progression:

```text
First working baseline:  < 40 px median error
Useful baseline:         < 25 px median error
Strong milestone:        < 15-20 px median error
```

## Practical labeling target

Start with 25 labeled pitches, then evaluate. Do not try to label 500 clips before you know whether the schema and tools are comfortable.
