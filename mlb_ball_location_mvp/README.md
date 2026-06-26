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
python prediction/predict_location.py --labels data/labels/synthetic --n-points 5 --method velocity --poly-degree 2 --use-actual-cross-frame --out data/predictions/synthetic_velocity_quad.json
python prediction/predict_location.py --labels data/labels/synthetic --n-points 5 --method ridge --out data/predictions/synthetic_ridge.json
```

You should see per-pitch pixel errors plus summary metrics. The velocity smoke test uses the labeled `cross_frame`; that verifies the trajectory math. **Default `--poly-degree` is 1 (linear)**, which is more stable on noisy manual clicks than quadratic. Later, remove `--use-actual-cross-frame` to test the harder live-style timing estimate.

## Step 2: Record pitch clips

Use your capture-card device index if available. Device `0` is just the default webcam/capture source.

```bash
python capture/record_clip.py --device 0 --out data/raw/pitch_001.mp4 --seconds 8
```

This also writes timing sidecars next to the clip:

```text
data/raw/pitch_001.mp4
data/raw/pitch_001.frames.jsonl
data/raw/pitch_001.meta.json
```

The recorder warms up the device, measures actual capture FPS, and writes the MP4 at that rate so **encoded frame count matches the sidecar**. Check `frames_in_sync` in `.meta.json`. Sidecars store **exact wall-clock timestamps** (`timing_source: capture_sidecar`).

For clips recorded before sidecars existed, backfill approximate timing:

```bash
python scripts/backfill_timing_sidecars.py --raw-dir data/raw
python scripts/verify_timing_sidecars.py --raw-dir data/raw --enrich-labels
```

Backfilled timing uses uniform `frame/fps` spacing (`backfill_uniform_fps`) — good for ms fields in labels, but **new pitches should use `record_clip.py`** for exact capture timing.

For interactive capture (press `r` to clip, `q` to quit):

```bash
python capture/record_clip.py --device 0 --preview-only --clip-seconds 6
```

You can also use OBS or your capture software. Just place clips in `data/raw/`.

## Step 3: Label one pitch

```bash
python labeling/manual_label_pitch.py \
  --video data/raw/pitch_001.mp4 \
  --out data/labels/pitch_001.json \
  --pitch-type fastball \
  --zone-result strike \
  --location-bucket middle
```

### What to label per pitch

```text
release_frame          last frame before first free-flight ball click
early_points           every visible ball position after release (full flight path)
target.cross_frame     frame at plate/strike-zone crossing
target.cross_x/y       ball center at crossing (not catcher glove)
pitch_type             fastball, slider, curveball, etc.
zone_result            strike, borderline, ball, unknown
location_bucket        middle, high, low, inside, outside, etc.
quality + notes        confidence, estimated crossing, anything odd
```

If `data/labels/pitch_001.json` already exists, the labeler **loads and resumes** it so you can fix earlier clicks. Pressing `s` saves progress and prints **validation warnings** for incomplete labels (missing release frame, too few early points, bad frame order, out-of-bounds coordinates). Saving is never blocked.

**Keep the label window at native resolution** while clicking — resizing can skew coordinates.

In **early-point mode (e)**, ball markers show on the **current frame only** so past clicks do not hide the ball. Press **t** to review the full path in target mode. On save, labels include a `timing` block plus `timestamp_ms` on each ball point when a sidecar exists:

```json
"timing": {
  "source": "capture_sidecar",
  "release_timestamp_ms": 3456.789,
  "cross_timestamp_ms": 4222.456,
  "release_to_cross_ms": 765.667
}
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
g         toggle grid overlay (hidden by default; 25 px / 100 px spacing)
z         toggle strike-zone box (if zone is in the label)
c         toggle mouse coordinate readout
s         save JSON
h         toggle help
q         quit
```

On GOAT (~10 frames release to plate), click **every visible ball frame** after release.
The prediction script still uses only the first `--n-points 3/5/7` later.

### Labeling checklist

```text
1. Find frame right before/at release -> press r
2. First frame with ball free from hand -> press e, click ball center
3. Click ball center on every visible frame until plate (auto-advances each click)
4. At crossing frame -> press t, click ball center at strike-zone plane
5. Set pitch_type ([/]), zone_result (;/'), location_bucket (,.)
6. m=confidence, b=toggle estimated crossing, i=notes in terminal
7. Press s to save
```

### Key controls

```text
n/p or arrows   next / previous frame
j/k             jump +/- 10 frames
space           play/pause scrub
f               jump to next unlabeled ball frame
r               set release_frame
e               ball-point mode
t               target-crossing mode
left click      add point (auto-advances in ball mode)
v               toggle current-frame-only markers (default) vs all frames
[/]             pitch type
;/'             zone result
,.              location bucket
m               label confidence
b               toggle crossing_estimated
i               edit notes (terminal prompt)
s               save
q               quit
```

Consistency rules:

- Click **center of ball**, not trail/glow.
- `release_frame` is before the first free-flight click.
- Crossing is plate/zone plane, not catcher glove.
- Skip frames where the ball is hidden instead of guessing.

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
  "pitch_id": "pitch_001",
  "video": "data/raw/pitch_001.mp4",
  "frame_width": 1920,
  "frame_height": 1080,
  "coordinate_space": "full_frame_pixels",
  "origin": "top_left",
  "x_direction": "right",
  "y_direction": "down",
  "fps": 60.0,
  "difficulty": "GOAT",
  "pitch_type": "fastball",
  "zone_result": "strike",
  "location_bucket": "middle",
  "release_frame": 100,
  "early_points": [
    {"frame": 101, "x": 421.0, "y": 210.0},
    {"frame": 102, "x": 433.0, "y": 216.0},
    {"frame": 103, "x": 446.0, "y": 224.0}
  ],
  "target": {
    "cross_frame": 110,
    "cross_x": 558.0,
    "cross_y": 412.0
  },
  "pitch_type": "fastball",
  "zone_result": "strike",
  "location_bucket": "middle",
  "quality": {
    "ball_visible": true,
    "crossing_estimated": false,
    "label_confidence": "high"
  }
  "notes": ""
}
```

`early_points` holds the **full observed flight path**. Prediction uses only the first N points via `--n-points`.

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

`--labels data/labels` loads real pitch JSON files only — it **ignores** `example_*.json` (schema sample lives in `data/examples/`).

Once you have at least a few labels, sweep how early you can predict:

```bash
python prediction/predict_location.py --labels data/labels --n-points 3 --method velocity --out data/predictions/eval_n3.json
python prediction/predict_location.py --labels data/labels --n-points 5 --method velocity --out data/predictions/eval_n5.json
python prediction/predict_location.py --labels data/labels --n-points 7 --method velocity --out data/predictions/eval_n7.json
python prediction/predict_location.py --labels data/labels --n-points 5 --method velocity --poly-degree 1 --out data/predictions/eval_velocity.json
```

Compare linear vs quadratic on real labels:

```bash
python prediction/predict_location.py --labels data/labels --n-points 5 --method velocity --poly-degree 2 --out data/predictions/eval_velocity_quad.json
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
