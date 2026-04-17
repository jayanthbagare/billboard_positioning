# Billboard Placement Analyzer — India Edition

A computer-vision pipeline that analyzes dashcam or walkcam footage to rank outdoor advertising surfaces (billboards, wall hoardings, compound-wall ads, flex banners) by their actual attention value for a driver or pedestrian. Built specifically for **Indian urban roads**: left-hand traffic (LHT), right-hand drive, mixed traffic, signal junctions, and the wide variety of Indian OOH (out-of-home) advertising geometry.

---

## Table of Contents

1. [Motivation & Design Philosophy](#motivation--design-philosophy)
2. [Key Differentiators from Western Baselines](#key-differentiators-from-western-baselines)
3. [Pipeline Overview](#pipeline-overview)
4. [Mathematical Model — Full Detail](#mathematical-model--full-detail)
   - [Frame Geometry Zones](#frame-geometry-zones)
   - [Surface Detection (Three-Pass)](#surface-detection-three-pass)
   - [Overlap De-duplication (IoU Threshold)](#overlap-de-duplication-iou-threshold)
   - [Saliency Score](#saliency-score)
   - [Motion Analysis: Optical Flow](#motion-analysis-optical-flow)
   - [Dwell Weighting](#dwell-weighting)
   - [IoU-Based Tracker with Velocity Prediction](#iou-based-tracker-with-velocity-prediction)
   - [Composite Scoring](#composite-scoring)
   - [LHT Position Weight](#lht-position-weight)
   - [Stop-Zone Score](#stop-zone-score)
   - [Distance Bonus/Penalty](#distancebonuspenalty)
   - [Junction / Cognitive Load Penalty](#junction--cognitive-load-penalty)
   - [Final Composite Clamp & Quality Thresholds](#final-composite-clamp--quality-thresholds)
5. [Configuration Reference](#configuration-reference)
6. [Outputs](#outputs)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Output Directory Structure](#output-directory-structure)
10. [Design Decisions & Tuning Notes](#design-decisions--tuning-notes)

---

## Motivation & Design Philosophy

Existing billboard-scoring tools are calibrated for Western road conditions: right-hand traffic (RHT), high average speeds, low junction density, and standard rectangular large-format billboards. Indian urban roads are fundamentally different:

| Factor | Western Baseline | India |
|---|---|---|
| Traffic side | Right-hand (RHT) | Left-hand (LHT) |
| Drive side | Left | Right |
| Near-side to driver | Right | Left |
| Avg. urban speed | 40–60 km/h | 10–30 km/h |
| Signal dwell time | Short | Long (60–120 s common) |
| Junction complexity | Moderate | Very high |
| Ad surface geometry | Tall vertical billboards | Wide wall hoardings, compound walls, flex banners |
| Dominant ad colours | Variable | Saffron, red, green dominant |

This tool corrects for all of these factors with explicit math described below.

---

## Key Differentiators from Western Baselines

| Feature | This Analyzer |
|---|---|
| **Stop detector** | Optical flow (Farnebäck dense flow) measures per-frame vehicle speed. Stationary frames (signal stops, junctions, jams) earn a dwell multiplier of up to **3×** |
| **LHT position bias** | Left-side surfaces get a **1.18×** multiplier (driver's near side in LHT). Right-side penalised at **0.82×** (far/median side) |
| **Junction penalty** | Steeper than Western baselines: **0.22** subtracted at high-flow frames, because Indian junction complexity consumes more driver cognitive load |
| **Wall hoarding geometry** | Aspect ratio bounds tuned: allows up to **14:1** width:height ratio to capture long compound-wall and building-face ads |
| **Stop-zone tagging** | Each track reports the fraction of its dwell time where the vehicle was stationary. A dedicated "Stop-Zone Premium Surfaces" section in the HTML report surfaces these |
| **Warm colour bonus** | Saliency formula includes a bonus for saffron (H 10–25°), red (H 0–10° and 170–180°), which are dominant Indian OOH colours |

---

## Pipeline Overview

```
Video file
    │
    ├─► Frame sampler (every ⌊vid_fps / target_fps⌋ frames)
    │
    ▼
MotionAnalyzer.process(frame)
    ├─► Background subtractor (MOG2) → exclusion mask
    └─► Farnebäck optical flow       → flow_mag, motion_state, stop_mult
    │
    ▼
detect_surfaces(frame, excl_mask, fw, fh)
    ├─► Pass A: structural edge contours
    ├─► Pass B: saturated colour blobs
    └─► Pass C: white/light compound walls
    │   (geometry filter + dedup + saliency per candidate)
    │
    ▼
Tracker.update(frame_idx, detections, motion_state, stop_mult, flow_mag)
    └─► IoU matching with velocity prediction → persistent tracks
    │
    ▼
score_tracks(finished_tracks, total_frames, mode)
    └─► dwell · saliency · size · stop_zone · LHT weight · distance · junction penalty
    │
    ▼
Outputs: ranked_surfaces.json  |  report.html  |  annotated_frames/
```

---

## Mathematical Model — Full Detail

### Frame Geometry Zones

Each video frame is divided into three vertical zones:

```
┌─────────────────────────────────────────┐  y = 0
│                                         │
│         SKY ZONE  (top 15%)             │  sky_cut = fh × 0.15
│                                         │
├─────────────────────────────────────────┤
│                                         │
│      VALID SURFACE ZONE                 │  candidates must have:
│      (15% – 78% of frame height)        │    y₀ ≥ sky_cut
│                                         │    y₁ ≤ road_cut
├─────────────────────────────────────────┤
│                                         │  road_cut = fh × 0.78
│   ROAD / VEHICLE HOOD ZONE (bottom 22%) │
│                                         │
└─────────────────────────────────────────┘  y = fh
```

- `SKY_FRAC = 0.15` — pole-mounted hoardings start at roughly 10% frame height; sky is excluded above 15%
- `ROAD_FRAC = 0.78` — Indian wall ads can be low-mounted (compound walls), so more vertical space is kept compared to Western baselines

A candidate bounding box clipped to the valid zone must satisfy a minimum valid height:

```
hv = y1 - y0   (clipped height)
hv  ≥  fh × 0.018
```

---

### Surface Detection (Three-Pass)

Detection is run at **0.5× resolution** for speed. All bounding boxes are then scaled back to full resolution.

#### Pass A — Structural Edge Contours

```python
gray    = cvtColor(small, BGR2GRAY)
blurred = GaussianBlur(gray, (5,5), σ=0)
edges_A = Canny(blurred, low=20, high=60)
closed_A = morphologyEx(edges_A, CLOSE, kernel=(5,5), iterations=1)
contours_A = findContours(closed_A, RETR_LIST, CHAIN_APPROX_SIMPLE)
```

Targets structural boundaries of buildings, wall panels, and hoarding frames.

#### Pass B — Saturated Colour Blobs

```python
sat_mask = inRange(hsv, (0, 70, 50), (180, 255, 255))
closed_B = morphologyEx(sat_mask, CLOSE, kernel=(5,5), iterations=2)
contours_B = findContours(closed_B, ...)
```

Targets billboard artwork and flex banner panels whose dominant property is high saturation (S ≥ 70, V ≥ 50).

#### Pass C — White/Light Compound Walls

```python
white_mask = inRange(hsv, (0, 0, 160), (180, 45, 255))
closed_C   = morphologyEx(white_mask, CLOSE, kernel=(5,5), iterations=2)
contours_C = findContours(closed_C, ...)
```

Targets white-painted compound walls and concrete building faces — extremely common in Indian streetscapes and frequently used for hand-painted or pasted advertising.

#### Geometry Filters (applied to all three passes)

For each candidate bounding box `(x, y, w, h)` scaled to full resolution:

| Filter | Condition | Rationale |
|---|---|---|
| Minimum area | `w × hv / (fw × fh)  ≥ 0.0004` | Catches distant hoardings (0.04% of frame) |
| Width ratio (lower) | `w / fw  ≥ 0.018` | Slightly below Western base to catch narrow pole panels |
| Width ratio (upper) | `w / fw  ≤ 0.85` | Exclude near-full-width regions (likely the road itself) |
| Aspect ratio | `0.10  ≤  w / hv  ≤ 14.0` | 14:1 allows long compound-wall banners; 0.10 allows tall narrow pole signs |
| Exclusion mask | `mean(excl_mask[y0:y1, x:x+w])  ≤ 75` | Reject regions dominated by moving objects (vehicles, pedestrians) |

#### Horizontal & Vertical Position Labels

```
Horizontal thirds:
  cx = x + w/2
  position = "left"   if cx < fw × 0.33
           = "right"  if cx > fw × 0.66
           = "centre" otherwise

Vertical thirds:
  cy = (y0 + y1) / 2
  vertical = "upper" if cy < fh × 0.38
           = "lower" if cy > fh × 0.65
           = "mid"   otherwise

Distance proxy (width ratio):
  distance = "near" if w/fw > 0.22
           = "far"  if w/fw < 0.05
           = "mid"  otherwise
```

---

### Overlap De-duplication (IoU Threshold)

After collecting candidates from all three passes, overlapping detections are deduplicated:

```
For each pair of candidates (a, b):
  intersection area = max(0, min(ax2, bx2) - max(ax1, bx1))
                    × max(0, min(ay2, by2) - max(ay1, by1))
  union area        = area_a + area_b - intersection
  IoU               = intersection / union

  if IoU > 0.28:  keep only the first-seen candidate
```

The 0.28 threshold is looser than the standard 0.50 to catch near-duplicate detections from different passes.

---

### Saliency Score

The saliency function estimates how visually attention-grabbing a surface ROI is, adapted for Indian OOH aesthetics:

```
S(roi) = 0.40 × contrast  +  0.28 × saturation  +  0.18 × warm_frac  −  0.14 × clutter
S(roi) = clip(S, 0, 1)
```

Where:

| Term | Computation | Weight | Rationale |
|---|---|---|---|
| `contrast` | `std(gray_roi) / 128` | 0.40 | High luminance variation = visually distinct surface |
| `saturation` | `mean(hsv[:,:,1]) / 255` | 0.28 | Saturated colours attract gaze |
| `warm_frac` | Fraction of pixels matching saffron (H 10–25°, S≥120, V≥80) or red (H 0–10° or 170–180°, S≥120, V≥80) | 0.18 | These hues dominate Indian political, religious, and commercial hoardings |
| `clutter` | `mean(Canny(gray, 50, 150) > 0)` | −0.14 | Very busy edge texture → background clutter → lower legibility |

---

### Motion Analysis: Optical Flow

The `MotionAnalyzer` class performs two concurrent analyses on each sampled frame.

#### Background Subtraction (Exclusion Mask)

```python
bg = createBackgroundSubtractorMOG2(history=100, varThreshold=32, detectShadows=False)
fg = bg.apply(frame)
fg = morphologyEx(fg, OPEN, ellipse_kernel_9×9)
excl_mask = dilate(fg, ellipse_kernel_9×9, iterations=2)
```

The resulting mask marks pixels belonging to moving objects (vehicles, auto-rickshaws, pedestrians, animals). Surfaces are rejected if more than 75/255 (≈29%) of their area is covered by this mask.

#### Dense Optical Flow (Farnebäck Algorithm)

Frames are downsampled to 480p height before flow computation:

```
scale = 480 / fh
small = resize(frame, (int(fw × scale), 480))
gray  = cvtColor(small, BGR2GRAY)

flow = calcOpticalFlowFarneback(
    prev_gray, gray,
    pyr_scale  = 0.5,   # image pyramid scale
    levels     = 2,     # pyramid levels
    winsize    = 15,    # smoothing window
    iterations = 2,
    poly_n     = 5,     # pixel neighbourhood for polynomial expansion
    poly_sigma = 1.1,   # Gaussian std for polynomial expansion
    flags      = 0
)

mag, _ = cartToPolar(flow[..., 0], flow[..., 1])
```

Flow is computed only over the middle region of the frame (rows 15%–75%) to exclude the sky (which moves due to camera vibration, not vehicle movement) and the road/hood (parallax artefacts):

```
mag_roi   = mag[int(h×0.15) : int(h×0.75), :]
flow_mag  = mean(mag_roi)   # pixels/frame at 480p resolution
```

Flow magnitude is smoothed over a **rolling window of 8 frames** to avoid single-frame spikes (road bumps, camera shake):

```
flow_history.append(flow_mag)       # deque(maxlen=8)
smooth_mag = mean(flow_history)
```

#### Motion State Classification

```
smooth_mag < 1.5  →  state = "stopped",  stop_mult = 3.0
smooth_mag < 6.0  →  state = "slow",     stop_mult = 1.6
otherwise         →  state = "moving",   stop_mult = 1.0
```

Thresholds are in **pixels/frame at 480p**. At 4 fps these correspond roughly to:
- `stopped` (< 1.5 px/frame): vehicle effectively stationary — signal stop, traffic jam, railway crossing
- `slow` (1.5–6.0 px/frame): slow crawl < ~15 km/h — congested traffic, approaching junction
- `moving` (≥ 6.0 px/frame): normal urban driving

---

### Dwell Weighting

Every time a tracker observation is made, the **weighted dwell counter** accumulates:

```
weighted_dwell += stop_mult    # for this frame
```

So a surface seen for N frames receives:

```
weighted_dwell = Σᵢ stop_mult(i)    for i in observed frames
```

A surface visible for 10 stationary frames accumulates `10 × 3.0 = 30` weighted dwell units — the same as 30 frames of moving traffic exposure.

The dwell sub-score normalises this to [0, 1] using a target of **20% of total processed frames** as the saturation point:

```
dwell_score = clip( weighted_dwell / (total_frames × 0.20) , 0, 1 )
```

---

### IoU-Based Tracker with Velocity Prediction

The tracker maintains a dictionary of live tracks and a list of finished tracks.

#### Velocity-Predicted Matching

Each track maintains a smoothed velocity `(vx, vy)`. Before matching, the track's position is predicted one frame forward:

```
predicted_bbox = [x + vx, y + vy, w, h]
```

Then every live track is matched against every current detection using IoU:

```
IoU(b1, b2) = intersection_area / union_area

intersection_area = max(0, min(b1.x2, b2.x2) - max(b1.x1, b2.x1))
                  × max(0, min(b1.y2, b2.y2) - max(b1.y1, b2.y1))

union_area = b1.area + b2.area - intersection_area
```

A match is accepted if `IoU ≥ IOU_THRESH = 0.10`. The low threshold (0.10 vs standard 0.50) allows the prediction to drift more, accommodating Indian traffic's large inter-frame position changes due to slow speed and camera shake.

#### Velocity Update (Exponential Moving Average)

On a successful match:

```
vx_new = 0.7 × (new_x - old_x)  +  0.3 × vx_old
vy_new = 0.7 × (new_y - old_y)  +  0.3 × vy_old
```

#### Track Lifecycle

- **New track**: created for any unmatched detection
- **Gap tolerance**: a track survives up to `MAX_GAP = 6` consecutive missed frames (Indian traffic has heavy occlusion)
- **Minimum age**: only tracks with `count ≥ MIN_FRAMES = 3` are considered for scoring
- **Finalised**: all surviving tracks are moved to `finished` at end of video

---

### Composite Scoring

Scoring uses a **weighted linear combination** of four sub-scores, with mode-specific weights:

```
composite = w_dwell × dwell_score
          + w_saliency × saliency_score
          + w_size × size_score
          + w_stop_zone × stop_zone_score
```

#### Mode Weights

| Sub-score | Driver mode | Pedestrian mode |
|---|---|---|
| `dwell` | **0.50** | 0.35 |
| `saliency` | 0.25 | **0.40** |
| `size` | 0.15 | 0.15 |
| `stop_zone` | 0.10 | 0.10 |

Driver mode is dwell-heavy because a driver's attention window is determined by how long they can look at a fixed point. Pedestrian mode is saliency-heavy because pedestrians can stop and choose to look; visual attractiveness drives attention.

#### Size Sub-score

```
area      = mean(area_ratios_per_frame)    # mean fraction of frame area occupied
size_score = clip( area / 0.003 , 0, 1 )
```

Saturates at 0.3% of frame area (0.003). A surface occupying 0.3% or more of the frame is considered "full size".

---

### LHT Position Weight

After the weighted linear combination, the composite score is multiplied by a position weight derived from Indian left-hand traffic geometry:

```
composite × = lht_weight(dominant_position)

lht_weight("left")   = 1.18    # near side — driver naturally glances left
lht_weight("centre") = 1.00    # baseline
lht_weight("right")  = 0.82    # far/median side — driver must look across traffic
```

The dominant position is determined by majority vote across all observed frames:

```
dom_pos = argmax_{p ∈ {left, centre, right}} count(positions == p)
```

**Rationale**: In India's LHT system, the driver sits on the right side of the vehicle. Their natural peripheral gaze falls to the left — toward the footpath, shops, and hoardings on the near side of the road. The right side of the frame corresponds to the median, dividers, and oncoming traffic (the far side), which demands less passive attention.

---

### Stop-Zone Score

```
stopped_frac  = count(motion_states == "stopped") / len(motion_states)
slow_frac     = count(motion_states == "slow")    / len(motion_states)

stop_zone_score = clip( stopped_frac × 1.0  +  slow_frac × 0.5 , 0, 1 )
```

A surface seen only at signal stops scores 1.0. A surface seen only during slow crawl scores 0.5. A surface seen exclusively at speed scores 0.0.

**Stop-zone label** (for the report):
```
stopped_frac > 0.5                    → "signal/junction stop"
stopped_frac + slow_frac > 0.5        → "slow traffic"
otherwise                             → "moving traffic"
```

**Stop-Zone Premium threshold**: surfaces where `stopped_pct > 40%` are highlighted in a dedicated section of the HTML report because their actual attention value is highest despite potentially lower raw dwell count.

---

### Distance Bonus/Penalty

A small fixed adjustment is applied based on dominant observed distance:

```
dom_dist = argmax_{d ∈ {near, mid, far}} count(distances == d)

Δ_dist = { "near": −0.04,   # too close — legibility window is short
            "mid":  +0.05,   # optimal reading distance
            "far":  −0.02 }  # barely visible — lower impact

composite += Δ_dist[dom_dist]
```

"Mid" distance (width ratio 5–22% of frame width) is the optimal reading distance for Indian road speeds. "Near" surfaces are passed too quickly to be read fully, even at slow Indian speeds.

---

### Junction / Cognitive Load Penalty

If the mean optical flow over the track's lifetime exceeds `FLOW_MOVING = 20.0`:

```
mean_flow = mean(flow_mags)

if mean_flow > FLOW_MOVING:
    penalty = JUNCTION_PENALTY × clip( (mean_flow - FLOW_MOVING) / 20.0, 0, 1 )
    composite -= penalty
```

Where `JUNCTION_PENALTY = 0.22`. The penalty scales linearly from 0 at `flow = 20` to 0.22 at `flow = 40` and beyond.

**Rationale**: When a vehicle is moving fast (or decelerating sharply into a complex Indian junction), the driver's visual attention is consumed by navigation. An advertising surface visible only during high-speed, high-complexity driving has low effective exposure. The penalty is set at 0.22 rather than the Western baseline because Indian junction geometry (5-way intersections, unmarked merges, mixed traffic classes) demands more cognitive load per unit of speed.

---

### Final Composite Clamp & Quality Thresholds

```
composite = clip(composite, 0.0, 1.0)

quality = "Excellent"  if composite > 0.65
        = "Good"       if composite > 0.45
        = "Fair"       if composite > 0.28
        = "Poor"       otherwise
```

---

## Configuration Reference

All tunable constants are at the top of `analyze.py`:

### Geometry

| Constant | Value | Meaning |
|---|---|---|
| `SKY_FRAC` | 0.15 | Top fraction excluded as sky |
| `ROAD_FRAC` | 0.78 | Bottom fraction excluded as road/hood |
| `MIN_AREA_FRAC` | 0.0004 | Minimum surface area as fraction of frame |
| `MIN_W_FRAC` | 0.018 | Minimum surface width as fraction of frame width |
| `MAX_W_FRAC` | 0.85 | Maximum surface width as fraction of frame width |
| `MAX_ASPECT` | 14.0 | Maximum width:height ratio (for wide wall hoardings) |
| `MIN_ASPECT` | 0.10 | Minimum width:height ratio (for tall narrow signs) |

### Tracking

| Constant | Value | Meaning |
|---|---|---|
| `IOU_THRESH` | 0.10 | Minimum IoU for track-detection match |
| `MAX_GAP` | 6 | Max consecutive missed frames before track retirement |
| `MIN_FRAMES` | 3 | Minimum observation frames for a track to be scored |

### LHT Position Weights

| Position | Weight | Meaning |
|---|---|---|
| `left` | 1.18 | Near side (footpath, shops) — driver's natural gaze direction |
| `centre` | 1.00 | Baseline |
| `right` | 0.82 | Far side (median, dividers, oncoming) |

### Motion State Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `FLOW_STOPPED` | 1.5 | Optical flow magnitude (px/frame @ 480p) below which vehicle is stationary |
| `FLOW_SLOW` | 6.0 | Flow below which vehicle is in slow crawl |
| `FLOW_MOVING` | 20.0 | Flow above which junction penalty begins |

### Dwell Multipliers

| State | Multiplier | Meaning |
|---|---|---|
| Stopped | 3.0 | Signal stop, traffic jam — maximum attention availability |
| Slow | 1.6 | Crawling traffic — gaze available for periphery |
| Moving | 1.0 | Normal driving baseline |

### Scoring Weights

#### Driver Mode
```
dwell=0.50, saliency=0.25, size=0.15, stop_zone=0.10
```

#### Pedestrian Mode
```
dwell=0.35, saliency=0.40, size=0.15, stop_zone=0.10
```

### Other

| Constant | Value | Meaning |
|---|---|---|
| `JUNCTION_PENALTY` | 0.22 | Max composite score deduction at high-flow frames |

---

## Outputs

### `ranked_surfaces.json`

JSON file containing:

```json
{
  "video": "path/to/video.mp4",
  "mode": "driver",
  "traffic_side": "LHT",
  "frames_processed": 847,
  "fps_used": 4.0,
  "motion_summary": {"stopped": 120, "slow": 210, "moving": 517},
  "surfaces": [
    {
      "surface_id": "S007",
      "rank": 1,
      "composite_score": 0.731,
      "dwell_score": 0.842,
      "weighted_dwell": 168.0,
      "saliency_score": 0.613,
      "size_score": 0.880,
      "stop_zone_score": 0.720,
      "stopped_pct": 62.0,
      "slow_pct": 10.0,
      "stop_label": "signal/junction stop",
      "lht_weight": 1.18,
      "frame_count": 56,
      "first_frame": 42,
      "last_frame": 310,
      "position": "left",
      "vertical": "mid",
      "distance": "mid",
      "aspect_ratio": 3.40,
      "mean_area": 0.00821,
      "quality": "Excellent",
      "recommendation": "Excellent — left side, mid range, signal/junction stop"
    },
    ...
  ]
}
```

### `report.html`

Self-contained HTML report with:
- Summary cards (surfaces ranked, frames analysed, top score, stop-zone count)
- **Top 3 Recommended Locations** with per-score breakdowns
- **Stop-Zone Premium Surfaces** table (surfaces with >40% stopped dwell)
- Full ranked table for top 25 surfaces
- Methodology explanation panel

### `annotated_frames/`

JPEG frames (every 5th sampled frame) showing:
- Bounding boxes coloured by distance: green (mid), light blue (near), blue (far)
- Label per box: `position|distance|sal:{score}`
- Top-right overlay: motion state, flow magnitude, stop multiplier
- Top-left overlay: frame index, surface count

---

## Installation

```bash
pip install opencv-python numpy
```

Python 3.8+ required. No GPU required — all processing runs on CPU via OpenCV.

---

## Usage

```bash
# Basic driver-mode analysis (recommended for dashcam footage)
python analyze.py --video mumbai.mp4 --fps 4 --mode driver

# Show top 15 surfaces instead of default 10
python analyze.py --video bangalore.mp4 --fps 4 --mode driver --top 15

# Pedestrian / walkcam mode
python analyze.py --video walk.mp4 --fps 3 --mode pedestrian

# Custom output directory, skip saving annotated frames (faster)
python analyze.py --video video.mp4 --fps 4 --mode driver --output ./results --no-frames
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to dashcam video (MP4, MOV, AVI) |
| `--fps` | 4.0 | Frames per second to sample. 4 fps recommended for Indian city speeds |
| `--mode` | `driver` | Viewer perspective: `driver` or `pedestrian` |
| `--top` | 10 | Number of top surfaces to print in terminal summary |
| `--no-frames` | false | Skip saving annotated frame JPEGs (speeds up processing) |
| `--output` | `billboard_output/<video_stem>/` | Output directory |

### Sampling Rate Selection

The analyzer samples every `⌊vid_fps / target_fps⌋` frames. For a 30 fps dashcam video:

| `--fps` | Interval | Samples per minute of footage |
|---|---|---|
| 2 | 15 frames | 120 |
| 4 | 7 frames | 240 |
| 6 | 5 frames | 360 |
| 10 | 3 frames | 600 |

4 fps is the recommended default for Indian city driving at 10–30 km/h. Higher values increase accuracy for fast roads; lower values speed up processing for long videos.

---

## Output Directory Structure

```
billboard_output/
└── <video_stem>/
    ├── ranked_surfaces.json
    ├── report.html
    └── annotated_frames/
        ├── frame_0000.jpg
        ├── frame_0005.jpg
        ├── frame_0010.jpg
        └── ...
```

---

## Design Decisions & Tuning Notes

### Why Farnebäck optical flow instead of feature matching?

Farnebäck dense flow is computed on the entire frame, giving a spatially distributed estimate of scene motion. Feature matching (e.g., ORB + RANSAC) is more accurate for ego-motion estimation but requires finding stable features, which fails on textureless Indian roads. Dense flow is robust to the low-texture, high-dust conditions common in Indian urban environments.

### Why background subtraction for exclusion, not optical flow?

Optical flow measures camera motion (ego-vehicle) not object motion. MOG2 background subtraction correctly identifies objects moving *differently from the background*, which is exactly what needs to be excluded — vehicles, people, animals. Using flow for exclusion would incorrectly exclude everything on a moving-camera frame.

### Why IoU threshold of 0.10 for tracking?

At 4 fps and Indian city speeds (10–30 km/h), a vehicle travels 0.7–2.1 meters between samples. A billboard 10 metres away subtends roughly 15–20% of frame width. At 20 km/h the camera moves ~1.4 m between samples, shifting the billboard by ~7% of frame width — enough to reduce IoU to ~0.4–0.6 for a perfect match. With camera shake (Indian roads), this can fall further. A threshold of 0.10 prevents premature track termination.

### Why MAX_GAP of 6?

Indian traffic has heavy occlusion: auto-rickshaws, buses, and trucks frequently block sightlines. A hoarding may be invisible for 6 frames (~1.5 seconds at 4 fps) even when the vehicle is stopped at a signal directly beneath it. A gap of 6 keeps the track alive through these occlusions.

### Why warm colour bonus in saliency?

Analysis of Indian OOH advertising shows that saffron (Pantone 1505), vermillion red, and BJP/Congress/religious organisation greens are statistically dominant in political and commercial hoardings across Hindi-belt, south Indian, and metro markets. These hues fall in specific HSV ranges that the formula targets. Without this bonus, plain white or pastel-background hoardings with high contrast could outrank visually dominant Indian-style advertising.

### Why stop-zone weighting matters more in India than in the West

In a typical Western city, a driver might stop at a signal for 30–60 seconds per kilometre of driving. In Mumbai, Bengaluru, or Delhi, a driver can spend 3–8 minutes per kilometre stopped at signals, railway crossings, and uncontrolled intersections. This means the dwell time at stop zones is not a minor bonus — it is the dominant factor in actual advertising exposure. The 3× multiplier reflects this structural difference.
