"""
Billboard Placement Analyzer — India Edition
=============================================
Left-hand traffic (LHT), right-hand drive.
Optimised for Indian urban roads: slow speeds, frequent stops,
mixed traffic, wall hoardings, compound-wall ads, signal junctions.

New vs base analyzer
────────────────────
• Stop detector        — optical flow measures vehicle speed per frame;
                         stationary frames (red lights, junctions) earn
                         a dwell multiplier up to 3×
• LHT position bias    — left-side surfaces weighted higher (driver's near
                         side in LHT); right-side penalised (far/median side)
• Steeper junction     — cognitive load penalty is stronger because Indian
  penalty                junction complexity is higher than Western baselines
• Wall hoarding        — aspect ratio and road-zone tuned for low, wide
  geometry               compound-wall and building-face ads
• Stop-zone tagging    — each track reports what fraction of its dwell time
                         occurred while vehicle was stopped; top stop-zone
                         surfaces are highlighted separately in the report

Usage
─────
    python analyze_india.py --video mumbai.mp4 --fps 4 --mode driver
    python analyze_india.py --video bangalore.mp4 --fps 4 --mode driver --top 15
    python analyze_india.py --video walk.mp4 --fps 3 --mode pedestrian
"""

import argparse, json, sys, time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from bbmp_rules import (
    BBMPMetadata,
    apply_bbmp_compliance,
    compliance_summary,
    RULE_DESCRIPTIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Geometry ─────────────────────────────────────────────────────────────────
# Sky: top 15% — pole-mounted hoardings start at ~10% frame height
SKY_FRAC       = 0.15
# Road/hood: bottom 30% — but wall ads can be low, so we keep more than base
ROAD_FRAC      = 0.78
# Surface size thresholds (fraction of frame area)
MIN_AREA_FRAC  = 0.0004   # 0.04% — catches distant hoardings
# Width thresholds (fraction of frame width)
MIN_W_FRAC     = 0.018    # slightly below base to catch narrow pole panels
MAX_W_FRAC     = 0.85
# Aspect ratio: Indian wall ads are often very wide (10:1), allow more
MAX_ASPECT     = 14.0
MIN_ASPECT     = 0.10

# ── Tracking ──────────────────────────────────────────────────────────────────
IOU_THRESH     = 0.10     # slightly lower — allows prediction to drift more
MAX_GAP        = 6        # frames a track can go missing (Indian traffic = lots of occlusion)
MIN_FRAMES     = 3        # minimum frames to count as a real surface

# ── LHT position weights (India — driver on right, near side = left) ──────────
LHT_POSITION_WEIGHT = {
    "left":   1.18,   # near side: footpath, shops, hoardings at eye level
    "centre": 1.00,
    "right":  0.82,   # far side: median, oncoming, divider
}

# ── Stop / slow detection ─────────────────────────────────────────────────────
# Optical flow magnitude (pixels/frame at native res) thresholds
FLOW_STOPPED   = 1.5    # effectively stationary (signal stop, traffic jam)
FLOW_SLOW      = 6.0    # slow crawl (<15kph) — still high attention
FLOW_MOVING    = 20.0   # normal urban driving

# Dwell multipliers per motion state
STOP_MULTIPLIER  = 3.0   # stopped at signal — maximum attention availability
SLOW_MULTIPLIER  = 1.6   # crawling — gaze available for periphery
MOVING_MULT      = 1.0   # normal driving baseline

# ── Junction / cognitive load penalty ─────────────────────────────────────────
# Applied when a surface is detected during a high-flow (fast motion) frame
# Steeper than base because Indian junction complexity > Western baseline
JUNCTION_PENALTY = 0.22   # subtracted from composite score at busy frames

# ── Scoring weights ───────────────────────────────────────────────────────────
DRIVER_WEIGHTS     = dict(dwell=0.50, saliency=0.25, size=0.15, stop_zone=0.10)
PEDESTRIAN_WEIGHTS = dict(dwell=0.35, saliency=0.40, size=0.15, stop_zone=0.10)


# ─────────────────────────────────────────────────────────────────────────────
# MOTION STATE DETECTOR
# Uses dense optical flow (Farneback) on downsampled frames.
# Returns (state_string, flow_magnitude, exclusion_mask).
# Also provides background subtraction for moving-object exclusion.
# ─────────────────────────────────────────────────────────────────────────────

class MotionAnalyzer:
    """
    Dual purpose:
    1. Background subtraction → exclusion mask for moving objects
    2. Optical flow magnitude → vehicle speed proxy → stop/slow/moving state
    """
    def __init__(self):
        self.bg        = cv2.createBackgroundSubtractorMOG2(
                            history=100, varThreshold=32, detectShadows=False)
        self.excl_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.prev_gray = None          # previous downsampled gray frame for flow
        self._flow_history = deque(maxlen=8)   # smoothed over last 8 samples

    def _downsample(self, frame):
        """Work at 480p for flow — fast and sufficient."""
        h, w = frame.shape[:2]
        scale = 480 / h
        return cv2.resize(frame, (int(w * scale), 480))

    def process(self, frame):
        """
        Returns:
            excl_mask  : uint8 mask, 255 = moving object pixel (full res)
            flow_mag   : float, mean optical flow magnitude (pixels at 480p)
            motion_state: "stopped" | "slow" | "moving"
            stop_mult  : float dwell multiplier for this frame
        """
        # ── Background subtraction (full res for accurate exclusion) ──────
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.excl_k)
        excl_mask = cv2.dilate(fg, self.excl_k, iterations=2)

        # ── Optical flow (downsampled for speed) ──────────────────────────
        small = self._downsample(frame)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        flow_mag = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Exclude top 15% (sky moves differently) and bottom 25% (road)
            h, w = mag.shape
            mag_roi = mag[int(h*0.15):int(h*0.75), :]
            flow_mag = float(np.mean(mag_roi))

        self.prev_gray = gray

        # Smooth over recent frames to avoid single-frame spikes
        self._flow_history.append(flow_mag)
        smooth_mag = float(np.mean(self._flow_history))

        # Classify motion state
        if smooth_mag < FLOW_STOPPED:
            state     = "stopped"
            stop_mult = STOP_MULTIPLIER
        elif smooth_mag < FLOW_SLOW:
            state     = "slow"
            stop_mult = SLOW_MULTIPLIER
        else:
            state     = "moving"
            stop_mult = MOVING_MULT

        return excl_mask, smooth_mag, state, stop_mult


# ─────────────────────────────────────────────────────────────────────────────
# SALIENCY PROXY
# Enhanced for Indian OOH: adds a warm-colour bonus (saffron, red, green
# are dominant in Indian advertising) and a text-density proxy.
# ─────────────────────────────────────────────────────────────────────────────

def saliency(roi):
    if roi is None or roi.size == 0:
        return 0.0

    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray)) / 128.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1])) / 255.0

    # Warm colour bonus: saffron (H 10-25), red (H 0-10 & 170-180), green (H 60-90)
    # These are extremely common in Indian political/religious/commercial hoardings
    warm  = cv2.inRange(hsv, (0,  120, 80), (25,  255, 255))
    warm += cv2.inRange(hsv, (165,120, 80), (180, 255, 255))
    warm_frac = float(np.mean(warm > 0))

    # Clutter (edges) — penalise very busy backgrounds
    edges   = cv2.Canny(gray, 50, 150)
    clutter = float(np.mean(edges > 0))

    score = (contrast * 0.40
             + sat     * 0.28
             + warm_frac * 0.18
             - clutter * 0.14)

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_surfaces(frame, excl_mask, fw, fh):
    """
    Three-pass detection:
      A) Edge + contour  — buildings, wall panels, structural hoardings
      B) Colour blob     — saturated billboard artwork, flex banners
      C) White/light     — white compound walls, concrete buildings
    Filtered by zone, size, aspect, and exclusion mask.
    """
    frame_area = fw * fh
    sky_cut    = int(fh * SKY_FRAC)
    road_cut   = int(fh * ROAD_FRAC)

    # Half-resolution processing for speed
    scale  = 0.5
    small  = cv2.resize(frame, (int(fw * scale), int(fh * scale)))

    gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv     = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    kernel  = np.ones((5, 5), np.uint8)

    # Pass A: structural edges
    edges_a  = cv2.Canny(blurred, 20, 60)
    closed_a = cv2.morphologyEx(edges_a, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts_a, _ = cv2.findContours(closed_a, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pass B: saturated colour blobs (billboard artwork, flex banners)
    sat_mask = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255))
    closed_b = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts_b, _ = cv2.findContours(closed_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pass C: white/light compound walls and buildings (very common in India)
    white_mask = cv2.inRange(hsv, (0, 0, 160), (180, 45, 255))
    closed_c   = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts_c, _  = cv2.findContours(closed_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    all_cnts = list(cnts_a) + list(cnts_b) + list(cnts_c)

    candidates = []
    seen       = []          # for overlap dedup (full-res coords)

    for cnt in all_cnts:
        if cv2.contourArea(cnt) < frame_area * scale**2 * MIN_AREA_FRAC:
            continue

        xs, ys, ws, hs = cv2.boundingRect(cnt)

        # Map to full resolution
        x = int(xs / scale); y = int(ys / scale)
        w = int(ws / scale); h = int(hs / scale)

        # Clamp to valid vertical zone
        y0 = max(y, sky_cut)
        y1 = min(y + h, road_cut)
        hv = y1 - y0
        if hv < fh * 0.018:
            continue

        w_ratio = w / fw
        if not (MIN_W_FRAC < w_ratio < MAX_W_FRAC):
            continue

        aspect = w / max(hv, 1)
        if not (MIN_ASPECT < aspect < MAX_ASPECT):
            continue

        # Exclude moving objects
        excl_roi = excl_mask[y0:y1, x:x+w]
        if excl_roi.size > 0 and np.mean(excl_roi) > 75:
            continue

        # Dedup overlapping detections
        skip = False
        for sx, sy, sw2, sh2 in seen:
            ix1 = max(x, sx);      iy1 = max(y0, sy)
            ix2 = min(x+w, sx+sw2); iy2 = min(y1, sy+sh2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1) * (iy2-iy1)
                union = w*hv + sw2*sh2 - inter
                if union > 0 and inter/union > 0.28:
                    skip = True; break
        if skip:
            continue
        seen.append((x, y0, w, hv))

        roi = frame[y0:y1, x:x+w]
        sal = saliency(roi)

        cx   = x + w // 2
        cy   = (y0 + y1) // 2
        pos  = "left" if cx < fw*0.33 else "right" if cx > fw*0.66 else "centre"
        vert = "upper" if cy < fh*0.38 else "lower" if cy > fh*0.65 else "mid"
        dist = "near" if w_ratio > 0.22 else "far" if w_ratio < 0.05 else "mid"

        candidates.append({
            "bbox":         [x, y0, w, hv],
            "area_ratio":   round(float(w * hv) / frame_area, 5),
            "width_ratio":  round(float(w_ratio), 4),
            "aspect_ratio": round(float(aspect), 2),
            "position":     pos,
            "vertical":     vert,
            "distance":     dist,
            "saliency":     round(sal, 4),
        })

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER  — IoU with velocity prediction
# Extended to record per-frame motion state for each track.
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1) * (y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / max(union, 1)


class Tracker:
    def __init__(self):
        self.tracks   = {}
        self.finished = []
        self.nid      = 0

    def _predict(self, t):
        x, y, w, h = t["bbox"]
        return [int(x + t.get("vx", 0)), int(y + t.get("vy", 0)), w, h]

    def update(self, fidx, dets, motion_state="moving", stop_mult=1.0, flow_mag=0.0):
        hit_t = set(); hit_d = set()

        for tid, t in self.tracks.items():
            predicted = self._predict(t)
            best, bd  = IOU_THRESH, -1
            for di, d in enumerate(dets):
                if di in hit_d: continue
                s = box_iou(predicted, d["bbox"])
                if s > best: best, bd = s, di

            if bd >= 0:
                d  = dets[bd]
                ox, oy = t["bbox"][0], t["bbox"][1]
                nx, ny = d["bbox"][0], d["bbox"][1]
                t["vx"]  = 0.7*(nx-ox) + 0.3*t.get("vx", 0)
                t["vy"]  = 0.7*(ny-oy) + 0.3*t.get("vy", 0)
                t["bbox"] = d["bbox"]; t["last"] = fidx
                t["count"]     += 1
                t["gap"]        = 0
                # Weighted dwell: each frame contributes stop_mult (1.0–3.0)
                t["weighted_dwell"] += stop_mult
                t["saliency"].append(d["saliency"])
                t["area_ratios"].append(d["area_ratio"])
                t["positions"].append(d["position"])
                t["distances"].append(d["distance"])
                t["verticals"].append(d["vertical"])
                t["motion_states"].append(motion_state)
                t["flow_mags"].append(flow_mag)
                hit_t.add(tid); hit_d.add(bd)

        for tid, t in self.tracks.items():
            if tid not in hit_t: t["gap"] += 1

        for di, d in enumerate(dets):
            if di not in hit_d:
                self.tracks[self.nid] = {
                    "id":             self.nid,
                    "bbox":           d["bbox"],
                    "first":          fidx,
                    "last":           fidx,
                    "count":          1,
                    "gap":            0,
                    "vx": 0.0, "vy": 0.0,
                    "weighted_dwell": stop_mult,
                    "saliency":       [d["saliency"]],
                    "area_ratios":    [d["area_ratio"]],
                    "positions":      [d["position"]],
                    "distances":      [d["distance"]],
                    "verticals":      [d["vertical"]],
                    "motion_states":  [motion_state],
                    "flow_mags":      [flow_mag],
                    "aspect":         d["aspect_ratio"],
                }
                self.nid += 1

        retire = [tid for tid, t in self.tracks.items() if t["gap"] >= MAX_GAP]
        for tid in retire:
            self.finished.append(self.tracks.pop(tid))

    def finalise(self):
        for t in self.tracks.values():
            self.finished.append(t)
        self.tracks = {}


# ─────────────────────────────────────────────────────────────────────────────
# SCORER  — India-specific
# ─────────────────────────────────────────────────────────────────────────────

def score_tracks(tracks, total_frames, mode):
    w = DRIVER_WEIGHTS if mode == "driver" else PEDESTRIAN_WEIGHTS

    scored = []
    for t in tracks:
        if t["count"] < MIN_FRAMES:
            continue

        # ── Dwell: use WEIGHTED dwell (stop frames count up to 3×) ───────
        # Normalise: weighted_dwell of 20% of total frames = 1.0
        dwell = min(t["weighted_dwell"] / max(total_frames * 0.20, 1), 1.0)

        # ── Saliency ──────────────────────────────────────────────────────
        sal  = float(np.mean(t["saliency"]))

        # ── Size ──────────────────────────────────────────────────────────
        area = float(np.mean(t["area_ratios"]))
        size = min(area / 0.003, 1.0)

        # ── Stop-zone score ───────────────────────────────────────────────
        # Fraction of dwell frames where vehicle was stopped or slow
        states = t["motion_states"]
        stopped_frac = sum(1 for s in states if s == "stopped") / max(len(states), 1)
        slow_frac    = sum(1 for s in states if s == "slow")    / max(len(states), 1)
        stop_zone_score = min(stopped_frac * 1.0 + slow_frac * 0.5, 1.0)

        # ── LHT position weight ───────────────────────────────────────────
        pos_c = {}
        for p in t["positions"]: pos_c[p] = pos_c.get(p, 0)+1
        dom_pos = max(pos_c, key=pos_c.get)
        lht_weight = LHT_POSITION_WEIGHT.get(dom_pos, 1.0)

        # ── Composite ─────────────────────────────────────────────────────
        comp = (w["dwell"]     * dwell
              + w["saliency"]  * sal
              + w["size"]      * size
              + w["stop_zone"] * stop_zone_score)

        # Apply LHT position multiplier
        comp *= lht_weight

        # Distance bonus/penalty
        dist_c = {}
        for d in t["distances"]: dist_c[d] = dist_c.get(d, 0)+1
        dom_dist = max(dist_c, key=dist_c.get)
        comp += {"near": -0.04, "mid": 0.05, "far": -0.02}.get(dom_dist, 0)

        # Junction / high-flow penalty
        # If most frames had high flow (fast movement), driver attention was consumed
        mean_flow = float(np.mean(t["flow_mags"]))
        if mean_flow > FLOW_MOVING:
            comp -= JUNCTION_PENALTY * min((mean_flow - FLOW_MOVING) / 20.0, 1.0)

        comp = float(np.clip(comp, 0, 1))

        vert_c = {}
        for v in t["verticals"]: vert_c[v] = vert_c.get(v, 0)+1
        dom_vert = max(vert_c, key=vert_c.get)

        q = ("Excellent" if comp > 0.65 else
             "Good"      if comp > 0.45 else
             "Fair"      if comp > 0.28 else "Poor")

        # Stop-zone label for report
        if stopped_frac > 0.5:
            stop_label = "signal/junction stop"
        elif stopped_frac + slow_frac > 0.5:
            stop_label = "slow traffic"
        else:
            stop_label = "moving traffic"

        scored.append({
            "surface_id":        f"S{t['id']:03d}",
            "composite_score":   round(comp, 3),
            "dwell_score":       round(dwell, 3),
            "weighted_dwell":    round(t["weighted_dwell"], 1),
            "saliency_score":    round(sal, 3),
            "size_score":        round(size, 3),
            "stop_zone_score":   round(stop_zone_score, 3),
            "stopped_pct":       round(stopped_frac * 100, 1),
            "slow_pct":          round(slow_frac * 100, 1),
            "stop_label":        stop_label,
            "lht_weight":        round(lht_weight, 2),
            "frame_count":       t["count"],
            "first_frame":       t["first"],
            "last_frame":        t["last"],
            "position":          dom_pos,
            "vertical":          dom_vert,
            "distance":          dom_dist,
            "aspect_ratio":      round(t.get("aspect", 1.0), 2),
            "mean_area":         round(area, 5),
            "quality":           q,
            "recommendation":    (
                f"{q} — {dom_pos} side, {dom_dist} range, {stop_label}"
            ),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(scored, 1):
        s["rank"] = i
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED FRAME WRITER
# ─────────────────────────────────────────────────────────────────────────────

# Motion state colours for frame overlay
STATE_COLOR = {
    "stopped": (0,  220, 80),   # green  — stopped = premium attention
    "slow":    (0,  180, 255),  # amber
    "moving":  (80, 80,  220),  # blue
}

def annotate(frame, surfaces, fidx, motion_state, flow_mag, stop_mult):
    out   = frame.copy()
    fh, fw = frame.shape[:2]
    thick  = max(2, fw // 700)
    fscale = max(0.5, fw / 2800)

    for s in surfaces:
        x, y, w, h = s["bbox"]
        dist  = s["distance"]
        pos   = s["position"]
        sal   = s["saliency"]
        # Box colour by distance
        color = ((0,200,80) if dist=="mid" else
                 (60,140,255) if dist=="near" else (0,120,255))
        cv2.rectangle(out, (x, y), (x+w, y+h), color, thick)
        label = f"{pos}|{dist}|sal:{sal:.2f}"
        cv2.putText(out, label, (x+3, max(y-5, int(fh*0.04))),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, color, thick-1, cv2.LINE_AA)

    # Motion state overlay — top-right corner
    sc = STATE_COLOR.get(motion_state, (180, 180, 180))
    state_txt = f"{motion_state.upper()}  flow:{flow_mag:.1f}  x{stop_mult:.1f}"
    tx = fw - int(fw * 0.38)
    cv2.putText(out, state_txt, (tx, int(fh*0.025)),
                cv2.FONT_HERSHEY_SIMPLEX, fscale*1.1, sc, thick, cv2.LINE_AA)

    # Frame info — top-left
    cv2.putText(out, f"frame {fidx} | {len(surfaces)} surfaces",
                (10, int(fh*0.025)),
                cv2.FONT_HERSHEY_SIMPLEX, fscale*1.1, (230, 230, 230), thick-1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT  — India edition
# ─────────────────────────────────────────────────────────────────────────────

def write_report(scored, total_frames, fps, video_name, mode, out_path, bbmp_stats=None):

    # Stop-zone premium section (surfaces where most dwell was stationary)
    stop_surfaces = [s for s in scored if s["stopped_pct"] > 40]

    rows = ""
    for s in scored[:25]:
        pct  = int(s["final_composite"] * 100)
        col  = "#1B5E20" if pct>65 else "#E65100" if pct>40 else "#B71C1C"
        spct = int(s["stop_zone_score"] * 100)
        bpct = int(s.get("bbmp_score", 1.0) * 100)
        bcol = "#1B5E20" if bpct >= 80 else "#E65100" if bpct >= 50 else "#B71C1C"
        all_flags = s.get("compliance_flags", []) + s.get("compliance_warnings", [])
        flags_html = ""
        if all_flags:
            tier_flag  = s.get("compliance_flags", [])
            tier_warn  = s.get("compliance_warnings", [])
            for f in tier_flag:
                short = f.split("_", 1)[0]
                flags_html += (f'<span style="background:#B71C1C;color:#fff;'
                               f'padding:1px 5px;border-radius:3px;font-size:10px;'
                               f'margin-right:2px">{short}</span>')
            for f in tier_warn:
                short = f.split("_", 1)[0]
                flags_html += (f'<span style="background:#E65100;color:#fff;'
                               f'padding:1px 5px;border-radius:3px;font-size:10px;'
                               f'margin-right:2px">~{short}</span>')
        else:
            flags_html = '<span style="color:#1B5E20;font-size:11px">✓</span>'
        rows += f"""<tr>
          <td><b>#{s['rank']}</b></td>
          <td>{s['surface_id']}</td>
          <td>
            <div style="background:#eee;border-radius:3px;height:13px;width:90px;display:inline-block">
              <div style="background:{col};width:{pct}%;height:100%;border-radius:3px"></div>
            </div>&nbsp;{s['final_composite']:.3f}
            <br><small style="color:#aaa">raw:{s['composite_score']:.3f}</small>
          </td>
          <td>{s['dwell_score']:.3f}<br><small style="color:#999">wt:{s['weighted_dwell']:.0f}</small></td>
          <td>{s['saliency_score']:.3f}</td>
          <td>
            <div style="background:#eee;border-radius:3px;height:10px;width:60px;display:inline-block">
              <div style="background:#FF6F00;width:{spct}%;height:100%;border-radius:3px"></div>
            </div>&nbsp;{s['stop_zone_score']:.2f}
          </td>
          <td>{s['stopped_pct']:.0f}%</td>
          <td>{s['lht_weight']:.2f}</td>
          <td>{s['position']}</td>
          <td>{s['distance']}</td>
          <td style="color:{col};font-weight:600">{s['final_quality']}</td>
          <td>
            <div style="background:#eee;border-radius:3px;height:8px;width:50px;display:inline-block;vertical-align:middle">
              <div style="background:{bcol};width:{bpct}%;height:100%;border-radius:3px"></div>
            </div>&nbsp;<span style="font-size:11px;color:{bcol}">{s.get('bbmp_score',1.0):.2f}</span>
            <br>{flags_html}
          </td>
        </tr>"""

    medals = ["🥇", "🥈", "🥉"]
    top3 = ""
    for i, s in enumerate(scored[:3]):
        pct  = int(s["final_composite"]*100)
        spct = int(s["stop_zone_score"]*100)
        stop_badge = ""
        if s["stopped_pct"] > 40:
            stop_badge = (f'<span style="background:#FF6F00;color:#fff;padding:2px 8px;'
                          f'border-radius:10px;font-size:11px;margin-left:8px">🛑 SIGNAL STOP ZONE</span>')
        # BBMP compliance badge
        all_flags = s.get("compliance_flags", []) + s.get("compliance_warnings", [])
        if not all_flags:
            bbmp_badge = ('<span style="background:#1B5E20;color:#fff;padding:2px 8px;'
                         'border-radius:10px;font-size:11px;margin-left:8px">✓ BBMP</span>')
        elif s.get("compliance_flags"):
            bbmp_badge = ('<span style="background:#B71C1C;color:#fff;padding:2px 8px;'
                         'border-radius:10px;font-size:11px;margin-left:8px">✗ BBMP FLAG</span>')
        else:
            bbmp_badge = ('<span style="background:#E65100;color:#fff;padding:2px 8px;'
                         'border-radius:10px;font-size:11px;margin-left:8px">⚠ BBMP WARN</span>')
        score_color = "#1B5E20" if pct > 65 else "#E65100" if pct > 40 else "#B71C1C"
        top3 += f"""
        <div style="background:#fff;border-left:5px solid #1B4F8A;border-radius:8px;
             padding:18px 22px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,0.09)">
          <div style="font-size:18px;font-weight:700;color:#1B4F8A">
            {medals[i]} {s['surface_id']}{stop_badge}{bbmp_badge}
          </div>
          <div style="font-size:32px;font-weight:800;color:{score_color};line-height:1.2">{pct}%
            <span style="font-size:14px;font-weight:400;color:#aaa">final</span>
            &nbsp;<span style="font-size:18px;color:#aaa">{int(s['composite_score']*100)}%
              <span style="font-size:11px">raw</span></span>
          </div>
          <div style="color:#555;margin:4px 0 10px">{s['recommendation']}</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
               gap:10px;font-size:13px">
            <div style="background:#F3F4F6;border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">Dwell (weighted)</div>
              <div style="font-weight:700;color:#333">{s['dwell_score']:.3f}
                <span style="font-weight:400;color:#777">(×{s['weighted_dwell']:.0f})</span></div>
            </div>
            <div style="background:#F3F4F6;border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">Saliency</div>
              <div style="font-weight:700;color:#333">{s['saliency_score']:.3f}</div>
            </div>
            <div style="background:#FFF3E0;border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">Stop-zone score</div>
              <div style="font-weight:700;color:#E65100">{s['stop_zone_score']:.3f}</div>
            </div>
            <div style="background:#F3F4F6;border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">Stopped frames</div>
              <div style="font-weight:700;color:#333">{s['stopped_pct']:.0f}%</div>
            </div>
            <div style="background:#F3F4F6;border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">LHT weight</div>
              <div style="font-weight:700;color:#333">{s['lht_weight']:.2f}×</div>
            </div>
            <div style="background:#{'EBF5EB' if not all_flags else 'FFF3E0' if not s.get('compliance_flags') else 'FEEBEE'};border-radius:6px;padding:8px 10px">
              <div style="color:#999;font-size:11px">BBMP score</div>
              <div style="font-weight:700;color:{'#1B5E20' if not all_flags else '#E65100' if not s.get('compliance_flags') else '#B71C1C'}">{s.get('bbmp_score',1.0):.3f}</div>
            </div>
          </div>
        </div>"""

    # BBMP compliance section
    bbmp_section = ""
    if bbmp_stats:
        tier_label = "Video + Metadata" if bbmp_stats.get("tier") == "metadata_checked" else "Video only (Tier 1)"
        rc = bbmp_stats.get("rule_counts", {})
        rule_rows = ""
        for code, cnt in sorted(rc.items(), key=lambda x: -x[1]):
            desc  = RULE_DESCRIPTIONS.get(code, code)
            is_t2 = code not in {"R7_junction_setback","R12_footpath_placement",
                                  "R13_road_projection","R14_vertical_stacking"}
            badge = ('<span style="background:#B71C1C;color:#fff;padding:1px 6px;'
                     'border-radius:4px;font-size:10px">CONFIRMED</span>'
                     if is_t2 else
                     '<span style="background:#E65100;color:#fff;padding:1px 6px;'
                     'border-radius:4px;font-size:10px">INFERRED</span>')
            rule_rows += f"""<tr>
              <td style="font-family:monospace;font-size:12px">{code.split('_',1)[0]}</td>
              <td>{desc}&nbsp;{badge}</td>
              <td style="text-align:center;font-weight:700">{cnt}</td>
            </tr>"""
        if not rule_rows:
            rule_rows = '<tr><td colspan="3" style="color:#1B5E20;text-align:center">No compliance issues detected</td></tr>'
        bbmp_section = f"""
        <div class="sec">
          <h2>🏛 BBMP / GBA Compliance Summary</h2>
          <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">
            <div class="card" style="border-top:3px solid #1B5E20">
              <div class="v" style="color:#1B5E20">{bbmp_stats.get('eligible',0)}</div>
              <div class="l">Eligible surfaces</div></div>
            <div class="card" style="border-top:3px solid #B71C1C">
              <div class="v" style="color:#B71C1C">{bbmp_stats.get('flagged',0)}</div>
              <div class="l">Flagged surfaces</div></div>
            <div class="card">
              <div class="v">{tier_label}</div>
              <div class="l">Compliance tier</div></div>
          </div>
          <table style="margin-bottom:8px"><thead><tr>
            <th style="width:110px">Rule</th><th>Description</th><th style="width:70px">Surfaces</th>
          </tr></thead><tbody>{rule_rows}</tbody></table>
          <p style="font-size:11px;color:#888;margin-top:6px">
            <b>CONFIRMED</b> — from supplied location metadata (Tier 2, higher confidence).&nbsp;
            <b>INFERRED</b> — derived from video analysis alone (Tier 1, lower confidence).&nbsp;
            Penalties are soft: non-compliant surfaces remain visible with a reduced final score.
          </p>
        </div>"""

    # Stop-zone premium section
    stop_section = ""
    if stop_surfaces:
        stop_rows = ""
        for s in stop_surfaces[:10]:
            stop_rows += f"""<tr>
              <td><b>#{s['rank']}</b></td><td>{s['surface_id']}</td>
              <td>{s['composite_score']:.3f}</td>
              <td>{s['stopped_pct']:.0f}%</td>
              <td>{s['stop_label']}</td>
              <td>{s['position']}</td>
              <td>{s['distance']}</td>
              <td style="color:#E65100;font-weight:600">{s['quality']}</td>
            </tr>"""
        stop_section = f"""
        <div class="sec">
          <h2>🛑 Stop-Zone Premium Surfaces</h2>
          <p style="font-size:13px;color:#555;margin-bottom:12px">
            Surfaces where the vehicle was stationary for &gt;40% of dwell time —
            signal junctions, railway crossings, traffic congestion.
            These have the highest actual attention value despite potentially
            lower raw dwell count.
          </p>
          <table><thead><tr>
            <th>Rank</th><th>ID</th><th>Score</th><th>Stopped%</th>
            <th>Context</th><th>Position</th><th>Distance</th><th>Quality</th>
          </tr></thead><tbody>{stop_rows}</tbody></table>
        </div>"""

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Billboard Placement Report — India</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,sans-serif;background:#F5F7FA;color:#222;line-height:1.5}}
.hdr{{background:linear-gradient(135deg,#1B4F8A,#0D3460);color:#fff;padding:24px 36px}}
.hdr h1{{font-size:22px;font-weight:700}}
.hdr p{{opacity:.85;margin-top:4px;font-size:13px}}
.badge{{display:inline-block;background:rgba(255,255,255,.2);border-radius:12px;
  padding:2px 10px;font-size:12px;margin-left:8px}}
.wrap{{max-width:1120px;margin:0 auto;padding:28px 18px}}
.cards{{display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:10px;padding:14px 18px;flex:1;min-width:130px;
  box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card .v{{font-size:22px;font-weight:700;color:#1B4F8A}}
.card .l{{font-size:11px;color:#999;margin-top:2px}}
.card.stop .v{{color:#E65100}}
h2{{font-size:16px;color:#1B4F8A;margin-bottom:12px;padding-bottom:6px;
  border-bottom:2px solid #D6E4F0}}
.sec{{margin-bottom:32px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:12.5px}}
th{{background:#1B4F8A;color:#fff;padding:9px 10px;text-align:left;font-weight:600}}
td{{padding:8px 10px;border-bottom:1px solid #f0f0f0;vertical-align:middle}}
tr:last-child td{{border:none}} tr:hover td{{background:#F0F7FF}}
.note{{background:#E8F4FD;border-left:4px solid #1B4F8A;border-radius:4px;
  padding:14px 18px;font-size:13px;color:#444;margin-top:18px;line-height:1.7}}
.note b{{color:#1B4F8A}}
</style></head><body>

<div class="hdr">
  <h1>Billboard Placement Analysis
    <span class="badge">🇮🇳 India LHT</span>
    <span class="badge">{mode.upper()} MODE</span>
  </h1>
  <p>{video_name} &nbsp;·&nbsp; {total_frames} frames @ {fps} fps
     &nbsp;·&nbsp; {len(scored)} surfaces ranked</p>
</div>

<div class="wrap">
  <div class="cards">
    <div class="card"><div class="v">{len(scored)}</div>
      <div class="l">Surfaces ranked</div></div>
    <div class="card"><div class="v">{total_frames}</div>
      <div class="l">Frames analysed</div></div>
    <div class="card"><div class="v">{scored[0]['composite_score'] if scored else 0:.2f}</div>
      <div class="l">Top score</div></div>
    <div class="card stop"><div class="v">{len(stop_surfaces)}</div>
      <div class="l">Stop-zone surfaces</div></div>
    <div class="card"><div class="v">{mode.title()}</div>
      <div class="l">Viewer mode</div></div>
    <div class="card" style="border-top:3px solid #1B5E20">
      <div class="v" style="color:#1B5E20">{bbmp_stats.get('eligible', len(scored)) if bbmp_stats else len(scored)}</div>
      <div class="l">BBMP Eligible</div></div>
  </div>

  <div class="sec"><h2>Top 3 Recommended Locations</h2>{top3}</div>

  {bbmp_section}

  {stop_section}

  <div class="sec">
    <h2>All Scored Surfaces (top 25)</h2>
    <table><thead><tr>
      <th>Rank</th><th>ID</th><th>Score</th><th>Dwell</th><th>Saliency</th>
      <th>Stop-zone</th><th>Stopped%</th><th>LHT×</th>
      <th>Position</th><th>Distance</th><th>Quality</th><th>BBMP</th>
    </tr></thead><tbody>{rows}</tbody></table>
  </div>

  <div class="note">
    <b>India-specific scoring explained</b><br><br>
    <b>Weighted dwell</b> — each frame's dwell contribution is multiplied by the
    vehicle's motion state: stopped (×3.0), slow crawl (×1.6), moving (×1.0).
    A surface seen during a 60-second signal stop scores far higher than the same
    surface passed at speed.<br><br>
    <b>Stop-zone score</b> — fraction of dwell time when vehicle was stationary or
    slow. The stop-zone premium surfaces table highlights locations where this is
    consistently high — these are signal junctions, railway crossings, and
    congestion points.<br><br>
    <b>LHT position weight</b> — in left-hand traffic (India), the driver sits
    on the right and naturally glances left toward the near-side footpath and
    buildings. Left-side surfaces receive a 1.18× multiplier; right-side (far/
    median side) surfaces receive 0.82×.<br><br>
    <b>Junction penalty</b> — surfaces predominantly seen during fast movement
    (high optical flow) receive a cognitive-load penalty, because driver attention
    is consumed by navigation rather than available for advertising.
  </div>
</div></body></html>"""

    with open(out_path, "w") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(video_path, fps=4.0, mode="driver", top_n=10,
        save_frames=True, output_dir=None, bbmp_meta=None):

    vp  = Path(video_path)
    if not vp.exists():
        print(f"Error: video not found — {vp}"); sys.exit(1)

    out = Path(output_dir) if output_dir else Path("billboard_output") / vp.stem
    out.mkdir(parents=True, exist_ok=True)
    ann = out / "annotated_frames"
    if save_frames:
        ann.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Billboard Analyzer — India Edition  🇮🇳")
    print(f"{'='*60}")
    print(f"  Video    : {vp.name}")
    print(f"  Mode     : {mode}  |  FPS : {fps}")
    print(f"  Traffic  : Left-hand (LHT)  |  Drive side: Right")
    print(f"  Output   : {out}")
    print(f"{'='*60}\n")

    cap      = cv2.VideoCapture(str(vp))
    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(vid_fps / fps))
    est      = total // interval

    print(f"  {fw}×{fh} @ {vid_fps:.0f}fps  →  sample every {interval} frames  (~{est} samples)\n")

    motion   = MotionAnalyzer()
    tracker  = Tracker()
    vi = pi  = 0
    t0       = time.time()

    # Counters for motion state summary
    state_counts = {"stopped": 0, "slow": 0, "moving": 0}

    while True:
        ret, frame = cap.read()
        if not ret: break

        if vi % interval == 0:
            excl, flow_mag, m_state, stop_mult = motion.process(frame)
            state_counts[m_state] = state_counts.get(m_state, 0) + 1

            surfs = detect_surfaces(frame, excl, fw, fh)
            tracker.update(pi, surfs,
                           motion_state=m_state,
                           stop_mult=stop_mult,
                           flow_mag=flow_mag)

            if save_frames and pi % 5 == 0:
                ann_f  = annotate(frame, surfs, pi, m_state, flow_mag, stop_mult)
                h2, w2 = ann_f.shape[:2]
                small  = cv2.resize(ann_f, (w2//2, h2//2))
                cv2.imwrite(str(ann / f"frame_{pi:04d}.jpg"), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])

            if pi % 10 == 0:
                el  = time.time() - t0
                eta = max(0, (est-pi) / max(pi / max(el, .1), .1))
                print(f"  [{pi:4d}/{est}]  {len(surfs):3d} surfs  "
                      f"{m_state:8s} (flow:{flow_mag:5.1f})  "
                      f"mult:{stop_mult:.1f}×  ETA:{eta:.0f}s")
            pi += 1

        vi += 1

    cap.release()
    tracker.finalise()

    elapsed = time.time() - t0
    total_s  = sum(state_counts.values()) or 1
    print(f"\n  Done: {pi} frames in {elapsed:.1f}s")
    print(f"  Motion breakdown:")
    print(f"    Stopped : {state_counts['stopped']:4d} frames "
          f"({100*state_counts['stopped']//total_s:2d}%)")
    print(f"    Slow    : {state_counts['slow']:4d} frames "
          f"({100*state_counts['slow']//total_s:2d}%)")
    print(f"    Moving  : {state_counts['moving']:4d} frames "
          f"({100*state_counts['moving']//total_s:2d}%)")
    print(f"  Raw tracks: {len(tracker.finished)}")

    scored = score_tracks(tracker.finished, pi, mode)
    print(f"  Scored (≥{MIN_FRAMES} frames): {len(scored)}\n")

    if not scored:
        print("No surfaces detected. Try --fps 4 or check video path.")
        sys.exit(0)

    # Apply BBMP / GBA compliance layer
    scored = apply_bbmp_compliance(scored, bbmp_meta)
    bbmp_stats = compliance_summary(scored)

    tier_label = bbmp_stats.get("tier", "video_only")
    print(f"  BBMP compliance ({tier_label}): "
          f"{bbmp_stats.get('eligible',0)}/{bbmp_stats.get('total_surfaces',0)} eligible\n")

    result = {
        "video":            str(vp),
        "mode":             mode,
        "traffic_side":     "LHT",
        "frames_processed": pi,
        "fps_used":         fps,
        "motion_summary":   state_counts,
        "bbmp_compliance":  bbmp_stats,
        "surfaces":         scored,
    }
    with open(out / "ranked_surfaces.json", "w") as f:
        json.dump(result, f, indent=2)
    write_report(scored, pi, fps, vp.name, mode, str(out / "report.html"), bbmp_stats)

    # Print summary
    stop_count = sum(1 for s in scored if s["stopped_pct"] > 40)
    print(f"{'─'*60}")
    print(f"  TOP {min(top_n, len(scored))}  ({mode.upper()} MODE — INDIA LHT)")
    print(f"{'─'*60}")
    for s in scored[:top_n]:
        bar    = "█" * int(s["final_composite"] * 24)
        stop_f = "🛑" if s["stopped_pct"] > 40 else "  "
        all_flags = s.get("compliance_flags", []) + s.get("compliance_warnings", [])
        bbmp_f = ""
        if s.get("compliance_flags"):
            bbmp_f = f"  [✗ {','.join(s['compliance_flags'])}]"
        elif s.get("compliance_warnings"):
            bbmp_f = f"  [⚠ {','.join(s['compliance_warnings'])}]"
        print(f"  {stop_f} #{s['rank']:2d}  {s['surface_id']}  "
              f"{bar:<24}  {s['final_composite']:.3f}"
              f" (raw:{s['composite_score']:.3f})"
              f"  {s['final_quality']}{bbmp_f}")

    print(f"\n  🛑 Stop-zone premium surfaces: {stop_count}")
    print(f"\n{'='*60}")
    print(f"  ranked_surfaces.json  |  report.html  |  annotated_frames/")
    print(f"{'='*60}\n")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Billboard Placement Analyzer — India Edition (LHT)")
    ap.add_argument("--video",     required=True,
                    help="Path to dashcam video (MP4/MOV/AVI)")
    ap.add_argument("--fps",       type=float, default=4.0,
                    help="Frames/sec to sample. Default 4 (recommended for Indian city speeds)")
    ap.add_argument("--mode",      choices=["driver", "pedestrian"], default="driver")
    ap.add_argument("--top",       type=int, default=10)
    ap.add_argument("--no-frames", action="store_true",
                    help="Skip saving annotated frames")
    ap.add_argument("--output",    default=None,
                    help="Output directory")
    # ── BBMP / GBA compliance metadata (all optional) ─────────────────────────
    ap.add_argument("--road-width-meters",    type=float, default=None,
                    help="Carriageway width in metres (enables R1 road-width check)")
    ap.add_argument("--road-name",            type=str,   default=None,
                    help="Road name for heritage corridor check (R19)")
    ap.add_argument("--road-type",
                    choices=["normal", "flyover", "railway", "sharp_curve"],
                    default="normal",
                    help="Road type — flyover/railway/sharp_curve trigger absolute prohibitions")
    ap.add_argument("--nearest-hoarding-m",   type=float, default=None,
                    help="Distance in metres to nearest existing hoarding (R6 spacing check)")
    ap.add_argument("--nearest-religious-site-m", type=float, default=None,
                    help="Distance in metres to nearest religious institution (R8)")
    args = ap.parse_args()

    bbmp_meta = BBMPMetadata(
        road_width_meters          = args.road_width_meters,
        road_name                  = args.road_name,
        road_type                  = args.road_type,
        nearest_hoarding_m         = args.nearest_hoarding_m,
        nearest_religious_site_m   = args.nearest_religious_site_m,
    )

    run(args.video, args.fps, args.mode, args.top,
        not args.no_frames, args.output, bbmp_meta)
