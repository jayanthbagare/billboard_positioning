"""
BBMP / GBA Outdoor Advertising Compliance Layer
================================================
Sources:
  - BBMP Advertisement Bye-Laws, 2024 (notified July 2025)
  - Greater Bengaluru Area (Advertisement) Draft Rules, 2025

Architecture
────────────
Tier 1: Video-derived checks — run always; lower confidence, stored as warnings.
Tier 2: Metadata-assisted checks — run when BBMPMetadata is supplied; stored as flags.

Penalty formula
───────────────
    bbmp_score      = max(0, 1 - Σ rule_penalties)
    final_composite = composite × (1 - BBMP_BLEND_WEIGHT × (1 - bbmp_score))

At BBMP_BLEND_WEIGHT = 0.35, a fully non-compliant surface (bbmp_score = 0) loses
35% of its composite score; a fully compliant one is unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Penalty per violated rule.  Aggregated as: bbmp_score = 1 − Σ penalties.
RULE_PENALTIES: dict[str, float] = {
    "R1_road_width_below_18m":  0.30,   # < 18 m road: no third-party ads
    "R6_hoarding_spacing":      0.20,   # < 175 m from nearest hoarding
    "R7_junction_setback":      0.20,   # inferred: within 30 ft of junction/circle
    "R8_religious_buffer":      0.25,   # < 25 m from religious site / 100 m access road
    "R10_flyover_rail":         0.40,   # on flyover or near railway overpass
    "R11_sharp_curve":          0.25,   # at a sharp road curve
    "R12_footpath_placement":   0.40,   # hoarding on/over footpath (inferred)
    "R13_road_projection":      0.40,   # projecting over public road (inferred)
    "R14_vertical_stacking":    0.35,   # two hoardings one above another
    "R19_heritage_corridor":    0.45,   # named heritage / protected corridor
}

# Compliance can drag composite by at most this fraction (soft-penalty cap).
BBMP_BLEND_WEIGHT: float = 0.35

# R1 — minimum carriageway width (BBMP 2024)
MIN_ROAD_WIDTH_M: float = 18.0

# R6 — minimum inter-hoarding spacing (BBMP 2024)
MIN_HOARDING_SPACING_M: float = 175.0

# R7 — stopped_pct threshold above which junction proximity is inferred
JUNCTION_STOPPED_PCT_THRESHOLD: float = 60.0

# R8 / R9 — religious institution buffers (GBA 2025)
RELIGIOUS_SITE_BUFFER_M: float = 25.0
RELIGIOUS_ACCESS_BUFFER_M: float = 100.0

# R19 — heritage / protected corridors (GBA 2025, absolute ban on third-party ads)
HERITAGE_CORRIDORS: frozenset[str] = frozenset({
    "kumara krupa road",
    "sankey road",
    "ambedkar veedhi",
    "palace road",
    "cubbon park",
    "lalbagh",
    "nrupathunga road",
    "maharani college road",
})

# Human-readable rule descriptions for the HTML report legend
RULE_DESCRIPTIONS: dict[str, str] = {
    "R1_road_width_below_18m":  "Road width < 18 m — third-party ads prohibited (BBMP 2024)",
    "R6_hoarding_spacing":      "Nearest hoarding < 175 m — minimum spacing violated (BBMP 2024)",
    "R7_junction_setback":      "Inferred junction proximity — 30 ft setback required (BBMP 2006)",
    "R8_religious_buffer":      "Within 25 m of religious site or 100 m of access road (GBA 2025)",
    "R10_flyover_rail":         "On flyover / near railway overpass — ads prohibited (GBA 2025)",
    "R11_sharp_curve":          "At sharp curve — ads prohibited for road safety (GBA 2025)",
    "R12_footpath_placement":   "Inferred footpath / low-level placement — ads on footpath prohibited",
    "R13_road_projection":      "Inferred road projection — hoardings must not overhang road",
    "R14_vertical_stacking":    "Possible vertical stacking — one hoarding above another prohibited",
    "R19_heritage_corridor":    "Heritage / protected corridor — third-party ads absolutely prohibited",
}


# ─────────────────────────────────────────────────────────────────────────────
# METADATA INPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BBMPMetadata:
    """
    Optional location metadata that enables Tier 2 compliance checks.
    Any field left as None simply skips that rule.
    """
    road_width_meters:          Optional[float] = None
    road_name:                  Optional[str]   = None
    road_type:                  str             = "normal"  # normal | flyover | railway | sharp_curve
    nearest_hoarding_m:         Optional[float] = None
    nearest_religious_site_m:   Optional[float] = None
    nearest_religious_access_m: Optional[float] = None

    def has_any(self) -> bool:
        return any([
            self.road_width_meters is not None,
            self.road_name is not None,
            self.road_type != "normal",
            self.nearest_hoarding_m is not None,
            self.nearest_religious_site_m is not None,
            self.nearest_religious_access_m is not None,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# RULE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _tier1_checks(surface: dict) -> list[str]:
    """
    Tier 1: infer rule violations from video-derived surface fields alone.
    Returns warning codes (lower confidence than Tier 2 flags).
    """
    warnings: list[str] = []

    # R12: footpath placement — lower vertical zone signals low / footpath-level mount
    if surface.get("vertical") == "lower":
        warnings.append("R12_footpath_placement")

    # R13: road projection — large centre-positioned surface suggests road overhang
    if surface.get("position") == "centre" and surface.get("mean_area", 0) > 0.04:
        warnings.append("R13_road_projection")

    # R7: junction setback — high stopped% means vehicle is consistently at a junction,
    # implying the hoarding may be within the 30 ft prohibited setback zone
    if surface.get("stopped_pct", 0) >= JUNCTION_STOPPED_PCT_THRESHOLD:
        warnings.append("R7_junction_setback")

    return warnings


def _tier2_checks(surface: dict, meta: BBMPMetadata) -> list[str]:
    """Tier 2: rule violations confirmed by explicit location metadata."""
    flags: list[str] = []

    if meta.road_width_meters is not None and meta.road_width_meters < MIN_ROAD_WIDTH_M:
        flags.append("R1_road_width_below_18m")

    if meta.road_type in ("flyover", "railway"):
        flags.append("R10_flyover_rail")

    if meta.road_type == "sharp_curve":
        flags.append("R11_sharp_curve")

    if meta.road_name:
        name_lc = meta.road_name.lower().strip()
        if any(h in name_lc for h in HERITAGE_CORRIDORS):
            flags.append("R19_heritage_corridor")

    if meta.nearest_hoarding_m is not None and meta.nearest_hoarding_m < MIN_HOARDING_SPACING_M:
        flags.append("R6_hoarding_spacing")

    if (meta.nearest_religious_site_m is not None
            and meta.nearest_religious_site_m < RELIGIOUS_SITE_BUFFER_M):
        flags.append("R8_religious_buffer")

    if (meta.nearest_religious_access_m is not None
            and meta.nearest_religious_access_m < RELIGIOUS_ACCESS_BUFFER_M):
        if "R8_religious_buffer" not in flags:
            flags.append("R8_religious_buffer")

    return flags


def _stacking_check(surfaces: list[dict]) -> None:
    """
    R14: detect likely vertical stacking across scored surfaces.
    Conditions: same horizontal position, overlapping frame windows,
    same distance, different vertical zones → both surfaces get an R14 warning.
    Mutates compliance_warnings in-place.
    """
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            s1, s2 = surfaces[i], surfaces[j]
            if s1["position"] != s2["position"]:
                continue
            if s1["vertical"] == s2["vertical"]:
                continue
            if s1.get("distance") != s2.get("distance"):
                continue
            # Require overlapping observation windows
            if s1["last_frame"] < s2["first_frame"] or s2["last_frame"] < s1["first_frame"]:
                continue
            for s in (s1, s2):
                if "R14_vertical_stacking" not in s["compliance_warnings"]:
                    s["compliance_warnings"].append("R14_vertical_stacking")


def _compute_bbmp_score(flags: list[str], warnings: list[str]) -> float:
    penalty = sum(RULE_PENALTIES.get(c, 0.0) for c in flags + warnings)
    return round(max(0.0, 1.0 - penalty), 3)


def _compute_final_composite(composite: float, bbmp_score: float) -> float:
    return round(float(max(0.0, min(1.0,
        composite * (1.0 - BBMP_BLEND_WEIGHT * (1.0 - bbmp_score))
    ))), 3)


def _quality_label(score: float) -> str:
    if score > 0.65: return "Excellent"
    if score > 0.45: return "Good"
    if score > 0.28: return "Fair"
    return "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def apply_bbmp_compliance(
    scored_surfaces: list[dict],
    metadata: Optional[BBMPMetadata] = None,
) -> list[dict]:
    """
    Apply BBMP/GBA compliance checks to scored surfaces.

    Adds to each surface dict:
      bbmp_score          float [0,1] — 1.0 = fully compliant
      bbmp_eligible       bool — False only when Tier 2 hard flags are present
      compliance_flags    list[str] — Tier 2 confirmed violations
      compliance_warnings list[str] — Tier 1 inferred violations (lower confidence)
      compliance_tier     "video_only" | "metadata_checked"
      final_composite     float — composite penalised by BBMP score
      final_quality       str — quality label based on final_composite

    Re-sorts and re-ranks surfaces by final_composite.
    """
    has_meta = metadata is not None and metadata.has_any()
    tier = "metadata_checked" if has_meta else "video_only"

    for surface in scored_surfaces:
        surface["compliance_flags"]    = _tier2_checks(surface, metadata) if metadata else []
        surface["compliance_warnings"] = _tier1_checks(surface)
        surface["compliance_tier"]     = tier

    _stacking_check(scored_surfaces)

    for surface in scored_surfaces:
        bs = _compute_bbmp_score(surface["compliance_flags"], surface["compliance_warnings"])
        fc = _compute_final_composite(surface["composite_score"], bs)
        surface["bbmp_score"]      = bs
        surface["bbmp_eligible"]   = len(surface["compliance_flags"]) == 0
        surface["final_composite"] = fc
        surface["final_quality"]   = _quality_label(fc)

    scored_surfaces.sort(key=lambda s: s["final_composite"], reverse=True)
    for i, s in enumerate(scored_surfaces, 1):
        s["rank"] = i

    return scored_surfaces


def compliance_summary(scored_surfaces: list[dict]) -> dict:
    """Aggregate compliance stats used for reporting and JSON output."""
    total = len(scored_surfaces)
    if not total:
        return {}
    eligible = sum(1 for s in scored_surfaces if s.get("bbmp_eligible", True))
    rule_counts: dict[str, int] = {}
    for s in scored_surfaces:
        for code in s.get("compliance_flags", []) + s.get("compliance_warnings", []):
            rule_counts[code] = rule_counts.get(code, 0) + 1
    return {
        "total_surfaces": total,
        "eligible":       eligible,
        "flagged":        total - eligible,
        "rule_counts":    rule_counts,
        "tier":           scored_surfaces[0].get("compliance_tier", "video_only"),
    }
