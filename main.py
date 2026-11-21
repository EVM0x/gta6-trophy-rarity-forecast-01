#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates as five quantiles at a future target.

Reads every *.in case in $MAKEFUN_INPUT_DIR (default /input/), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output/). Standard library only,
no network, no randomness, no clock: the same input always gives the same
bytes.

Model, per trophy, all in logit space:
  * collapse the 10-minute read grid into Exophase refresh events (a read
    whose rate vector differs from the previous kept read),
  * weighted least-squares slope of logit(rate) on u(t) = ln(1 + hours since
    2026-11-19T00:00:00Z) over the last 24 h, weights exp(-age/24h),
  * shrink that slope toward the median slope of the trophy's tier,
  * extrapolate to the target, damped and capped so a 4.5-day horizon cannot
    blow up,
  * a bounded denominator nudge when players_tracked is growing,
  * spread from the horizon plus the trophy's own fit residual,
  * quantiles at z = (-1.645, -0.674, 0, 0.674, 1.645).
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

# --- constants -------------------------------------------------------------

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
TIERS = ("platinum", "gold", "silver", "bronze")

HIST_WINDOW_H = 24.0     # slope window, hours before the trophy's last read
SHRINK_K = 3.0           # pseudo-count pulling a trophy's slope to its tier
DAMP_RATE = 0.35         # damping strength beyond DAMP_FROM log-units of du
DAMP_FROM = 1.0
STEP_CAP = 1.5           # |b_eff * du| cap, logits
S0 = 0.30                # base logit spread
RESID_CAP = 0.6          # cap on a trophy's own residual contribution
RESID_DEFAULT = 0.25     # residual for a trophy with no usable fit
RESID_FLOOR = 0.12       # a fit residual never claims less model error than this
RESID_PRIOR = 2.0        # pseudo-count shrinking a residual toward its tier's

USE_DENOM = True         # switch for the players_tracked denominator term
DENOM_WEIGHT = 0.35
DENOM_CAP = 1.0
DENOM_MIN_PTS = 3        # refreshes needed before the denominator slope is used
DENOM_MAX_SLOPE = 3.0    # |d ln N / du| cap, so one bad count cannot swing it
DENOM_SLOW_LO = -0.10    # "rate still climbing slowly": step in [LO, HI]
DENOM_SLOW_HI = 0.45

RATE_LO = 0.01
RATE_HI = 99.99
PLATEAU_100 = 99.995     # a trophy printed at 100.00 stays at 100

# --- small helpers ---------------------------------------------------------


def num(x):
    """A finite float, or None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_ts(s):
    if not isinstance(s, str) or not s:
        return None
    txt = s.strip()
    if txt.endswith("Z") or txt.endswith("z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def u_of(dt):
    """ln(1 + hours since the release epoch); guarded against tiny arguments."""
    hours = (dt - EPOCH).total_seconds() / 3600.0
    return math.log(max(1e-6, 1.0 + hours))


def clamp_rate(x):
    if x is None:
        return None
    return min(RATE_HI, max(RATE_LO, x))


def logit_rate(x):
    """logit of a rate in percent, clamped to 0.01..99.99."""
    c = clamp_rate(x)
    if c is None:
        c = RATE_LO
    p = c / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(z):
    if z >= 40.0:
        return 100.0
    if z <= -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-z))


def round4(x):
    if not math.isfinite(x):
        return 0.0
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def finalize(qs):
    """Clamp to 0..100, round half-up to 4 dp, force non-decreasing."""
    vals = []
    for q in qs:
        v = q if math.isfinite(q) else 0.0
        v = min(100.0, max(0.0, v))
        vals.append(round4(v))
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return vals


def id_key(value):
    """A hashable key for one trophy id.

    The output rule is one row per id, so an id has to be usable as a set
    member. A case file can carry a list or an object in that slot, which is
    not hashable, and the TypeError that raised used to cost the whole case
    its output rather than the one malformed trophy its row.
    """
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    try:
        return ("json", json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return ("repr", repr(value))


def median(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def fit_slope(points, t_last, window_h=HIST_WINDOW_H):
    """Weighted LS slope of y on u over the window; returns (slope, rms).

    points: [(datetime, u, y)].  Falls back to every point when the window
    holds fewer than two.  Returns (None, None) when no slope is defined.
    """
    sel = [p for p in points if (t_last - p[0]).total_seconds() <= window_h * 3600.0]
    if len(sel) < 2:
        sel = points
    if len(sel) < 2:
        return None, None
    ws = [math.exp(-max(0.0, (t_last - p[0]).total_seconds()) / 86400.0) for p in sel]
    sw = sum(ws)
    if sw <= 0.0:
        return None, None
    ub = sum(w * p[1] for w, p in zip(ws, sel)) / sw
    yb = sum(w * p[2] for w, p in zip(ws, sel)) / sw
    sxx = sum(w * (p[1] - ub) ** 2 for w, p in zip(ws, sel))
    if sxx <= 1e-12:
        return None, None
    sxy = sum(w * (p[1] - ub) * (p[2] - yb) for w, p in zip(ws, sel))
    b = sxy / sxx
    a = yb - b * ub
    ss = sum(w * (p[2] - (a + b * p[1])) ** 2 for w, p in zip(ws, sel))
    return b, math.sqrt(max(0.0, ss / sw))


# --- case handling ---------------------------------------------------------


def collapse(history, n_trophies):
    """Keep the first read and every read whose rate vector differs from the
    previous kept one: those are Exophase refresh events."""
    kept = []
    prev = None
    for h in history:
        if not isinstance(h, dict):
            continue
        # the timestamp gate comes first: a read with no usable time must not
        # become the baseline the next read is compared against
        t = parse_ts(h.get("at")) or parse_ts(h.get("slot"))
        if t is None:
            continue
        rates = h.get("rates")
        if not isinstance(rates, list):
            rates = []
        row = []
        for i in range(n_trophies):
            row.append(num(rates[i]) if i < len(rates) else None)
        key = tuple(row)
        if prev is not None and key == prev:
            continue
        kept.append({
            "t": t,
            "u": u_of(t),
            "n": num(h.get("players_tracked")),
            "rates": row,
        })
        prev = key
    return kept


def trophy_series(kept, idx):
    pts = []
    last_rate = None
    last_u = None
    last_t = None
    for r in kept:
        v = r["rates"][idx] if idx < len(r["rates"]) else None
        if v is None:
            continue
        pts.append((r["t"], r["u"], logit_rate(v)))
        last_rate = v
        last_u = r["u"]
        last_t = r["t"]
    return pts, last_rate, last_u, last_t


def forecast_case(case):
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    history = case.get("history")
    if not isinstance(history, list):
        history = []

    target = parse_ts(case.get("target"))
    at = parse_ts(case.get("at"))
    horizon_min = num(case.get("horizon_minutes"))
    if horizon_min is None:
        if target is not None and at is not None:
            horizon_min = (target - at).total_seconds() / 60.0
        else:
            horizon_min = 0.0
    horizon_days = max(0.0, horizon_min) / 1440.0
    fallback_sigma = 0.5 * math.sqrt(max(0.0, horizon_min) / 1440.0)

    kept = collapse(history, len(trophies))

    # ---- per-trophy fits --------------------------------------------------
    rows = []
    for idx, tr in enumerate(trophies):
        tier = ""
        if isinstance(tr, dict):
            t = tr.get("tier")
            if isinstance(t, str):
                tier = t.strip().lower()
        pts, last_rate, last_u, last_t = trophy_series(kept, idx)
        slope = None
        resid = None
        if len(pts) >= 2:
            t_ref = last_t if last_t is not None else pts[-1][0]
            slope, resid = fit_slope(pts, t_ref)
        rows.append({
            "id": tr.get("id") if isinstance(tr, dict) else None,
            "tier": tier,
            "n": len(pts),
            "slope": slope,
            "resid": resid,
            "last_rate": last_rate,
            "last_u": last_u,
            "last_t": last_t,
        })

    # ---- tier shrinkage ---------------------------------------------------
    tier_slopes = {}
    for tier in TIERS:
        vals = [r["slope"] for r in rows
                if r["tier"] == tier and r["slope"] is not None and r["n"] >= 2]
        tier_slopes[tier] = median(vals)
    all_slopes = [r["slope"] for r in rows if r["slope"] is not None and r["n"] >= 2]
    global_slope = median(all_slopes)
    if global_slope is None:
        global_slope = 0.0
    for tier in TIERS:
        if tier_slopes[tier] is None:
            tier_slopes[tier] = global_slope

    tier_resids = {}
    for tier in TIERS:
        vals = [r["resid"] for r in rows
                if r["tier"] == tier and r["resid"] is not None]
        tier_resids[tier] = median(vals)
    all_resids = [r["resid"] for r in rows if r["resid"] is not None]
    global_resid = median(all_resids)

    # ---- denominator: is players_tracked visibly growing? -----------------
    n_pts = [(r["t"], r["u"], math.log(r["n"]))
             for r in kept if r["n"] is not None and r["n"] > 0.0]
    denom_slope = None
    if (len(n_pts) >= DENOM_MIN_PTS
            and kept[-1]["n"] is not None and kept[0]["n"] is not None):
        if kept[-1]["n"] > kept[0]["n"] * 1.05:
            t_ref = n_pts[-1][0]
            denom_slope, _ = fit_slope(n_pts, t_ref)
            if denom_slope is not None and denom_slope <= 0.0:
                denom_slope = None
            if denom_slope is not None:
                denom_slope = min(DENOM_MAX_SLOPE, denom_slope)

    u_target = u_of(target) if target is not None else None

    # ---- forecast ---------------------------------------------------------
    out_rows = []
    for r in rows:
        last_rate = r["last_rate"]
        if last_rate is None or r["last_u"] is None or u_target is None:
            base = last_rate if last_rate is not None else RATE_LO
            qs = [pct_of_logit(logit_rate(base) + z * fallback_sigma) for z in Z]
            if last_rate is not None and last_rate >= PLATEAU_100:
                qs = [100.0] * 5
            out_rows.append({"id": r["id"], "q": finalize(qs)})
            continue

        du = u_target - r["last_u"]
        tier = r["tier"] if r["tier"] in tier_slopes else ""
        b_tier = tier_slopes[tier] if tier in tier_slopes else global_slope
        if r["slope"] is not None and r["n"] >= 2:
            b_eff = (r["n"] * r["slope"] + SHRINK_K * b_tier) / (r["n"] + SHRINK_K)
        else:
            b_eff = b_tier

        # damp the slope for long extrapolations, then cap the step
        b_damped = b_eff * math.exp(-DAMP_RATE * max(0.0, du - DAMP_FROM))
        step = b_damped * du
        step = max(-STEP_CAP, min(STEP_CAP, step))
        m = logit_rate(last_rate) + step

        # denominator nudge, only for a rate that is still climbing slowly
        if (USE_DENOM and denom_slope is not None
                and DENOM_SLOW_LO <= step <= DENOM_SLOW_HI
                and RATE_LO < last_rate < 99.5):
            growth = denom_slope * du
            growth = DENOM_CAP * math.tanh(growth / DENOM_CAP)
            if growth > 0.0:
                m -= DENOM_WEIGHT * growth

        resid = r["resid"]
        tier_resid = tier_resids[tier] if tier in tier_resids else None
        if tier_resid is None:
            tier_resid = global_resid
        if resid is None:
            resid = tier_resid
        elif tier_resid is not None:
            # shrink a trophy's own residual toward its tier's median one, so a
            # two-point fit cannot pass for certainty
            k = float(r["n"])
            resid = math.sqrt((k * resid * resid + RESID_PRIOR * tier_resid * tier_resid)
                              / (k + RESID_PRIOR))
        if resid is None:
            resid = RESID_DEFAULT
        if r["n"] >= 2:
            resid = max(resid, RESID_FLOOR)
        resid = min(RESID_CAP, max(0.0, resid))
        du_pos = max(0.0, du)
        sigma = math.sqrt(S0 * S0 + S0 * S0 * du_pos + resid * resid)

        qs = [pct_of_logit(m + z * sigma) for z in Z]
        if last_rate >= PLATEAU_100:
            qs = [100.0] * 5
        out_rows.append({"id": r["id"], "q": finalize(qs)})

    # one row per trophy id, each id once, case order preserved
    seen = set()
    uniq = []
    for row in out_rows:
        key = id_key(row["id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return {"trophies": uniq}


def process(path):
    with open(path, "r", encoding="utf-8") as fh:
        case = json.load(fh)
    if not isinstance(case, dict):
        case = {}
    return forecast_case(case)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    paths = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in paths:
        stem = os.path.basename(path)[:-3]
        try:
            payload = process(path)
        except Exception:
            payload = {"trophies": []}
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
        try:
            with open(os.path.join(out_dir, stem + ".out"), "w",
                      encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
