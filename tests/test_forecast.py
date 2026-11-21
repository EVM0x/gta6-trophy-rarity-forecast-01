"""Tests for the trophy-rarity forecaster.

The module promises three things in its docstring — standard library only, no
network, no randomness, no clock, and "the same input always gives the same
bytes" — plus a specific model in logit space. None of it was checked by
anything. These tests pin the helpers, the model's invariants, and the
.in/.out file contract.

Run: python3 -m unittest discover -s tests -v
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as fc  # noqa: E402

EPOCH = fc.EPOCH


def ts(hours):
    """A timestamp `hours` after the release epoch, as the case files spell it."""
    return (EPOCH + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def case(trophies, history, **extra):
    payload = {"trophies": trophies, "history": history}
    payload.update(extra)
    return payload


class NumTest(unittest.TestCase):
    def test_plain_numbers_pass_through(self):
        self.assertEqual(fc.num(3), 3.0)
        self.assertEqual(fc.num("2.5"), 2.5)
        self.assertEqual(fc.num(0), 0.0)

    def test_nothing_that_is_not_a_number_passes(self):
        for bad in (None, "", "abc", [], {}, float("nan"), float("inf"),
                    float("-inf")):
            self.assertIsNone(fc.num(bad), bad)

    def test_a_boolean_is_not_a_number(self):
        # bool is an int subclass; True must not become 1.0 and read as a rate.
        self.assertIsNone(fc.num(True))
        self.assertIsNone(fc.num(False))


class ParseTimestampTest(unittest.TestCase):
    def test_a_z_suffix_is_utc(self):
        got = fc.parse_ts("2026-11-19T00:00:00Z")
        self.assertEqual(got, EPOCH)
        self.assertEqual(got.tzinfo, timezone.utc)

    def test_a_lowercase_z_suffix_is_utc(self):
        self.assertEqual(fc.parse_ts("2026-11-19T00:00:00z"), EPOCH)

    def test_a_naive_stamp_is_read_as_utc_not_local(self):
        got = fc.parse_ts("2026-11-19T00:00:00")
        self.assertIsNotNone(got)
        self.assertEqual(got.utcoffset(), timedelta(0))
        self.assertEqual(got, EPOCH)

    def test_an_offset_is_honoured(self):
        got = fc.parse_ts("2026-11-19T07:00:00+07:00")
        self.assertEqual(got, EPOCH)

    def test_junk_is_none(self):
        for bad in (None, "", "   ", "yesterday", 12, [], "2026-13-45T00:00:00Z"):
            self.assertIsNone(fc.parse_ts(bad), bad)


class UOfTest(unittest.TestCase):
    def test_u_is_zero_at_the_epoch(self):
        self.assertAlmostEqual(fc.u_of(EPOCH), 0.0)

    def test_u_is_log_one_plus_hours(self):
        for hours in (1, 5, 24, 100):
            self.assertAlmostEqual(fc.u_of(EPOCH + timedelta(hours=hours)),
                                   math.log(1.0 + hours), places=12)

    def test_u_strictly_increases(self):
        stamps = [fc.u_of(EPOCH + timedelta(hours=h)) for h in (0, 1, 2, 24, 96)]
        for a, b in zip(stamps, stamps[1:]):
            self.assertLess(a, b)

    def test_u_survives_a_time_before_the_epoch(self):
        # 1 + hours goes negative here; the log must not raise.
        value = fc.u_of(EPOCH - timedelta(hours=5))
        self.assertTrue(math.isfinite(value))
        self.assertLess(value, 0.0)


class RateTest(unittest.TestCase):
    def test_rates_are_clamped_to_the_open_interval(self):
        self.assertEqual(fc.clamp_rate(0.0), fc.RATE_LO)
        self.assertEqual(fc.clamp_rate(-10.0), fc.RATE_LO)
        self.assertEqual(fc.clamp_rate(100.0), fc.RATE_HI)
        self.assertEqual(fc.clamp_rate(1e9), fc.RATE_HI)

    def test_clamping_leaves_an_ordinary_rate_alone(self):
        self.assertEqual(fc.clamp_rate(42.5), 42.5)

    def test_clamping_none_stays_none(self):
        self.assertIsNone(fc.clamp_rate(None))

    def test_logit_and_pct_round_trip(self):
        for rate in (0.01, 0.5, 12.5, 50.0, 87.5, 99.99):
            self.assertAlmostEqual(fc.pct_of_logit(fc.logit_rate(rate)), rate,
                                   places=6)

    def test_the_extremes_do_not_blow_up(self):
        self.assertEqual(fc.pct_of_logit(40.0), 100.0)
        self.assertEqual(fc.pct_of_logit(1e6), 100.0)
        self.assertEqual(fc.pct_of_logit(-40.0), 0.0)
        self.assertEqual(fc.pct_of_logit(-1e6), 0.0)


class RoundTest(unittest.TestCase):
    def test_rounding_is_half_up_to_four_places(self):
        self.assertEqual(fc.round4(1.00005), 1.0001)
        self.assertEqual(fc.round4(1.00004), 1.0)
        self.assertEqual(fc.round4(0.123456), 0.1235)

    def test_a_non_finite_value_rounds_to_zero(self):
        self.assertEqual(fc.round4(float("nan")), 0.0)
        self.assertEqual(fc.round4(float("inf")), 0.0)


class FinalizeTest(unittest.TestCase):
    def test_values_are_clamped_into_range(self):
        self.assertEqual(fc.finalize([-5.0, 0.0, 105.0]), [0.0, 0.0, 100.0])

    def test_the_result_never_decreases(self):
        self.assertEqual(fc.finalize([50.0, 10.0, 30.0]), [50.0, 50.0, 50.0])

    def test_five_quantiles_come_back_five_wide(self):
        self.assertEqual(len(fc.finalize([1.0, 2.0, 3.0, 4.0, 5.0])), 5)


class MedianTest(unittest.TestCase):
    def test_an_odd_count_takes_the_middle(self):
        self.assertEqual(fc.median([3.0, 1.0, 2.0]), 2.0)

    def test_an_even_count_averages_the_two_middles(self):
        self.assertEqual(fc.median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_an_empty_list_has_no_median(self):
        self.assertIsNone(fc.median([]))


class FitSlopeTest(unittest.TestCase):
    def points(self, slope, n=6, step_h=1.0):
        """y = slope * u exactly, so the fit has to recover `slope`."""
        out = []
        for i in range(n):
            t = EPOCH + timedelta(hours=step_h * i)
            u = fc.u_of(t)
            out.append((t, u, slope * u))
        return out

    def test_a_perfect_line_gives_back_its_slope(self):
        pts = self.points(0.8)
        slope, resid = fc.fit_slope(pts, pts[-1][0])
        self.assertAlmostEqual(slope, 0.8, places=9)
        self.assertAlmostEqual(resid, 0.0, places=9)

    def test_a_flat_series_has_a_zero_slope(self):
        pts = self.points(0.0)
        slope, resid = fc.fit_slope(pts, pts[-1][0])
        self.assertAlmostEqual(slope, 0.0, places=9)
        self.assertAlmostEqual(resid, 0.0, places=9)

    def test_the_residual_is_the_weighted_rms(self):
        pts = self.points(0.5)
        pts[2] = (pts[2][0], pts[2][1], pts[2][2] + 0.4)
        slope, resid = fc.fit_slope(pts, pts[-1][0])
        self.assertGreater(resid, 0.0)
        self.assertLess(resid, 0.4)

    def test_one_point_cannot_define_a_slope(self):
        self.assertEqual(fc.fit_slope(self.points(1.0, n=1), EPOCH), (None, None))

    def test_no_points_define_no_slope(self):
        self.assertEqual(fc.fit_slope([], EPOCH), (None, None))

    def test_a_vertical_series_has_no_slope(self):
        # Every point at the same u: sxx is zero and the slope is undefined.
        pts = [(EPOCH, 0.0, 1.0), (EPOCH, 0.0, 2.0), (EPOCH, 0.0, 3.0)]
        self.assertEqual(fc.fit_slope(pts, EPOCH), (None, None))

    def test_older_points_carry_less_weight(self):
        # A stale point far below the recent trend. The weighted fit must lean
        # toward the recent points, so its slope is steeper than a plain
        # least-squares fit through the same three points.
        pts = [(EPOCH, fc.u_of(EPOCH), 0.0),
               (EPOCH + timedelta(hours=72), fc.u_of(EPOCH + timedelta(hours=72)), 5.0),
               (EPOCH + timedelta(hours=96), fc.u_of(EPOCH + timedelta(hours=96)), 20.0)]
        weighted, _ = fc.fit_slope(pts, pts[-1][0])

        us = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ub = sum(us) / len(us)
        yb = sum(ys) / len(ys)
        sxx = sum((u - ub) ** 2 for u in us)
        sxy = sum((u - ub) * (y - yb) for u, y in zip(us, ys))
        unweighted = sxy / sxx

        self.assertGreater(weighted, unweighted)

    def test_the_window_drops_old_points_but_keeps_two(self):
        # Six hourly points plus one from a week earlier. The window is 24 h, so
        # the week-old point is dropped.
        pts = self.points(1.0, n=6)
        ancient = (EPOCH - timedelta(days=7), fc.u_of(EPOCH - timedelta(days=7)), 99.0)
        with_ancient = [ancient] + pts
        slope, _ = fc.fit_slope(with_ancient, pts[-1][0])
        self.assertAlmostEqual(slope, 1.0, places=6)


class CollapseTest(unittest.TestCase):
    def test_the_first_read_is_kept(self):
        kept = fc.collapse([{"at": ts(0), "rates": [1.0]}], 1)
        self.assertEqual(len(kept), 1)

    def test_an_identical_read_is_dropped(self):
        history = [{"at": ts(0), "rates": [1.0]}, {"at": ts(1), "rates": [1.0]}]
        self.assertEqual(len(fc.collapse(history, 1)), 1)

    def test_a_changed_read_is_kept(self):
        history = [{"at": ts(0), "rates": [1.0]}, {"at": ts(1), "rates": [1.5]}]
        self.assertEqual(len(fc.collapse(history, 1)), 2)

    def test_a_read_that_repeats_the_first_after_a_change_is_kept(self):
        # A -> B -> A is three refresh events, not one.
        history = [{"at": ts(0), "rates": [1.0]}, {"at": ts(1), "rates": [2.0]},
                   {"at": ts(2), "rates": [1.0]}]
        self.assertEqual(len(fc.collapse(history, 3 if False else 1)), 3)

    def test_a_read_with_no_usable_time_is_skipped(self):
        history = [{"at": "nope", "rates": [1.0]}, {"at": ts(1), "rates": [1.0]}]
        kept = fc.collapse(history, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["t"], fc.parse_ts(ts(1)))

    def test_a_read_with_no_time_does_not_become_the_baseline(self):
        # Skipping the untimed read must not make the next read look like a
        # repeat of it: the first *timed* read is the baseline and is kept.
        history = [{"at": "nope", "rates": [5.0]}, {"at": ts(1), "rates": [5.0]}]
        kept = fc.collapse(history, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["rates"], [5.0])

    def test_a_slot_key_stands_in_for_at(self):
        kept = fc.collapse([{"slot": ts(2), "rates": [1.0]}], 1)
        self.assertEqual(len(kept), 1)

    def test_a_short_rate_row_is_padded_with_none(self):
        kept = fc.collapse([{"at": ts(0), "rates": [1.0]}], 3)
        self.assertEqual(kept[0]["rates"], [1.0, None, None])

    def test_a_long_rate_row_is_truncated_to_the_trophy_count(self):
        kept = fc.collapse([{"at": ts(0), "rates": [1.0, 2.0, 3.0]}], 2)
        self.assertEqual(kept[0]["rates"], [1.0, 2.0])

    def test_a_missing_rate_list_is_all_none(self):
        kept = fc.collapse([{"at": ts(0)}], 2)
        self.assertEqual(kept[0]["rates"], [None, None])

    def test_a_non_dict_entry_is_skipped(self):
        kept = fc.collapse(["junk", None, {"at": ts(0), "rates": [1.0]}], 1)
        self.assertEqual(len(kept), 1)

    def test_players_tracked_is_carried_through(self):
        kept = fc.collapse([{"at": ts(0), "rates": [1.0], "players_tracked": 900}], 1)
        self.assertEqual(kept[0]["n"], 900.0)


class TrophySeriesTest(unittest.TestCase):
    def test_a_missing_rate_is_left_out_of_the_series(self):
        kept = [{"t": EPOCH, "u": 0.0, "n": None, "rates": [1.0]},
                {"t": EPOCH + timedelta(hours=1), "u": 0.7, "n": None, "rates": [None]}]
        pts, last_rate, _last_u, last_t = fc.trophy_series(kept, 0)
        self.assertEqual(len(pts), 1)
        self.assertEqual(last_rate, 1.0)
        self.assertEqual(last_t, EPOCH)

    def test_the_last_point_wins(self):
        kept = [{"t": EPOCH, "u": 0.0, "n": None, "rates": [1.0]},
                {"t": EPOCH + timedelta(hours=1), "u": 0.7, "n": None, "rates": [2.0]}]
        _pts, last_rate, _u, _t = fc.trophy_series(kept, 0)
        self.assertEqual(last_rate, 2.0)

    def test_an_empty_series_reports_nothing(self):
        pts, last_rate, last_u, last_t = fc.trophy_series([], 0)
        self.assertEqual(pts, [])
        self.assertIsNone(last_rate)
        self.assertIsNone(last_u)
        self.assertIsNone(last_t)

    def test_series_values_are_logits(self):
        kept = [{"t": EPOCH, "u": 0.0, "n": None, "rates": [50.0]}]
        pts, _r, _u, _t = fc.trophy_series(kept, 0)
        self.assertAlmostEqual(pts[0][2], 0.0, places=9)


class ForecastShapeTest(unittest.TestCase):
    def test_an_empty_case_has_no_trophies(self):
        self.assertEqual(fc.forecast_case({}), {"trophies": []})

    def test_every_trophy_gets_five_non_decreasing_quantiles(self):
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}],
            [{"at": ts(0), "rates": [10.0]}, {"at": ts(2), "rates": [12.0]}],
            target=ts(48), at=ts(2)))
        q = out["trophies"][0]["q"]
        self.assertEqual(len(q), 5)
        for low, high in zip(q, q[1:]):
            self.assertLessEqual(low, high)

    def test_quantiles_stay_inside_zero_and_hundred(self):
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "bronze"}],
            [{"at": ts(0), "rates": [99.9]}, {"at": ts(2), "rates": [99.95]}],
            target=ts(24 * 30), at=ts(2)))
        for value in out["trophies"][0]["q"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_the_id_comes_back_unchanged(self):
        out = fc.forecast_case(case(
            [{"id": "trophy-42", "tier": "silver"}],
            [{"at": ts(0), "rates": [5.0]}], target=ts(24), at=ts(0)))
        self.assertEqual(out["trophies"][0]["id"], "trophy-42")

    def test_a_repeated_id_is_reported_once(self):
        out = fc.forecast_case(case(
            [{"id": "dup", "tier": "gold"}, {"id": "dup", "tier": "gold"}],
            [{"at": ts(0), "rates": [5.0, 6.0]}], target=ts(24), at=ts(0)))
        self.assertEqual(len(out["trophies"]), 1)

    def test_case_order_is_preserved(self):
        out = fc.forecast_case(case(
            [{"id": "z"}, {"id": "a"}, {"id": "m"}],
            [{"at": ts(0), "rates": [1.0, 2.0, 3.0]}], target=ts(24), at=ts(0)))
        self.assertEqual([row["id"] for row in out["trophies"]], ["z", "a", "m"])

    def test_a_malformed_id_does_not_cost_the_other_trophies(self):
        # An id that is a list or an object cannot be a set member. Coercing it
        # to a hashable key keeps the row instead of raising, which used to
        # throw the whole case away and write an empty forecast.
        out = fc.forecast_case(case(
            [{"id": "good", "tier": "gold"}, {"id": {"bad": 1}, "tier": "gold"},
             {"id": [1, 2], "tier": "gold"}],
            [{"at": ts(0), "rates": [10.0, 20.0, 30.0]}], target=ts(24), at=ts(0)))
        self.assertEqual(len(out["trophies"]), 3)
        self.assertEqual(out["trophies"][0]["id"], "good")
        self.assertEqual(out["trophies"][1]["id"], {"bad": 1})

    def test_a_malformed_id_is_still_deduplicated(self):
        out = fc.forecast_case(case(
            [{"id": [1, 2]}, {"id": [1, 2]}],
            [{"at": ts(0), "rates": [10.0, 20.0]}], target=ts(24), at=ts(0)))
        self.assertEqual(len(out["trophies"]), 1)

    def test_a_missing_id_and_an_id_named_missing_stay_apart(self):
        out = fc.forecast_case(case(
            [{"id": "missing"}, {"tier": "gold"}],
            [{"at": ts(0), "rates": [10.0, 20.0]}], target=ts(24), at=ts(0)))
        self.assertEqual(len(out["trophies"]), 2)

    def test_a_trophy_with_no_id_still_gets_a_row(self):
        out = fc.forecast_case(case([{"tier": "gold"}], [{"at": ts(0), "rates": [3.0]}],
                                    target=ts(24), at=ts(0)))
        self.assertEqual(len(out["trophies"]), 1)


class ForecastBehaviourTest(unittest.TestCase):
    def rising(self, target_h=48):
        return fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}],
            [{"at": ts(h), "rates": [10.0 + 2.0 * h]} for h in range(0, 6)],
            target=ts(target_h), at=ts(5)))

    def test_a_rising_trophy_forecasts_above_its_last_read(self):
        self.assertGreater(self.rising()["trophies"][0]["q"][2], 20.0)

    def test_a_falling_trophy_forecasts_below_its_last_read(self):
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}],
            [{"at": ts(h), "rates": [60.0 - 2.0 * h]} for h in range(0, 6)],
            target=ts(48), at=ts(5)))
        self.assertLess(out["trophies"][0]["q"][2], 50.0)

    def test_a_hundred_percent_trophy_stays_at_a_hundred(self):
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "platinum"}],
            [{"at": ts(h), "rates": [100.0]} for h in range(0, 4)],
            target=ts(48), at=ts(3)))
        self.assertEqual(out["trophies"][0]["q"], [100.0] * 5)

    def test_a_hundred_percent_trophy_stays_at_a_hundred_with_no_history(self):
        out = fc.forecast_case(case([{"id": "a"}], [{"at": ts(0), "rates": [100.0]}],
                                    target=ts(48), at=ts(0)))
        self.assertEqual(out["trophies"][0]["q"], [100.0] * 5)

    def test_a_longer_horizon_gives_a_wider_spread(self):
        def width(hours):
            out = fc.forecast_case(case(
                [{"id": "a", "tier": "gold"}],
                [{"at": ts(h), "rates": [10.0 + 2.0 * h]} for h in range(0, 6)],
                target=ts(hours), at=ts(5)))
            q = out["trophies"][0]["q"]
            return q[4] - q[0]

        self.assertLess(width(12), width(96))

    def test_the_step_cap_holds_over_a_four_day_horizon(self):
        # A steep slope over a long horizon must not run past the cap, so the
        # median cannot exceed the last read by more than the cap allows.
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "bronze"}],
            [{"at": ts(h), "rates": [1.0 + 0.01 * h]} for h in range(0, 6)],
            target=ts(24 * 5), at=ts(5)))
        median = out["trophies"][0]["q"][2]
        ceiling = fc.pct_of_logit(fc.logit_rate(1.05) + fc.STEP_CAP)
        self.assertLessEqual(median, ceiling + 1e-6)

    def test_a_single_read_falls_back_to_the_plain_spread(self):
        out = fc.forecast_case(case([{"id": "a", "tier": "gold"}],
                                    [{"at": ts(0), "rates": [40.0]}],
                                    target=ts(48), at=ts(0)))
        q = out["trophies"][0]["q"]
        self.assertEqual(len(q), 5)
        self.assertLess(q[0], 40.0)
        self.assertGreater(q[4], 40.0)

    def test_the_horizon_is_derived_when_it_is_not_given(self):
        explicit = fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}],
            [{"at": ts(0), "rates": [10.0]}, {"at": ts(2), "rates": [12.0]}],
            target=ts(50), at=ts(2), horizon_minutes=48 * 60))
        derived = fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}],
            [{"at": ts(0), "rates": [10.0]}, {"at": ts(2), "rates": [12.0]}],
            target=ts(50), at=ts(2)))
        self.assertEqual(explicit, derived)

    def test_a_trophy_missing_from_the_last_read_uses_its_own_last_read(self):
        out = fc.forecast_case(case(
            [{"id": "a", "tier": "gold"}, {"id": "b", "tier": "gold"}],
            [{"at": ts(0), "rates": [10.0, 20.0]},
             {"at": ts(2), "rates": [12.0, None]}],
            target=ts(48), at=ts(2)))
        self.assertEqual(len(out["trophies"]), 2)

    def test_a_rate_below_the_floor_is_held_at_the_floor(self):
        out = fc.forecast_case(case([{"id": "a"}], [{"at": ts(0), "rates": [0.0]}],
                                    target=ts(24), at=ts(0)))
        for value in out["trophies"][0]["q"]:
            self.assertGreaterEqual(value, 0.0)

    def test_players_tracked_growth_pulls_a_slow_climb_down(self):
        # Same rates, but one case shows the tracked population growing. The
        # denominator term must make the second forecast no higher.
        history = [{"at": ts(h), "rates": [10.0 + 0.05 * h]} for h in range(0, 6)]
        flat = fc.forecast_case(case([{"id": "a", "tier": "gold"}],
                                     [dict(h, players_tracked=1000) for h in history],
                                     target=ts(48), at=ts(5)))
        growing = fc.forecast_case(case([{"id": "a", "tier": "gold"}],
                                        [dict(h, players_tracked=1000 + 100 * i)
                                         for i, h in enumerate(history)],
                                        target=ts(48), at=ts(5)))
        self.assertLessEqual(growing["trophies"][0]["q"][2],
                             flat["trophies"][0]["q"][2])

    def test_a_case_that_is_not_a_dict_is_an_empty_forecast(self):
        self.assertEqual(fc.forecast_case({}), {"trophies": []})

    def test_trophies_that_are_not_a_list_are_ignored(self):
        self.assertEqual(fc.forecast_case({"trophies": "nope", "history": []}),
                         {"trophies": []})

    def test_the_same_case_forecasts_the_same_numbers(self):
        first = self.rising()
        second = self.rising()
        self.assertEqual(first, second)


class RobustnessTest(unittest.TestCase):
    """Nothing a case file can hold may raise, and the shape must hold."""

    def cases(self, seed=7, count=2000):
        import random
        rng = random.Random(seed)
        pool = [None, True, False, 0, -1, 1e308, "x", "", [], {}, [1, None],
                "2026-11-19T00:00:00Z", float("nan"), 3.5, 1e-9, 10 ** 30]

        def value(depth=0):
            roll = rng.random()
            if depth > 2 or roll < 0.5:
                return rng.choice(pool)
            if roll < 0.75:
                return [value(depth + 1) for _ in range(rng.randint(0, 3))]
            keys = ("id", "tier", "rates", "at", "slot", "players_tracked",
                    "target", "horizon_minutes", "trophies", "history")
            return {rng.choice(keys): value(depth + 1) for _ in range(rng.randint(0, 4))}

        for _ in range(count):
            yield {
                "trophies": [value(1) for _ in range(rng.randint(0, 4))],
                "history": [value(1) for _ in range(rng.randint(0, 6))],
                "target": value(2), "at": value(2), "horizon_minutes": value(2),
            }

    def test_no_malformed_case_raises_or_breaks_the_shape(self):
        for payload in self.cases():
            out = fc.forecast_case(payload)
            self.assertIsInstance(out["trophies"], list)
            for row in out["trophies"]:
                q = row["q"]
                self.assertEqual(len(q), 5)
                for value in q:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)
                for low, high in zip(q, q[1:]):
                    self.assertLessEqual(low, high)


class FileContractTest(unittest.TestCase):
    def setUp(self):
        self.in_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.in_dir, True)
        self.addCleanup(shutil.rmtree, self.out_dir, True)

    def write_case(self, name, payload):
        path = os.path.join(self.in_dir, name + ".in")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    def run_main(self):
        old = os.environ.get("MAKEFUN_INPUT_DIR"), os.environ.get("MAKEFUN_OUTPUT_DIR")
        os.environ["MAKEFUN_INPUT_DIR"] = self.in_dir
        os.environ["MAKEFUN_OUTPUT_DIR"] = self.out_dir
        try:
            return fc.main()
        finally:
            for key, value in zip(("MAKEFUN_INPUT_DIR", "MAKEFUN_OUTPUT_DIR"), old):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def read_out(self, stem):
        with open(os.path.join(self.out_dir, stem + ".out"), encoding="utf-8") as fh:
            return fh.read()

    def test_one_out_file_per_in_file(self):
        self.write_case("a", case([{"id": "x"}], [{"at": ts(0), "rates": [5.0]}]))
        self.write_case("b", case([{"id": "y"}], [{"at": ts(0), "rates": [6.0]}]))
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["a.out", "b.out"])

    def test_the_output_is_compact_json_with_a_trailing_newline(self):
        self.write_case("a", case([{"id": "x"}], [{"at": ts(0), "rates": [5.0]}]))
        self.run_main()
        text = self.read_out("a")
        self.assertTrue(text.endswith("\n"))
        self.assertNotIn(", ", text)
        self.assertNotIn(": ", text)
        self.assertEqual(json.loads(text)["trophies"][0]["id"], "x")

    def test_the_same_input_gives_the_same_bytes(self):
        payload = case([{"id": "x", "tier": "gold"}],
                       [{"at": ts(h), "rates": [10.0 + h]} for h in range(0, 5)],
                       target=ts(48), at=ts(4))
        self.write_case("a", payload)
        self.run_main()
        first = self.read_out("a")
        self.run_main()
        self.assertEqual(first, self.read_out("a"))

    def test_a_broken_case_does_not_stop_the_run(self):
        self.write_case("bad", "{not json")
        self.write_case("good", case([{"id": "x"}], [{"at": ts(0), "rates": [5.0]}]))
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(json.loads(self.read_out("bad")), {"trophies": []})
        self.assertEqual(json.loads(self.read_out("good"))["trophies"][0]["id"], "x")

    def test_a_case_that_is_not_an_object_is_an_empty_forecast(self):
        self.write_case("a", "[1, 2, 3]")
        self.run_main()
        self.assertEqual(json.loads(self.read_out("a")), {"trophies": []})

    def test_a_malformed_id_does_not_empty_the_case_file(self):
        payload = case([{"id": "good", "tier": "gold"}, {"id": {"bad": 1}}],
                       [{"at": ts(0), "rates": [10.0, 20.0]}],
                       target=ts(24), at=ts(0))
        self.write_case("a", payload)
        self.run_main()
        rows = json.loads(self.read_out("a"))["trophies"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "good")

    def test_an_empty_input_directory_writes_nothing(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_non_in_files_are_ignored(self):
        self.write_case("a", case([{"id": "x"}], [{"at": ts(0), "rates": [5.0]}]))
        with open(os.path.join(self.in_dir, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("ignore me")
        self.run_main()
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["a.out"])


if __name__ == "__main__":
    unittest.main()
