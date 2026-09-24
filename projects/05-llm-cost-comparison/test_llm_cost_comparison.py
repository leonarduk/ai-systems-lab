"""Tests for llm_cost_comparison.py.

Covers the pure cost-calculation functions, pricing/config loading, GPU
detection parsing (with a mocked subprocess runner), and export helpers.
Interactive input() flows are intentionally not exercised here — the
interactive functions are thin wrappers over the tested pure functions.
"""

from __future__ import annotations

import json
import math
import os
import time
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import llm_cost_comparison as m

# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


def test_workload_monthly_totals():
    w = m.Workload(requests_per_day=1000, avg_input_tokens=500, avg_output_tokens=300)
    assert w.monthly_input_tokens == 1000 * 500 * 30
    assert w.monthly_output_tokens == 1000 * 300 * 30
    assert w.monthly_total_tokens == w.monthly_input_tokens + w.monthly_output_tokens


# --------------------------------------------------------------------------
# Hosted cost
# --------------------------------------------------------------------------


def test_hosted_monthly_cost():
    w = m.Workload(
        requests_per_day=1000, avg_input_tokens=1_000_000 / 1000, avg_output_tokens=0
    )
    # 1 request/day-equivalent scaling chosen so monthly input tokens == 30M exactly
    cost = m.hosted_monthly_cost(w, input_per_million=2.0, output_per_million=10.0)
    assert cost == pytest.approx(30_000_000 / 1_000_000 * 2.0)


def test_hosted_monthly_cost_combines_input_and_output():
    w = m.Workload(requests_per_day=100, avg_input_tokens=100, avg_output_tokens=50)
    cost = m.hosted_monthly_cost(w, input_per_million=3.0, output_per_million=15.0)
    expected = (w.monthly_input_tokens / 1_000_000 * 3.0) + (
        w.monthly_output_tokens / 1_000_000 * 15.0
    )
    assert cost == pytest.approx(expected)


# --------------------------------------------------------------------------
# Local cost math
# --------------------------------------------------------------------------


def test_hours_needed_for_workload():
    # 3,600,000 tokens at 1000 tok/s -> 3,600,000 / (1000*3600) = 1 hour
    hours = m.hours_needed_for_workload(
        total_monthly_tokens=3_600_000, tokens_per_sec=1000
    )
    assert hours == pytest.approx(1.0)


def test_hours_needed_for_workload_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        m.hours_needed_for_workload(1000, 0)


# --------------------------------------------------------------------------
# Scaling a workload to what local throughput can actually produce
# --------------------------------------------------------------------------


def test_scale_workload_to_local_capacity_unchanged_when_feasible():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=300)
    effective, feasible, coverage_pct = m.scale_workload_to_local_capacity(
        w, tokens_per_sec=1000
    )
    assert effective is w
    assert feasible is True
    assert coverage_pct == pytest.approx(100.0)


def test_scale_workload_to_local_capacity_scales_down_when_infeasible():
    # 500 req/day * 4800 tokens/req * 30 days = 72,000,000 tokens/month.
    # At 10 tok/s, that needs 72e6/(10*3600) = 2000 hours — 720 exist in a
    # month, so coverage = 720/2000 = 36%.
    w = m.Workload(requests_per_day=500, avg_input_tokens=4000, avg_output_tokens=800)
    effective, feasible, coverage_pct = m.scale_workload_to_local_capacity(
        w, tokens_per_sec=10
    )
    assert feasible is False
    assert coverage_pct == pytest.approx(36.0)
    assert effective.requests_per_day == pytest.approx(500 * 0.36)
    # Same average input/output tokens per request — only volume scales.
    assert effective.avg_input_tokens == 4000
    assert effective.avg_output_tokens == 800
    # The scaled workload should now be exactly at the feasibility boundary.
    hours_needed = m.hours_needed_for_workload(effective.monthly_total_tokens, 10)
    assert hours_needed == pytest.approx(m.HOURS_PER_MONTH)


def test_local_monthly_cost_owned_splits_fixed_and_variable():
    # $3600 hardware / 3 years -> $100/month fixed, regardless of usage
    cost_idle = m.local_monthly_cost_owned(
        hardware_cost=3600,
        lifetime_years=3,
        power_watts=450,
        electricity_rate_per_kwh=0.15,
        hours_needed_per_month=0,
    )
    assert cost_idle == pytest.approx(100.0)

    cost_used = m.local_monthly_cost_owned(
        hardware_cost=3600,
        lifetime_years=3,
        power_watts=1000,  # 1 kW for easy math
        electricity_rate_per_kwh=0.10,
        hours_needed_per_month=10,
    )
    # fixed 100 + variable (1kW * $0.10/hr * 10hr) = 100 + 1 = 101
    assert cost_used == pytest.approx(101.0)


def test_local_monthly_cost_owned_rejects_zero_lifetime():
    with pytest.raises(ValueError):
        m.local_monthly_cost_owned(1000, 0, 100, 0.1, 10)


def test_local_monthly_cost_rented():
    assert m.local_monthly_cost_rented(
        hourly_rate=2.5, hours_needed_per_month=40
    ) == pytest.approx(100.0)


def test_cost_per_million_tokens():
    assert m.cost_per_million_tokens(
        monthly_cost=50, monthly_total_tokens=25_000_000
    ) == pytest.approx(2.0)


def test_cost_per_million_tokens_rejects_zero_tokens():
    with pytest.raises(ValueError):
        m.cost_per_million_tokens(50, 0)


# --------------------------------------------------------------------------
# Comparison rows
# --------------------------------------------------------------------------


def test_build_local_row_owned():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(
        w,
        tokens_per_sec=100,
        mode="own",
        hardware_cost=3600,
        lifetime_years=3,
        power_watts=450,
        electricity_rate_per_kwh=0.15,
    )
    assert row.name == "Local (buy hardware)"
    assert row.monthly_cost > 0
    assert row.cost_per_million_tokens > 0
    assert "compute-hrs/month" in row.notes


def test_build_local_row_flags_when_throughput_cannot_keep_up_in_real_time():
    # A huge workload against a slow tokens/sec needs more compute-hours than
    # exist in a month (720). The cost is real — it's what running a fleet of
    # machines flat-out, 24/7, all month would cost — but the notes must say
    # plainly that this only covers part of the workload on a single machine
    # rather than implying the full requested volume was delivered for that
    # price.
    w = m.Workload(requests_per_day=50000, avg_input_tokens=500, avg_output_tokens=300)
    row = m.build_local_row(
        w,
        tokens_per_sec=7.7,
        mode="existing",
        power_watts=100,
        electricity_rate_per_kwh=0.15,
    )
    assert "covers only ~" in row.notes
    assert "machines" in row.notes
    assert row.feasible is False
    assert row.monthly_cost > 0


# A workload needing more compute-hours than a month contains, reused by
# every fleet test below so they all describe the same scenario:
#   500 req/day * 4800 tokens/req * 30 days = 72,000,000 tokens/month
#   at 10 tok/s that is 72e6 / (10 * 3600) = 2000 machine-hours
#   720 hours exist in a month, so ceil(2000 / 720) = 3 machines
# Note 3 * 720 = 2160 > 2000: the fleet has 160 hours of spare capacity, and
# nothing may be billed for it.
_FLEET_WORKLOAD = dict(
    requests_per_day=500, avg_input_tokens=4000, avg_output_tokens=800
)
_FLEET_TOKENS = 72_000_000
_FLEET_HOURS = 2000.0
_FLEET_MACHINES = 3


def test_fleet_fixture_matches_its_stated_arithmetic():
    # The expected costs below are hand-computed from these numbers, so if
    # the fixture drifts the other tests would silently assert the wrong
    # thing. Pin it.
    w = m.Workload(**_FLEET_WORKLOAD)
    assert w.monthly_total_tokens == _FLEET_TOKENS
    assert m.hours_needed_for_workload(w.monthly_total_tokens, 10) == pytest.approx(
        _FLEET_HOURS
    )
    assert math.ceil(_FLEET_HOURS / m.HOURS_PER_MONTH) == _FLEET_MACHINES
    assert _FLEET_MACHINES * m.HOURS_PER_MONTH > _FLEET_HOURS


def test_build_local_row_charges_variable_cost_for_hours_needed_not_fleet_capacity():
    # "existing" is a purely variable mode: the only cost is electricity
    # while generating. Three machines for 667 hours each burn exactly the
    # same power as one machine for 2000 hours, so the bill is for 2000
    # machine-hours — NOT 3 * 720 = 2160, which would charge for 160 hours
    # of idle spare capacity nobody uses.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD),
        tokens_per_sec=10,
        mode="existing",
        power_watts=1000,  # 1 kW for easy math
        electricity_rate_per_kwh=0.10,
    )
    assert row.feasible is False
    # 1 kW * $0.10/kWh * 2000 hr = $200.00, not 3 * (1 * 0.10 * 720) = $216.
    assert row.monthly_cost == pytest.approx(200.0)
    assert row.cost_per_million_tokens == pytest.approx(200.0 / 72.0)
    assert "3 machines" in row.notes


def test_build_local_row_rent_charges_hours_needed_not_fleet_capacity():
    # Renting is billed by the hour, so a fleet delivering the workload
    # rents 2000 GPU-hours in total. Capping each machine at 720 hours and
    # multiplying by 3 would invoice 2160 hours — inflating the rented-cloud
    # option by 8% here, and by nearly 2x just past the 720-hour boundary.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD), tokens_per_sec=10, mode="rent", hourly_rate=2.0
    )
    assert row.feasible is False
    assert row.monthly_cost == pytest.approx(2.0 * _FLEET_HOURS)  # $4000, not $4320


def test_build_local_row_owned_mode_scales_hardware_amortization_by_machine_count():
    # The fixed hardware amortization must be multiplied by the machine
    # count — charging one card's amortization for a three-card fleet is
    # the under-estimate issue #52 is about.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD),
        tokens_per_sec=10,
        mode="own",
        hardware_cost=3600,  # $100/month amortized per machine
        lifetime_years=3,
        power_watts=0,  # isolate the fixed component
        electricity_rate_per_kwh=0.0,
    )
    assert row.feasible is False
    assert row.monthly_cost == pytest.approx(300.0)  # 3 machines * $100/month


def test_build_local_row_owned_mode_splits_fixed_and_variable_correctly():
    # Both components at once: fixed scales by machines, variable by hours.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD),
        tokens_per_sec=10,
        mode="own",
        hardware_cost=3600,
        lifetime_years=3,
        power_watts=1000,
        electricity_rate_per_kwh=0.10,
    )
    # 3 * $100 amortization + 1 kW * $0.10 * 2000 hr = $300 + $200 = $500.
    # Scaling the whole per-machine cost instead would give 3 * (100 + 72)
    # = $516, over-charging the electricity.
    assert row.monthly_cost == pytest.approx(500.0)


def test_build_local_row_always_on_scales_idle_draw_but_not_generation():
    # A 24/7 server's idle draw is owed for all 720 hours per machine, so it
    # triples with the fleet. The extra draw while generating is variable,
    # so it is charged once for the 2000 machine-hours of actual work.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD),
        tokens_per_sec=10,
        mode="always_on",
        idle_watts=100,
        extra_watts=900,
        electricity_rate_per_kwh=0.10,
    )
    idle = _FLEET_MACHINES * 0.1 * 0.10 * m.HOURS_PER_MONTH  # $21.60
    generation = 0.9 * 0.10 * _FLEET_HOURS  # $180.00
    assert row.monthly_cost == pytest.approx(idle + generation)  # $201.60


@pytest.mark.parametrize(
    "mode, kwargs",
    [
        ("existing", dict(power_watts=1000, electricity_rate_per_kwh=0.10)),
        ("rent", dict(hourly_rate=2.0)),
    ],
)
def test_purely_variable_modes_keep_the_same_rate_per_million(mode, kwargs):
    # For modes with no fixed component, $/1M is a property of the hardware
    # and the tariff, not of how big the workload is. A feasible run and an
    # infeasible three-machine run at the same tok/s must price identically.
    small = m.build_local_row(
        m.Workload(requests_per_day=10, avg_input_tokens=4000, avg_output_tokens=800),
        tokens_per_sec=10,
        mode=mode,
        **kwargs,
    )
    fleet = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD), tokens_per_sec=10, mode=mode, **kwargs
    )
    assert small.feasible is True
    assert fleet.feasible is False
    assert fleet.cost_per_million_tokens == pytest.approx(small.cost_per_million_tokens)


def test_owned_mode_rate_per_million_falls_as_tokens_spread_the_fixed_cost():
    # Named for what it asserts. An earlier version of this called the
    # effect a rise, which the assertion below contradicts and which the
    # docstring repeated: between machine boundaries the fixed cost is
    # spread over more tokens, so $/1M falls. The rise happens *at* a
    # boundary, which is the next test.
    common = dict(
        tokens_per_sec=10,
        mode="own",
        hardware_cost=3600,
        lifetime_years=3,
        power_watts=1000,
        electricity_rate_per_kwh=0.10,
    )
    # Same tokens/hour ratio, but small enough for one machine.
    small = m.build_local_row(
        m.Workload(requests_per_day=100, avg_input_tokens=4000, avg_output_tokens=800),
        **common,
    )
    fleet = m.build_local_row(m.Workload(**_FLEET_WORKLOAD), **common)
    assert small.feasible is True
    assert fleet.feasible is False
    assert fleet.cost_per_million_tokens < small.cost_per_million_tokens


def test_owned_mode_rate_per_million_jumps_at_a_machine_boundary():
    # The sawtooth. Two extra hours of work either side of the 720-hour
    # line cost a whole extra card's amortization, so $/1M steps up even
    # though the workload barely grew. This is the effect issue #52 exists
    # to surface, and it is invisible to a small-vs-large comparison,
    # which only shows the downward trend between boundaries.
    common = dict(
        tokens_per_sec=10,
        mode="own",
        hardware_cost=3600,  # $100/month per machine
        lifetime_years=3,
        power_watts=1000,
        electricity_rate_per_kwh=0.10,
    )
    # 179 req/day * 4800 tokens * 30 = 25,776,000 tokens -> 716 hours.
    just_under = m.build_local_row(
        m.Workload(requests_per_day=179, avg_input_tokens=4000, avg_output_tokens=800),
        **common,
    )
    # 181 req/day -> 26,064,000 tokens -> 724 hours, so a second machine.
    just_over = m.build_local_row(
        m.Workload(requests_per_day=181, avg_input_tokens=4000, avg_output_tokens=800),
        **common,
    )
    assert just_under.feasible is True
    assert just_over.feasible is False
    # 1 * $100 + $0.10 * 716 = $171.60 over 25.776M tokens
    assert just_under.cost_per_million_tokens == pytest.approx(171.6 / 25.776)
    # 2 * $100 + $0.10 * 724 = $272.40 over 26.064M tokens
    assert just_over.cost_per_million_tokens == pytest.approx(272.4 / 26.064)
    assert just_over.cost_per_million_tokens > just_under.cost_per_million_tokens


def test_always_on_single_machine_charges_idle_once():
    # num_machines == 1 must leave the idle term exactly as the helper
    # computes it — the `idle_watts * num_machines` scaling has to be a
    # no-op below the boundary, not an off-by-one.
    w = m.Workload(requests_per_day=100, avg_input_tokens=4000, avg_output_tokens=800)
    hours = m.hours_needed_for_workload(w.monthly_total_tokens, 10)
    assert hours < m.HOURS_PER_MONTH
    row = m.build_local_row(
        w,
        tokens_per_sec=10,
        mode="always_on",
        idle_watts=100,
        extra_watts=900,
        electricity_rate_per_kwh=0.10,
    )
    expected = m.local_monthly_cost_always_on(100, 900, 0.10, hours)
    assert row.monthly_cost == pytest.approx(expected)


def test_build_local_row_uses_one_machine_exactly_at_the_month_boundary():
    # 180 req/day * 4800 tokens * 30 = 25,920,000 tokens; at 10 tok/s that
    # is exactly HOURS_PER_MONTH. ceil() must not round this up to 2 — the
    # max(1, ...) and the <= in `feasible` both sit on this edge.
    w = m.Workload(requests_per_day=180, avg_input_tokens=4000, avg_output_tokens=800)
    assert m.hours_needed_for_workload(w.monthly_total_tokens, 10) == pytest.approx(
        m.HOURS_PER_MONTH
    )
    row = m.build_local_row(w, tokens_per_sec=10, mode="rent", hourly_rate=2.0)
    assert row.feasible is True
    assert row.monthly_cost == pytest.approx(2.0 * m.HOURS_PER_MONTH)


def test_build_local_row_costs_the_full_workload_not_just_what_one_machine_makes():
    # $/1M is computed against the workload's full monthly total, which the
    # fleet does deliver. Dividing by one machine's 720 hours of output
    # would report a rate for tokens the user never asked for.
    row = m.build_local_row(
        m.Workload(**_FLEET_WORKLOAD),
        tokens_per_sec=10,
        mode="rent",
        hourly_rate=2.0,
    )
    assert row.cost_per_million_tokens == pytest.approx(
        row.monthly_cost / _FLEET_TOKENS * 1_000_000
    )


def test_build_local_row_rejects_a_zero_token_workload():
    # ceil(0 / 720) is 0, so there is no max(1, ...) floor on the machine
    # count: a zero-token workload has no hours to cost and is rejected
    # downstream by cost_per_million_tokens. Clamping to one machine would
    # only have produced a $0 row for a workload that does not exist.
    with pytest.raises(
        ValueError, match=r"monthly_total_tokens must be > 0 to cost a local option"
    ):
        m.build_local_row(
            m.Workload(requests_per_day=0, avg_input_tokens=0, avg_output_tokens=0),
            tokens_per_sec=10,
            mode="rent",
            hourly_rate=2.0,
        )


def test_build_local_row_no_warning_when_throughput_is_sufficient():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(
        w,
        tokens_per_sec=100,
        mode="existing",
        power_watts=100,
        electricity_rate_per_kwh=0.15,
    )
    assert "needs ~" not in row.notes


def test_build_local_row_rented():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(w, tokens_per_sec=100, mode="rent", hourly_rate=2.0)
    assert row.name == "Local (rented cloud GPU)"
    assert row.monthly_cost > 0


def test_build_local_row_rented_preserves_custom_name():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(
        w, tokens_per_sec=100, mode="rent", hourly_rate=2.0, name="Local H100 rental"
    )
    assert row.name == "Local H100 rental"


def test_build_local_row_rejects_unknown_mode():
    w = m.Workload(100, 500, 500)
    with pytest.raises(ValueError):
        m.build_local_row(w, tokens_per_sec=100, mode="bogus")


def test_build_hosted_rows_all_and_filtered(tmp_path):
    pricing = {
        "providers": {
            "claude": {
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": 5.0,
                        "output_per_million": 25.0,
                    },
                    "haiku-4.5": {
                        "display_name": "Claude Haiku 4.5",
                        "input_per_million": 1.0,
                        "output_per_million": 5.0,
                    },
                }
            },
            "deepseek": {
                "models": {
                    "deepseek-v3": {
                        "display_name": "DeepSeek-V3",
                        "input_per_million": 0.27,
                        "output_per_million": 1.10,
                    }
                }
            },
        }
    }
    w = m.Workload(1000, 500, 300)

    all_rows = m.build_hosted_rows(w, pricing)
    assert len(all_rows) == 3
    names = {r.name for r in all_rows}
    assert names == {"Claude Opus 5", "Claude Haiku 4.5", "DeepSeek-V3"}

    filtered = m.build_hosted_rows(w, pricing, selected={"claude/haiku-4.5"})
    assert len(filtered) == 1
    assert filtered[0].name == "Claude Haiku 4.5"

    # Haiku should be cheaper than Opus for the same workload
    haiku_cost = next(r.monthly_cost for r in all_rows if r.name == "Claude Haiku 4.5")
    opus_cost = next(r.monthly_cost for r in all_rows if r.name == "Claude Opus 5")
    assert haiku_cost < opus_cost


_WARN_PRICING = {
    "providers": {
        "claude": {
            "models": {
                "opus-5": {
                    "display_name": "Claude Opus 5",
                    "input_per_million": 5.0,
                    "output_per_million": 25.0,
                }
            }
        }
    }
}


def test_warn_unknown_model_keys_reports_to_stderr(capsys):
    unknown = m.warn_unknown_model_keys(
        _WARN_PRICING, {"claude/opus-5", "claude/typo-model"}
    )
    assert unknown == ["claude/typo-model"]
    captured = capsys.readouterr()
    assert "Warning: unknown model key" in captured.err
    assert "claude/typo-model" in captured.err
    # stdout carries the comparison table; a warning there would corrupt
    # anything parsing it.
    assert captured.out == ""
    # The valid keys are listed, so the user can see what they meant to
    # type rather than being told only that they were wrong.
    assert "claude/opus-5" in captured.err


def test_warn_unknown_model_keys_is_silent_when_all_keys_are_known(capsys):
    assert m.warn_unknown_model_keys(_WARN_PRICING, {"claude/opus-5"}) == []
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("selected", [None, set()])
def test_warn_unknown_model_keys_is_silent_without_a_selection(capsys, selected):
    # None means "compare against everything"; an empty set means "local
    # only". Neither is a mistake.
    assert m.warn_unknown_model_keys(_WARN_PRICING, selected) == []
    assert capsys.readouterr().err == ""


def test_build_hosted_rows_still_drops_unknown_keys_without_printing(capsys):
    # The row builder keeps its behaviour — unknown keys simply match
    # nothing — but no longer prints. Both callers invoke it inside a
    # per-scenario loop, so warning here repeated the same message once
    # per scenario, and a row builder that writes to stderr cannot be
    # reused by a caller formatting its own output.
    rows = m.build_hosted_rows(
        m.Workload(1000, 500, 300),
        _WARN_PRICING,
        selected={"claude/opus-5", "claude/typo-model"},
    )
    assert [r.name for r in rows] == ["Claude Opus 5"]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_unknown_key_warns_once_for_a_multi_scenario_run(tmp_path: Path, capsys):
    # The regression the move exists to prevent: three presets must not
    # produce three copies of the same warning.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload_presets": ["casual", "coding_agent", "team_tool"],
                "local": {
                    "mode": "existing",
                    "tokens_per_sec": 40,
                    "power_watts": 450,
                    "electricity_rate_per_kwh": 0.15,
                },
                "pricing_file": str(pricing_path),
                "selected_models": ["claude/opus-5", "claude/typo-model"],
            }
        ),
        encoding="utf-8",
    )
    assert m.run_non_interactive(config_path, export_fmt=None, export_path=None) == 0
    err = capsys.readouterr().err
    assert err.count("claude/typo-model") == 1


def test_build_hosted_rows_raises_config_error_on_malformed_pricing():
    pricing = {
        "providers": {
            "claude": {
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": "not-a-number",
                        "output_per_million": 25.0,
                    }
                }
            }
        }
    }
    w = m.Workload(1000, 500, 300)
    with pytest.raises(m.ConfigError, match="claude/opus-5"):
        m.build_hosted_rows(w, pricing)


@pytest.mark.parametrize(
    "bad_value",
    [
        None,  # missing
        0,  # zero
        -1.0,  # negative
        "1.0",  # non-numeric
        float("nan"),  # NaN
        float("inf"),  # inf
        True,  # bool (int subclass)
    ],
)
def test_build_hosted_rows_direct_call_rejects_invalid_price(bad_value):
    # build_hosted_rows is called directly here with a hand-constructed
    # pricing dict that bypassed load_pricing. The guard must still raise
    # ConfigError with the established message rather than silently
    # computing a cost from an invalid price.
    pricing = {
        "providers": {
            "claude": {
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": bad_value,
                        "output_per_million": 25.0,
                    }
                }
            }
        }
    }
    w = m.Workload(1000, 500, 300)
    with pytest.raises(
        m.ConfigError,
        match=r"pricing model 'claude/opus-5' is missing a numeric input_per_million",
    ):
        m.build_hosted_rows(w, pricing)


def test_build_hosted_rows_direct_call_rejects_invalid_output_price():
    pricing = {
        "providers": {
            "claude": {
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": 5.0,
                        "output_per_million": 0,
                    }
                }
            }
        }
    }
    w = m.Workload(1000, 500, 300)
    with pytest.raises(
        m.ConfigError,
        match=r"pricing model 'claude/opus-5' is missing a numeric output_per_million",
    ):
        m.build_hosted_rows(w, pricing)


# --------------------------------------------------------------------------
# Real pricing.json shipped alongside the script
# --------------------------------------------------------------------------


# A function-local import may be deliberate — an optional dependency, or
# breaking an import cycle. Two escape hatches, both of which make the
# reason visible at the import site rather than leaving a reader to guess:
DEFERRED_IMPORT_MARKER = "deferred-import:"


def _function_local_imports(source: str) -> list:
    """Every import inside a function, minus the deliberately deferred ones.

    Exempt if the import sits under a ``try`` whose handlers catch exactly
    ``ImportError`` (bare ``except ImportError:`` or a tuple such as
    ``except (ImportError, ModuleNotFoundError):`` containing it) — the
    optional-dependency idiom — or if its line carries a
    ``# deferred-import: <reason>`` comment. A bare ``except:`` or a
    handler for some other exception type (e.g. ``except ValueError:``)
    does NOT exempt the import: only the exact ImportError idiom does.
    """
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()

    exempt_lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            (isinstance(handler.type, ast.Name) and handler.type.id == "ImportError")
            or (
                isinstance(handler.type, ast.Tuple)
                and any(
                    isinstance(e, ast.Name) and e.id == "ImportError"
                    for e in handler.type.elts
                )
            )
            for handler in node.handlers
        )
        if not catches_import_error:
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    exempt_lines.add(inner.lineno)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, (ast.Import, ast.ImportFrom)):
                continue
            if inner.lineno in exempt_lines:
                continue
            if DEFERRED_IMPORT_MARKER in lines[inner.lineno - 1]:
                continue
            names = ", ".join(a.name for a in inner.names)
            offenders.append(f"{node.name}() line {inner.lineno}: {names}")
    return offenders


def test_no_undeclared_function_local_imports():
    # Issue #76 asks for `import re` to be hoisted. Hoisting only the one
    # the issue names leaves the pattern in place — there were three — so
    # this pins the rule rather than the instance.
    #
    # It is not an absolute ban. A deferred import is sometimes right, so
    # the rule is "say why": an optional dependency guarded by
    # try/except ImportError passes untouched, and anything else passes
    # with a `# deferred-import: <reason>` comment. What it stops is the
    # unexplained one, which is what all three of these were.
    #
    # Structural rather than textual: a grep for "    import " misses
    # `from x import y` and matches inside the docstrings of this very
    # module, which discuss imports.
    source = Path(m.__file__).read_text(encoding="utf-8")
    offenders = _function_local_imports(source)
    assert offenders == [], "undeclared function-local imports: " + "; ".join(offenders)


def test_the_deferred_import_escape_hatches_work():
    # A guard nobody can satisfy gets deleted the first time it is
    # inconvenient. Prove both exits are real, against synthetic source,
    # so the rule above is enforceable rather than absolute.
    banned = "def f():\n    import json\n"
    assert _function_local_imports(banned)

    marked = "def f():\n    import json  # deferred-import: breaks a cycle\n"
    assert _function_local_imports(marked) == []

    optional = (
        "def f():\n"
        "    try:\n"
        "        import tomllib\n"
        "    except ImportError:\n"
        "        tomllib = None\n"
    )
    assert _function_local_imports(optional) == []

    optional_tuple = (
        "def f():\n"
        "    try:\n"
        "        import tomllib\n"
        "    except (ImportError, ModuleNotFoundError):\n"
        "        tomllib = None\n"
    )
    assert _function_local_imports(optional_tuple) == []


def test_except_other_than_import_error_does_not_exempt_deferred_import():
    # Regression test: the guard's job is to flag unexplained deferred
    # imports. A handler for some *other* exception type must not be
    # mistaken for the ImportError optional-dependency idiom just because
    # its name happens to contain the substring "Error" — and a bare
    # `except:` must not be treated as catching ImportError either.
    wrong_error_type = (
        "def f():\n"
        "    try:\n"
        "        import json\n"
        "    except ValueError:\n"
        "        json = None\n"
    )
    assert _function_local_imports(wrong_error_type)

    bare_except = (
        "def f():\n"
        "    try:\n"
        "        import json\n"
        "    except:\n"
        "        json = None\n"
    )
    assert _function_local_imports(bare_except)


def test_hoisted_modules_are_actually_used():
    # The counterpart: hoisting only helps if the name is still needed.
    # An unused module-level import is F401, and CI's blocking flake8
    # selection is E9,F63,F7,F82, so nothing else here would notice.
    #
    # A name can be referenced in ways the AST does not surface as a Name
    # node — a string annotation, an __all__ entry — so a bare AST check
    # could fail on an import that is genuinely used. Requiring the name
    # to be absent textually as well makes a false positive much harder,
    # at the cost of missing an import mentioned only in a comment.
    import ast
    import re as _re

    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.value.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    body_without_imports = "\n".join(
        line
        for line in source.splitlines()
        if not _re.match(r"\s*(import|from)\s", line)
    )
    unused = [
        name
        for name in sorted(imported - used)
        if not _re.search(rf"\b{_re.escape(name)}\b", body_without_imports)
    ]
    assert unused == []


_CLAUDE_PRICING_PAGE = (
    "<h2>Claude Opus 5</h2><td>$5 / MTok</td><td>$25 / MTok</td>"
    "<h2>Claude Sonnet 5</h2><td>$2 / MTok</td><td>$10 / MTok</td>"
    "<h2>Claude Haiku 4.5</h2><td>$1 / MTok</td><td>$5 / MTok</td>"
)


def _pricing_file_with_extra_claude_model(tmp_path: Path) -> Path:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "as_of": "2020-01-01",
                "providers": {
                    "claude": {
                        "display_name": "Anthropic Claude",
                        "models": {
                            "opus-5": {
                                "display_name": "Claude Opus 5",
                                "input_per_million": 99.0,
                                "output_per_million": 99.0,
                            },
                            # Dated snapshot in the shipped file that the
                            # scrape does not produce.
                            "sonnet-5-2026-09": {
                                "display_name": "Claude Sonnet 5 (2026-09)",
                                "input_per_million": 3.0,
                                "output_per_million": 15.0,
                            },
                        },
                    },
                    "deepseek": {"display_name": "DeepSeek", "models": {}},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fetch_claude_pricing_keeps_models_it_did_not_scrape(
    tmp_path: Path, monkeypatch
):
    # Assigning a fresh "models" dict drops every key the scrape did not
    # produce. The shipped file carries sonnet-5-2026-09, and a user may
    # have added their own entries; one --update-pricing would delete
    # them with no warning.
    path = _pricing_file_with_extra_claude_model(tmp_path)
    monkeypatch.setattr(
        m.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(_CLAUDE_PRICING_PAGE.encode("utf-8")),
    )
    assert m.fetch_claude_pricing(path) is True

    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["providers"]["claude"]["models"]
    assert models["sonnet-5-2026-09"]["input_per_million"] == 3.0
    # Scraped models are updated in place, not merely added alongside.
    assert models["opus-5"]["input_per_million"] == 5.0
    assert models["opus-5"]["output_per_million"] == 25.0
    # Other providers untouched.
    assert "deepseek" in data["providers"]


def test_fetch_claude_pricing_writes_the_shipped_model_keys(
    tmp_path: Path, monkeypatch
):
    # The keys have to match what pricing.json and users' selected_models
    # already use. Deriving them from display names ("claude-opus-5",
    # "claude-haiku-45") would add a parallel set and orphan the originals.
    path = _pricing_file_with_extra_claude_model(tmp_path)
    monkeypatch.setattr(
        m.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(_CLAUDE_PRICING_PAGE.encode("utf-8")),
    )
    m.fetch_claude_pricing(path)
    models = json.loads(path.read_text(encoding="utf-8"))["providers"]["claude"][
        "models"
    ]
    assert {"opus-5", "sonnet-5", "haiku-4.5"} <= set(models)


@pytest.mark.parametrize(
    "page, why",
    [
        # Output below input: the regex paired a model with the wrong
        # number, or picked up the next model's input rate.
        (
            "<h2>Claude Opus 5</h2>$25 / MTok $5 / MTok"
            "<h2>Claude Sonnet 5</h2>$10 / MTok $2 / MTok",
            "inverted",
        ),
        # Equal: almost certainly the same figure matched twice.
        (
            "<h2>Claude Opus 5</h2>$5 / MTok $5 / MTok"
            "<h2>Claude Sonnet 5</h2>$2 / MTok $2 / MTok",
            "equal",
        ),
        # Out of range: a seat price or an annual figure.
        (
            "<h2>Claude Opus 5</h2>$5000 / MTok $25000 / MTok"
            "<h2>Claude Sonnet 5</h2>$2 / MTok $10 / MTok",
            "too large",
        ),
    ],
)
def test_fetch_claude_pricing_refuses_an_implausible_pair(
    tmp_path: Path, monkeypatch, capsys, page, why
):
    # The docstring promises the shipped values are left in place rather
    # than clobbered with garbage on a page change. Returning False only
    # when *nothing* parses does not deliver that: a reflowed page that
    # parses to the wrong numbers is the more likely failure, and the
    # more damaging one, since every figure the tool prints derives
    # from these.
    path = _pricing_file_with_extra_claude_model(tmp_path)
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        m.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(page.encode("utf-8")),
    )
    assert m.fetch_claude_pricing(path) is False
    assert path.read_text(encoding="utf-8") == before
    assert "implausible" in capsys.readouterr().err


def test_fetch_claude_pricing_accepts_the_real_published_rates(
    tmp_path: Path, monkeypatch
):
    # Guard against a sanity check so tight it rejects reality: the
    # current published rates must pass it.
    path = _pricing_file_with_extra_claude_model(tmp_path)
    monkeypatch.setattr(
        m.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeHTTPResponse(_CLAUDE_PRICING_PAGE.encode("utf-8")),
    )
    assert m.fetch_claude_pricing(path) is True
    models = json.loads(path.read_text(encoding="utf-8"))["providers"]["claude"][
        "models"
    ]
    assert (
        models["sonnet-5"]["input_per_million"],
        models["sonnet-5"]["output_per_million"],
    ) == (2.0, 10.0)
    assert (
        models["haiku-4.5"]["input_per_million"],
        models["haiku-4.5"]["output_per_million"],
    ) == (1.0, 5.0)


def test_load_shipped_pricing_file_is_well_formed():
    pricing = m.load_pricing()
    assert "providers" in pricing
    assert "claude" in pricing["providers"]
    assert "deepseek" in pricing["providers"]
    models = list(m.iter_models(pricing))
    assert len(models) > 0
    for _, _, info in models:
        assert info["input_per_million"] > 0
        assert info["output_per_million"] > 0


def test_load_pricing_raises_config_error_on_invalid_json(tmp_path: Path):
    bad_path = tmp_path / "pricing.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(m.ConfigError, match="not valid JSON"):
        m.load_pricing(bad_path)


@pytest.mark.parametrize("encoding", ["utf-16", "latin-1"])
def test_load_pricing_raises_config_error_on_non_utf8_file(tmp_path: Path, encoding):
    # The file is opened as UTF-8, so any other encoding raises
    # UnicodeDecodeError. It is a ValueError like JSONDecodeError but not a
    # subclass of it, so the JSON handler does not cover it and the file
    # escaped as a raw traceback — the exact outcome issue #61 is about.
    path = tmp_path / "pricing.json"
    path.write_bytes('{"as_of": "caf\u00e9"}'.encode(encoding))
    with pytest.raises(m.ConfigError, match="is not valid UTF-8"):
        m.load_pricing(path)


def test_load_pricing_raises_config_error_when_path_is_a_directory(tmp_path: Path):
    # IsADirectoryError is an OSError but not a FileNotFoundError, so
    # pointing pricing_file at a directory produced a traceback rather than
    # the "pricing file not found" message users would expect to see.
    directory = tmp_path / "pricing.json"
    directory.mkdir()
    with pytest.raises(m.ConfigError, match="could not be read"):
        m.load_pricing(directory)


def test_load_pricing_raises_config_error_on_missing_file(tmp_path: Path):
    with pytest.raises(m.ConfigError, match="pricing file not found"):
        m.load_pricing(tmp_path / "missing-pricing.json")


def test_load_pricing_tolerates_extra_top_level_keys(tmp_path: Path):
    # Regression guard: load_pricing only requires `as_of` and `providers`.
    # Real pricing files may carry extra metadata keys, and tightening
    # validation to reject unknown keys would silently break them.
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "as_of": "2026-01-01",
                "providers": {
                    "claude": {
                        "models": {
                            "opus-5": {
                                "display_name": "Claude Opus 5",
                                "input_per_million": 5.0,
                                "output_per_million": 25.0,
                            }
                        }
                    }
                },
                "extra": "value",
                "notes": "some future metadata",
            }
        ),
        encoding="utf-8",
    )

    pricing = m.load_pricing(pricing_path)
    assert pricing["as_of"] == "2026-01-01"
    assert "claude" in pricing["providers"]


def test_load_pricing_warns_but_loads_when_as_of_missing(tmp_path: Path, capsys):
    # as_of is a staleness signal, not an input to the arithmetic, and the
    # summary line already reads it as .get("as_of", "unknown date"). A
    # hard failure would block a hand-written minimal pricing file over a
    # metadata string, so this warns and proceeds.
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps({"providers": {"claude": {"models": {}}}}), encoding="utf-8"
    )
    assert m.load_pricing(path) == {"providers": {"claude": {"models": {}}}}
    assert "no 'as_of' date" in capsys.readouterr().err


def test_load_pricing_is_quiet_when_as_of_is_present(tmp_path: Path, capsys):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps({"as_of": "2026-01-01", "providers": {}}), encoding="utf-8"
    )
    m.load_pricing(path)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("providers", ["a string", ["a", "list"], 42, None])
def test_load_pricing_rejects_a_non_object_providers(tmp_path: Path, providers):
    # iter_models calls .items() on it, so anything else is an
    # AttributeError several frames away from the file that caused it.
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps({"as_of": "2026-01-01", "providers": providers}), encoding="utf-8"
    )
    with pytest.raises(m.ConfigError, match="'providers' that is not an object"):
        m.load_pricing(path)


def test_load_pricing_raises_config_error_when_providers_missing(tmp_path: Path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({"as_of": "2026-01-01"}), encoding="utf-8")
    with pytest.raises(m.ConfigError, match="providers"):
        m.load_pricing(path)


def test_load_pricing_error_names_the_file(tmp_path: Path):
    # The path matters more than the key list: in a non-interactive run the
    # user may not know which pricing file was picked up.
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({"as_of": "2026-01-01"}), encoding="utf-8")
    with pytest.raises(m.ConfigError, match=str(path)):
        m.load_pricing(path)


def test_load_pricing_raises_config_error_when_top_level_is_not_object(
    tmp_path: Path,
):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(m.ConfigError, match="JSON object"):
        m.load_pricing(path)


def test_load_pricing_accepts_file_with_required_keys(tmp_path: Path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "as_of": "2026-01-01",
                "providers": {
                    "claude": {
                        "models": {
                            "opus-5": {
                                "display_name": "Claude Opus 5",
                                "input_per_million": 5.0,
                                "output_per_million": 25.0,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    data = m.load_pricing(path)
    assert data["as_of"] == "2026-01-01"
    assert "claude" in data["providers"]


# --------------------------------------------------------------------------
# fetch_claude_pricing (mocked urllib — no real network needed)
# --------------------------------------------------------------------------


_CLAUDE_PRICING_HTML = """
<html><body>
<h2>Claude Opus 5</h2>
<p>Input $5.00 per million tokens, Output $25.00 per million tokens</p>
<h2>Claude Sonnet 5</h2>
<p>Input $2.00 per million tokens, Output $10.00 per million tokens</p>
<h2>Claude Haiku 4.5</h2>
<p>Input $1.00 per million tokens, Output $5.00 per million tokens</p>
</body></html>
"""


def test_fetch_claude_pricing_updates_file(monkeypatch, tmp_path: Path):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "providers": {
                    "deepseek": {
                        "display_name": "DeepSeek (direct)",
                        "models": {
                            "deepseek-v4-flash-cache-miss": {
                                "display_name": "DeepSeek Flash",
                                "input_per_million": 0.14,
                                "output_per_million": 0.28,
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(_CLAUDE_PRICING_HTML.encode("utf-8"))

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_claude_pricing(pricing_path) is True

    updated = json.loads(pricing_path.read_text(encoding="utf-8"))
    claude_models = updated["providers"]["claude"]["models"]
    assert claude_models["opus-5"]["input_per_million"] == pytest.approx(5.0)
    assert claude_models["opus-5"]["output_per_million"] == pytest.approx(25.0)
    assert claude_models["sonnet-5"]["input_per_million"] == pytest.approx(2.0)
    assert claude_models["sonnet-5"]["output_per_million"] == pytest.approx(10.0)
    assert claude_models["haiku-4.5"]["input_per_million"] == pytest.approx(1.0)
    assert claude_models["haiku-4.5"]["output_per_million"] == pytest.approx(5.0)
    # Other providers preserved.
    assert "deepseek" in updated["providers"]
    assert updated["source_claude"] == m.CLAUDE_PRICING_URL


def test_fetch_claude_pricing_returns_false_on_network_failure(
    monkeypatch, tmp_path: Path
):
    pricing_path = tmp_path / "pricing.json"
    original = {
        "providers": {
            "claude": {
                "display_name": "Anthropic Claude",
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": 5.0,
                        "output_per_million": 25.0,
                    }
                },
            }
        }
    }
    pricing_path.write_text(json.dumps(original), encoding="utf-8")

    def fake_urlopen(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_claude_pricing(pricing_path) is False
    # Existing entries left untouched.
    assert json.loads(pricing_path.read_text(encoding="utf-8")) == original


def test_fetch_claude_pricing_returns_false_on_unparseable_page(
    monkeypatch, tmp_path: Path
):
    pricing_path = tmp_path / "pricing.json"
    original = {
        "providers": {
            "claude": {
                "display_name": "Anthropic Claude",
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": 5.0,
                        "output_per_million": 25.0,
                    }
                },
            }
        }
    }
    pricing_path.write_text(json.dumps(original), encoding="utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(b"<html><body>no prices here</body></html>")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_claude_pricing(pricing_path) is False
    assert json.loads(pricing_path.read_text(encoding="utf-8")) == original


# --------------------------------------------------------------------------
# Rendering / export
# --------------------------------------------------------------------------


def test_render_table_orders_cheapest_first_and_notes_multiple():
    rows = [
        m.ComparisonRow("Expensive", monthly_cost=100.0, cost_per_million_tokens=10.0),
        m.ComparisonRow("Cheap", monthly_cost=10.0, cost_per_million_tokens=1.0),
    ]
    table = m.render_table(rows)
    assert table.index("Cheap") < table.index("Expensive")
    assert "10.0x" in table


def test_render_table_empty():
    assert "no rows" in m.render_table([])


def test_render_table_never_picks_infeasible_row_as_cheapest():
    rows = [
        m.ComparisonRow(
            "Cheap but infeasible",
            monthly_cost=5.0,
            cost_per_million_tokens=0.01,
            feasible=False,
        ),
        m.ComparisonRow(
            "Real option", monthly_cost=100.0, cost_per_million_tokens=10.0
        ),
    ]
    table = m.render_table(rows)
    assert "Cheapest: Real option" in table
    # Infeasible row still shown, but after the real option, not ranked first.
    assert table.index("Real option") < table.index("Cheap but infeasible")


def test_render_table_shows_no_cheapest_line_when_all_infeasible():
    # Each row's cost is real (what running flat-out 24/7 would cost — see
    # build_local_row), so it's still shown as a normal dollar figure. But
    # with no row that fully covers the workload, there's no meaningful
    # "cheapest full-replacement option" to declare.
    rows = [
        m.ComparisonRow(
            "A", monthly_cost=10.0, cost_per_million_tokens=1.0, feasible=False
        ),
        m.ComparisonRow(
            "B", monthly_cost=5.0, cost_per_million_tokens=0.5, feasible=False
        ),
    ]
    table = m.render_table(rows)
    assert "Cheapest:" not in table
    assert "$10.00" in table and "$5.00" in table


def test_render_table_gbp_currency_uses_pound_symbol():
    rows = [m.ComparisonRow("A", monthly_cost=10.0, cost_per_million_tokens=1.0)]
    table = m.render_table(rows, currency="GBP")
    assert "£10.00" in table
    assert "$" not in table


def test_render_combined_table_is_one_matrix_with_a_column_per_scenario():
    scenario_rows = [
        (
            "Casual",
            [m.ComparisonRow("Local", monthly_cost=10.0, cost_per_million_tokens=1.0)],
        ),
        (
            "Production",
            [m.ComparisonRow("Local", monthly_cost=100.0, cost_per_million_tokens=1.0)],
        ),
    ]
    report = m.render_combined_table(scenario_rows)
    # One row for the option, one column per scenario — not one section per
    # scenario, and the $/1M rate (workload-independent) appears only once.
    assert report.count("Local") == 1
    assert "Casual" in report and "Production" in report
    assert "$10.00" in report and "$100.00" in report
    assert "$1.00" in report  # shared $/1M tokens column, shown once


def test_render_combined_table_shows_real_cost_for_infeasible_cells():
    # A cell for a scenario the local option can't fully cover still shows
    # its real cost (running flat-out 24/7 — see build_local_row), not a
    # placeholder. Callers are expected to have already scaled such
    # scenarios' workloads down to what the hardware can actually produce
    # (see scale_workload_to_local_capacity) before building these rows.
    scenario_rows = [
        (
            "Casual",
            [
                m.ComparisonRow(
                    "Local",
                    monthly_cost=10.0,
                    cost_per_million_tokens=1.0,
                    feasible=True,
                )
            ],
        ),
        (
            "Production",
            [
                m.ComparisonRow(
                    "Local",
                    monthly_cost=100.0,
                    cost_per_million_tokens=1.0,
                    feasible=False,
                )
            ],
        ),
    ]
    report = m.render_combined_table(scenario_rows)
    assert "$100.00" in report
    assert "$10.00" in report


def test_render_combined_table_sorts_rows_by_per_million_rate_ascending():
    scenario_rows = [
        (
            "Casual",
            [
                m.ComparisonRow(
                    "Pricier per token", monthly_cost=5.0, cost_per_million_tokens=9.0
                ),
                m.ComparisonRow(
                    "Cheaper per token", monthly_cost=50.0, cost_per_million_tokens=1.0
                ),
            ],
        )
    ]
    report = m.render_combined_table(scenario_rows)
    assert report.index("Cheaper per token") < report.index("Pricier per token")


def test_render_combined_table_empty():
    assert "no rows" in m.render_combined_table([])


def test_convert_rows_currency_divides_by_rate():
    rows = [m.ComparisonRow("A", monthly_cost=127.0, cost_per_million_tokens=12.7)]
    converted = m.convert_rows_currency(rows, usd_per_gbp=1.27)
    assert converted[0].monthly_cost == pytest.approx(100.0)
    assert converted[0].cost_per_million_tokens == pytest.approx(10.0)
    assert converted[0].name == "A"


def test_convert_rows_currency_rejects_zero_rate():
    rows = [m.ComparisonRow("A", monthly_cost=127.0, cost_per_million_tokens=12.7)]
    with pytest.raises(ValueError, match="usd_per_gbp must be > 0"):
        m.convert_rows_currency(rows, usd_per_gbp=0)


def test_export_csv_and_json(tmp_path: Path):
    rows = [
        m.ComparisonRow("A", 10.0, 1.0, "note-a"),
        m.ComparisonRow("B", 5.0, 0.5, "note-b"),
    ]

    csv_path = tmp_path / "out.csv"
    m.export_csv(rows, csv_path)
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "option" in csv_content
    assert "A" in csv_content and "B" in csv_content
    # cheapest (B) should be written before A
    assert csv_content.index("B") < csv_content.index("A")

    json_path = tmp_path / "out.json"
    m.export_json(rows, json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["option"] == "B"
    assert data[1]["option"] == "A"
    assert data[0]["monthly_cost_usd"] == 5.0


def test_export_csv_and_json_use_currency_suffix(tmp_path: Path):
    rows = [m.ComparisonRow("A", 10.0, 1.0, "note-a")]

    csv_path = tmp_path / "out.csv"
    m.export_csv(rows, csv_path, currency="GBP")
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "monthly_cost_gbp" in csv_content

    json_path = tmp_path / "out.json"
    m.export_json(rows, json_path, currency="GBP")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["monthly_cost_gbp"] == 10.0


def test_export_csv_and_json_export_real_cost_for_infeasible_rows(tmp_path: Path):
    # An infeasible row's monthly_cost is real (the cost of running
    # flat-out, 24/7, all month — see build_local_row), so it's exported
    # as a normal number like any other row, not a placeholder.
    rows = [
        m.ComparisonRow(
            "Infeasible local", 999.0, 0.01, "note-infeasible", feasible=False
        ),
    ]

    csv_path = tmp_path / "out.csv"
    m.export_csv(rows, csv_path)
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "999.0000" in csv_content
    assert "0.0100" in csv_content

    json_path = tmp_path / "out.json"
    m.export_json(rows, json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["monthly_cost_usd"] == 999.0
    assert data[0]["cost_per_million_tokens_usd"] == 0.01


def test_export_combined_csv_and_json_include_scenario_column(tmp_path: Path):
    scenario_rows = [
        ("Casual", [m.ComparisonRow("Local", 10.0, 1.0, "note-a")]),
        ("Production", [m.ComparisonRow("Local", 100.0, 2.0, "note-b")]),
    ]

    csv_path = tmp_path / "out.csv"
    m.export_combined_csv(scenario_rows, csv_path)
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "scenario" in csv_content
    assert "Casual" in csv_content and "Production" in csv_content

    json_path = tmp_path / "out.json"
    m.export_combined_json(scenario_rows, json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert {row["scenario"] for row in data} == {"Casual", "Production"}
    assert all("monthly_cost_usd" in row for row in data)


# --------------------------------------------------------------------------
# GPU detection (mocked subprocess — no real nvidia-smi needed to test)
# --------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_detect_nvidia_gpu_parses_output():
    def fake_runner(*args, **kwargs):
        return _FakeCompletedProcess("NVIDIA GeForce RTX 4090, 24564, 210.5, 450\n")

    info = m.detect_nvidia_gpu(runner=fake_runner)
    assert info == {
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total_mib": 24564.0,
        "power_draw_w": 210.5,
        "power_limit_w": 450.0,
    }


def test_detect_nvidia_gpu_handles_na_power_draw():
    def fake_runner(*args, **kwargs):
        return _FakeCompletedProcess("NVIDIA GeForce RTX 3060, 12288, [N/A], 170\n")

    info = m.detect_nvidia_gpu(runner=fake_runner)
    assert info["power_draw_w"] is None
    assert info["power_limit_w"] == 170.0


def test_detect_nvidia_gpu_returns_none_when_not_found():
    def fake_runner(*args, **kwargs):
        raise FileNotFoundError()

    assert m.detect_nvidia_gpu(runner=fake_runner) is None


def test_detect_nvidia_gpu_returns_none_on_nonzero_exit():
    def fake_runner(*args, **kwargs):
        return _FakeCompletedProcess("", returncode=1)

    assert m.detect_nvidia_gpu(runner=fake_runner) is None


def test_detect_nvidia_gpu_returns_none_on_timeout():
    def fake_runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    assert m.detect_nvidia_gpu(runner=fake_runner) is None


# --------------------------------------------------------------------------
# WMI-based GPU detection fallback (Windows, non-NVIDIA vendors)
# --------------------------------------------------------------------------


def test_detect_gpu_wmi_returns_none_on_non_windows(monkeypatch):
    # Asserting only `is None` cannot fail: on Linux the wmic fallback also
    # returns None, because there is no wmic to run. Verified by deleting
    # the platform guard — the test still passed. What the guard actually
    # buys is not spawning a subprocess at all, so assert that instead.
    monkeypatch.setattr(m.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: pytest.fail("wmic spawned on non-Windows")
    )
    assert m.detect_gpu_wmi() is None


def test_detect_gpu_wmi_uses_wmi_package_when_available(monkeypatch):
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")

    class _FakeController:
        Name = "AMD Radeon RX 7900 XTX"
        AdapterCompatibility = "Advanced Micro Devices, Inc."
        DriverVersion = "31.0.24033.1003"

    class _FakeWmiModule:
        @staticmethod
        def WMI():
            class _Conn:
                @staticmethod
                def Win32_VideoController():
                    return [_FakeController()]

            return _Conn()

    monkeypatch.setitem(__import__("sys").modules, "wmi", _FakeWmiModule)
    info = m.detect_gpu_wmi()
    assert info == {
        "name": "AMD Radeon RX 7900 XTX",
        "vendor": "Advanced Micro Devices, Inc.",
        "driver_version": "31.0.24033.1003",
    }


def test_detect_gpu_wmi_falls_back_to_wmic_when_wmi_package_missing(monkeypatch):
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    # Ensure the `wmi` package import fails so we exercise the wmic path.
    monkeypatch.setitem(__import__("sys").modules, "wmi", None)

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        assert "wmic" in cmd[0]
        assert "/format:list" in cmd
        return _FakeCompletedProcess(
            "AdapterCompatibility=Intel Corporation\r\n"
            "DriverVersion=31.0.101.4502\r\n"
            "Name=Intel(R) UHD Graphics 770\r\n"
            "\r\n"
        )

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    info = m.detect_gpu_wmi()
    assert info == {
        "name": "Intel(R) UHD Graphics 770",
        "vendor": "Intel Corporation",
        "driver_version": "31.0.101.4502",
    }


def test_detect_gpu_wmi_returns_none_when_wmic_unavailable(monkeypatch):
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setitem(__import__("sys").modules, "wmi", None)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    assert m.detect_gpu_wmi() is None


def test_detect_gpu_prefers_nvidia_smi_when_available(monkeypatch):
    def fake_runner(*args, **kwargs):
        return _FakeCompletedProcess("NVIDIA GeForce RTX 4090, 24564, 210.5, 450\n")

    # WMI must not be consulted when nvidia-smi succeeds.
    monkeypatch.setattr(
        m, "detect_gpu_wmi", lambda: (_ for _ in ()).throw(AssertionError())
    )
    info = m.detect_gpu(runner=fake_runner)
    assert info["name"] == "NVIDIA GeForce RTX 4090"


def test_detect_gpu_falls_back_to_wmi_when_nvidia_smi_fails(monkeypatch):
    def fake_runner(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(
        m,
        "detect_gpu_wmi",
        lambda: {
            "name": "AMD Radeon RX 7900 XTX",
            "vendor": "Advanced Micro Devices, Inc.",
            "driver_version": "31.0.24033.1003",
        },
    )
    info = m.detect_gpu(runner=fake_runner)
    assert info["name"] == "AMD Radeon RX 7900 XTX"


def test_detect_gpu_returns_none_when_both_paths_fail(monkeypatch):
    def fake_runner(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(m, "detect_gpu_wmi", lambda: None)
    assert m.detect_gpu(runner=fake_runner) is None


def test_format_gpu_summary_with_all_fields():
    info = {
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total_mib": 24564.0,
        "power_draw_w": 210.5,
        "power_limit_w": 450.0,
    }
    summary = m.format_gpu_summary(info)
    assert "NVIDIA GeForce RTX 4090" in summary
    assert "24564 MiB VRAM" in summary
    assert "210 W draw" in summary
    assert "450 W limit" in summary


def test_format_gpu_summary_handles_none_fields_without_crashing():
    # Regression test: nvidia-smi reporting "[N/A]" (parsed to None by
    # _safe_float) used to crash with TypeError on f"{None:.0f}".
    info = {
        "name": "NVIDIA GeForce RTX 3060",
        "memory_total_mib": None,
        "power_draw_w": None,
        "power_limit_w": 170.0,
    }
    summary = m.format_gpu_summary(info)
    assert "VRAM unknown" in summary
    assert "power draw unknown" in summary
    assert "170 W limit" in summary


# --------------------------------------------------------------------------
# Non-interactive end-to-end run
# --------------------------------------------------------------------------


def test_run_non_interactive_end_to_end(tmp_path: Path, capsys):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "providers": {
                    "claude": {
                        "models": {
                            "opus-5": {
                                "display_name": "Claude Opus 5",
                                "input_per_million": 5.0,
                                "output_per_million": 25.0,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {
            "mode": "own",
            "hardware_cost": 1600,
            "lifetime_years": 3,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
            "tokens_per_sec": 40,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    export_path = tmp_path / "out.json"
    exit_code = m.run_non_interactive(
        config_path, export_fmt="json", export_path=export_path
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Local (buy hardware)" in out
    assert "Claude Opus 5" in out
    assert export_path.exists()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert {row["option"] for row in exported} == {
        "Local (buy hardware)",
        "Claude Opus 5",
    }


# --------------------------------------------------------------------------
# Benchmark helpers (mocked urllib — no real server needed)
# --------------------------------------------------------------------------


def test_validate_http_url_accepts_http_and_https():
    assert m._validate_http_url("http://localhost:11434") == "http://localhost:11434"
    assert m._validate_http_url("https://example.com") == "https://example.com"


def test_validate_http_url_rejects_other_schemes():
    with pytest.raises(ValueError):
        m._validate_http_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        m._validate_http_url("ftp://example.com")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("localhost:11434", "http://localhost:11434"),
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("example.com", "http://example.com"),
        ("  localhost:11434  ", "http://localhost:11434"),
    ],
)
def test_validate_http_url_prepends_http_when_scheme_missing(raw, expected):
    assert m._validate_http_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "http://",
        # Normalizes to "http://://x" — a scheme with no host behind it.
        "://x",
    ],
)
def test_validate_http_url_rejects_input_with_no_host(raw):
    with pytest.raises(ValueError, match="must include a host"):
        m._validate_http_url(raw)


def test_validate_http_url_keeps_path_on_scheme_less_input():
    assert m._validate_http_url("localhost:11434/v1") == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "raw",
    [
        "http://localhost:11434",
        "https://example.com",
        "HTTP://localhost:11434",
    ],
)
def test_validate_http_url_does_not_double_prepend(raw):
    result = m._validate_http_url(raw)
    assert result.lower().startswith(("http://", "https://"))
    assert "http://http" not in result.lower()
    assert "https://http" not in result.lower()


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_benchmark_ollama_computes_tokens_per_sec(monkeypatch):
    body = json.dumps({"eval_count": 100, "eval_duration": 2_000_000_000}).encode(
        "utf-8"
    )

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    tokens_per_sec = m.benchmark_ollama("http://localhost:11434", "llama3")
    # 100 tokens / 2 seconds = 50 tok/s
    assert tokens_per_sec == pytest.approx(50.0)


def test_benchmark_ollama_raises_on_missing_fields(monkeypatch):
    body = json.dumps({"response": "no eval fields here"}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        m.benchmark_ollama("http://localhost:11434", "llama3")


def test_benchmark_ollama_rejects_non_http_url():
    with pytest.raises(ValueError):
        m.benchmark_ollama("file:///etc/passwd", "llama3")


def test_benchmark_openai_compatible_uses_usage_completion_tokens(monkeypatch):
    body = json.dumps(
        {
            "choices": [{"message": {"content": "irrelevant"}}],
            "usage": {"completion_tokens": 42},
        }
    ).encode("utf-8")

    times = iter([100.0, 100.5])  # 0.5s elapsed

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(m.time, "monotonic", lambda: next(times))

    tokens_per_sec = m.benchmark_openai_compatible(
        "http://localhost:1234", "some-model"
    )
    assert tokens_per_sec == pytest.approx(42 / 0.5)


def test_benchmark_openai_compatible_falls_back_to_word_count(monkeypatch):
    body = json.dumps(
        {"choices": [{"message": {"content": "one two three four"}}]}
    ).encode("utf-8")

    times = iter([0.0, 1.0])  # 1s elapsed

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(m.time, "monotonic", lambda: next(times))

    tokens_per_sec = m.benchmark_openai_compatible(
        "http://localhost:1234", "some-model"
    )
    assert tokens_per_sec == pytest.approx(4.0)  # 4 words / 1s


def test_benchmark_openai_compatible_rejects_non_http_url():
    with pytest.raises(ValueError):
        m.benchmark_openai_compatible("ftp://example.com", "some-model")


def test_measure_gpu_power_during_returns_average_reading(monkeypatch):
    # A single peak sample is one noisy driver reading away from being an
    # outlier; averaging every reading taken during the run is what makes
    # the estimate stable between runs of the same hardware.
    readings = iter(
        [
            {"name": "RTX 4090", "power_draw_w": 40.0},
            {"name": "RTX 4090", "power_draw_w": 380.0},
        ]
    )
    polled = threading.Event()

    def next_reading(runner=None):
        polled.set()
        return next(readings, {"name": "RTX 4090", "power_draw_w": 380.0})

    monkeypatch.setattr(m, "detect_nvidia_gpu", next_reading)
    result, avg = m.measure_gpu_power_during(
        lambda: (polled.wait(5), "done")[1],
        runner=lambda *a, **k: None,
        poll_interval=0.01,
    )
    assert result == "done"
    # Only two readings are queued, so the average must be strictly between
    # them (never equal to the peak) regardless of how many polls actually
    # ran before func() returned.
    assert avg is not None and 40.0 <= avg <= 380.0


def test_gpu_poll_interval_default_is_one_second():
    # The change issue #71 asks for. Pinned against the constant and the
    # signature's default, so neither can drift from the other.
    import inspect

    assert m.GPU_POLL_INTERVAL_SECONDS == 1.0
    default = (
        inspect.signature(m.measure_gpu_power_during)
        .parameters["poll_interval"]
        .default
    )
    assert default == m.GPU_POLL_INTERVAL_SECONDS


def test_nvidia_smi_timeout_constant_is_the_one_actually_used(monkeypatch):
    # The constant exists to stop the join timeout and the subprocess
    # timeout drifting apart. It only does that if detect_nvidia_gpu
    # really passes it.
    seen = {}

    class _Result:
        returncode = 0
        stdout = "RTX 4090, 24576, 100.0, 450.0\n"

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        seen["timeout"] = timeout
        return _Result()

    # Set the constant to a value no literal would coincide with.
    # Asserting against its real value proves nothing while that value is
    # still 5.0 — the hardcoded timeout=5 satisfies it equally.
    monkeypatch.setattr(m, "NVIDIA_SMI_TIMEOUT_SECONDS", 12.5)
    m.detect_nvidia_gpu(runner=fake_run)
    assert seen["timeout"] == 12.5


def test_measure_gpu_power_during_samples_before_waiting(monkeypatch):
    # A benchmark shorter than the poll interval must still produce a
    # reading. This is what makes widening 0.5s to 1.0s safe: if the loop
    # waited first, doubling the interval would double the window in which
    # a quick benchmark returns no measurement at all.
    #
    # Checking only that the average is non-None cannot show this — a
    # wait-first loop still appends a reading after func returns but
    # before the join, so the average comes out the same. Verified: that
    # version survives the mutation. So the work itself waits on the
    # poller, and completes only if sampling happened while it ran.
    polled = threading.Event()

    def spy_detect(runner=None):
        polled.set()
        return {"power_draw_w": 321.0}

    def work():
        # With a 30-second interval, a loop that waits before its first
        # sample cannot set this in time, and the result says so rather
        # than the test hanging.
        return "sampled during work" if polled.wait(timeout=5) else "never polled"

    monkeypatch.setattr(m, "detect_nvidia_gpu", spy_detect)
    result, avg = m.measure_gpu_power_during(
        work, runner=lambda *a, **k: None, poll_interval=30.0
    )
    assert result == "sampled during work"
    assert avg == pytest.approx(321.0)


def test_measure_gpu_power_during_stops_polling_when_the_work_finishes(monkeypatch):
    # The poll thread is a daemon, so a leak would not fail the suite —
    # it would just keep spawning nvidia-smi for the rest of the process.
    calls = []

    def counting_detect(runner=None):
        calls.append(1)
        return {"power_draw_w": 100.0}

    monkeypatch.setattr(m, "detect_nvidia_gpu", counting_detect)
    m.measure_gpu_power_during(
        lambda: None, runner=lambda *a, **k: None, poll_interval=0.01
    )
    settled = len(calls)
    time.sleep(0.1)
    assert len(calls) == settled


def test_measure_gpu_power_during_returns_none_average_without_gpu(monkeypatch):
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    result, avg = m.measure_gpu_power_during(
        lambda: 42, runner=lambda *a, **k: None, poll_interval=0.01
    )
    assert result == 42
    assert avg is None


def test_average_gpu_power_w_averages_multiple_samples(monkeypatch):
    readings = iter(
        [
            {"name": "RTX 4090", "power_draw_w": 10.0},
            {"name": "RTX 4090", "power_draw_w": 20.0},
            {"name": "RTX 4090", "power_draw_w": 30.0},
        ]
    )
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: next(readings))
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    avg = m.average_gpu_power_w(samples=3, interval=0.0)
    assert avg == pytest.approx(20.0)


def test_average_gpu_power_w_returns_none_without_gpu(monkeypatch):
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    assert m.average_gpu_power_w(samples=3, interval=0.0) is None


def test_fetch_octopus_agile_rate_parses_current_slot(monkeypatch):
    now = m.datetime.now(m.timezone.utc)
    valid_from = (now.replace(microsecond=0)).isoformat().replace("+00:00", "Z")
    products_body = json.dumps({"results": [{"code": "AGILE-24-10-01"}]}).encode(
        "utf-8"
    )
    rates_body = json.dumps(
        {
            "results": [
                {
                    "valid_from": valid_from,
                    "valid_to": None,
                    "value_inc_vat": 24.83,
                }
            ]
        }
    ).encode("utf-8")

    seen = []

    def fake_urlopen(req, timeout=None):
        # Both Octopus calls now pass a Request rather than a bare URL, so
        # the stub reads req.full_url. Keeping the old `in url` test would
        # raise TypeError, which fetch_octopus_agile_rate's blanket
        # `except Exception` would swallow into a silent None.
        seen.append(req)
        if "standard-unit-rates" in req.full_url:
            return _FakeHTTPResponse(rates_body)
        return _FakeHTTPResponse(products_body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    rate = m.fetch_octopus_agile_rate("C")
    assert rate == pytest.approx(0.2483)

    # Both the product lookup and the rate lookup identify themselves.
    assert len(seen) == 2
    for req in seen:
        assert req.get_header("User-agent") == m.USER_AGENT


def test_user_agent_carries_the_real_version():
    # A User-Agent that misstates its version is worse than none: an
    # operator diagnosing a misbehaving client looks up the wrong code.
    # The original literal "llm-cost-comparison/1.0" was already wrong.
    assert m.USER_AGENT == f"llm-cost-comparison/{m.VERSION}"
    assert "1.0" not in m.USER_AGENT or m.VERSION.startswith("1.0")


def test_fetch_octopus_agile_rate_returns_none_on_failure(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_octopus_agile_rate("C") is None


def test_fetch_octopus_agile_rate_returns_none_when_no_agile_product(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeHTTPResponse(json.dumps({"results": []}).encode("utf-8"))

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_octopus_agile_rate("C") is None


@pytest.fixture(autouse=True)
def _clear_fx_rate_provider_order_cache():
    """Clear the cached FX provider order before and after every test.

    ``_resolve_fx_rate_provider_order()`` is ``lru_cache``-decorated so the
    env var is only read once per process. Any test that mutates
    ``FX_RATE_PROVIDER_ORDER`` (or that relies on the default order) needs a
    fresh resolution, so this fixture clears the cache around each test to
    avoid cross-test contamination.
    """
    m._resolve_fx_rate_provider_order.cache_clear()
    yield
    m._resolve_fx_rate_provider_order.cache_clear()


def test_fetch_fx_rate_parses_response(monkeypatch):
    body = json.dumps(
        {"amount": 1, "base": "GBP", "date": "2026-07-28", "rates": {"USD": 1.27}}
    ).encode("utf-8")

    def fake_urlopen(url, timeout=None):
        assert "from=GBP" in url and "to=USD" in url
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") == pytest.approx(1.27)


def test_fetch_fx_rate_returns_none_on_failure(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") is None


def test_fetch_fx_rate_falls_back_to_next_provider(monkeypatch):
    body = json.dumps({"rates": {"USD": 1.3}}).encode("utf-8")
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if "frankfurter" in url:
            raise OSError("blocked")
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") == pytest.approx(1.3)
    assert len(calls) == 3


def test_fetch_fx_rate_falls_back_to_yahoo_when_all_fx_providers_fail(monkeypatch):
    body = json.dumps(
        {"chart": {"result": [{"meta": {"regularMarketPrice": 1.31}}]}}
    ).encode("utf-8")
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if "yahoo" in url:
            return _FakeHTTPResponse(body)
        raise OSError("blocked")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") == pytest.approx(1.31)
    assert len(calls) == 4


def test_fetch_fx_rate_returns_none_when_yahoo_also_fails(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise OSError("blocked")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") is None


# --------------------------------------------------------------------------
# _resolve_fx_rate_provider_order (env override + caching)
# --------------------------------------------------------------------------


def test_resolve_fx_rate_provider_order_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("FX_RATE_PROVIDER_ORDER", raising=False)
    m._resolve_fx_rate_provider_order.cache_clear()
    assert m._resolve_fx_rate_provider_order() == m.DEFAULT_FX_RATE_PROVIDER_ORDER


def test_resolve_fx_rate_provider_order_honours_env_override(monkeypatch):
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "exchangerate.host,frankfurter.dev")
    m._resolve_fx_rate_provider_order.cache_clear()
    assert m._resolve_fx_rate_provider_order() == (
        "exchangerate.host",
        "frankfurter.dev",
    )


def test_resolve_fx_rate_provider_order_drops_unknown_keys(monkeypatch):
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "not-a-real-provider,frankfurter.app")
    m._resolve_fx_rate_provider_order.cache_clear()
    assert m._resolve_fx_rate_provider_order() == ("frankfurter.app",)


def test_resolve_fx_rate_provider_order_falls_back_when_all_keys_unknown(
    monkeypatch,
):
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "typo1,typo2")
    m._resolve_fx_rate_provider_order.cache_clear()
    assert m._resolve_fx_rate_provider_order() == m.DEFAULT_FX_RATE_PROVIDER_ORDER


def test_resolve_fx_rate_provider_order_falls_back_when_env_empty(monkeypatch):
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "")
    m._resolve_fx_rate_provider_order.cache_clear()
    assert m._resolve_fx_rate_provider_order() == m.DEFAULT_FX_RATE_PROVIDER_ORDER


def test_resolve_fx_rate_provider_order_is_cached(monkeypatch):
    # First call resolves and caches; a subsequent env mutation without
    # clearing the cache must not change the returned order.
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "frankfurter.dev")
    m._resolve_fx_rate_provider_order.cache_clear()
    first = m._resolve_fx_rate_provider_order()
    assert first == ("frankfurter.dev",)

    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "exchangerate.host")
    second = m._resolve_fx_rate_provider_order()
    assert second == first  # cached — env change ignored until cache_clear()

    info = m._resolve_fx_rate_provider_order.cache_info()
    assert info.hits >= 1


def test_fetch_fx_rate_uses_env_override_order(monkeypatch):
    # With the env override set, only the named provider should be tried
    # before falling through to Yahoo.
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "exchangerate.host")
    m._resolve_fx_rate_provider_order.cache_clear()

    body = json.dumps({"rates": {"USD": 1.42}}).encode("utf-8")
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if "exchangerate.host" in url:
            return _FakeHTTPResponse(body)
        raise OSError("should not be called")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") == pytest.approx(1.42)
    assert len(calls) == 1
    assert "exchangerate.host" in calls[0]


def test_default_provider_order_matches_url_templates_order():
    """Locks in the "no-env behaviour must remain byte-identical" constraint:
    the no-env default must resolve providers in the same order as the
    original ``FX_RATE_URL_TEMPLATES`` tuple, so a future reorder of one
    without the other doesn't silently change which provider is tried first.
    """
    assert m.DEFAULT_FX_RATE_PROVIDER_ORDER == tuple(
        m.FX_RATE_PROVIDER_TEMPLATES.keys()
    )
    for key, url_template in zip(
        m.DEFAULT_FX_RATE_PROVIDER_ORDER, m.FX_RATE_URL_TEMPLATES
    ):
        assert m.FX_RATE_PROVIDER_TEMPLATES[key] == url_template


def test_fetch_fx_rate_tries_yahoo_after_env_override_provider_fails(monkeypatch):
    # With the env override set to a single provider that fails, Yahoo must
    # still be tried — the override must not skip the Yahoo last resort.
    monkeypatch.setenv("FX_RATE_PROVIDER_ORDER", "exchangerate.host")
    m._resolve_fx_rate_provider_order.cache_clear()

    body = json.dumps(
        {"chart": {"result": [{"meta": {"regularMarketPrice": 1.31}}]}}
    ).encode("utf-8")
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if "yahoo" in url:
            return _FakeHTTPResponse(body)
        raise OSError("blocked")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.fetch_fx_rate("GBP", "USD") == pytest.approx(1.31)
    assert len(calls) == 2
    assert "exchangerate.host" in calls[0]
    assert "yahoo" in calls[1]


# --------------------------------------------------------------------------
# prompt_float minimum enforcement
# --------------------------------------------------------------------------


def test_prompt_float_rejects_below_minimum(monkeypatch, capsys):
    answers = iter(["0", "-5", "10"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    value = m.prompt_float("Tokens/sec", minimum=0.001)
    assert value == 10.0
    out = capsys.readouterr().out
    assert out.count(">= 0.001") == 2  # rejected "0" and "-5" before accepting "10"


def test_prompt_float_accepts_default_without_minimum_check(monkeypatch):
    # An empty answer takes the default even when a minimum is set: defaults
    # are author-supplied and therefore trusted, so they skip the check.
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert m.prompt_float("x", default=5.0, minimum=1.0) == 5.0


def test_prompt_float_propagates_eof(monkeypatch):
    # Exhausted stdin must surface as EOFError, not silently return the
    # default — otherwise an enclosing `while True` loop can spin forever.
    def raise_eof(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)
    with pytest.raises(EOFError):
        m.prompt_float("x", default=5.0)


def test_prompt_yes_no_propagates_eof(monkeypatch):
    def raise_eof(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)
    with pytest.raises(EOFError):
        m.prompt_yes_no("Continue?", default=True)


def test_prompt_choice_propagates_eof(monkeypatch):
    def raise_eof(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)
    with pytest.raises(EOFError):
        m.prompt_choice("Pick", ["a", "b"], default="a")


def test_run_interactive_terminates_cleanly_on_exhausted_stdin(monkeypatch, capsys):
    # Regression test for the unbounded `while True` confirmation loops:
    # when stdin is exhausted mid-setup, run_interactive must return
    # non-zero with a clear message rather than hang or raise.
    def raise_eof(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)
    # No saved last-run file, so the flow goes straight to interactive setup.
    monkeypatch.setattr(m, "load_last_run", lambda path=None: None)

    exit_code = m.run_interactive()
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "stdin closed" in err


# --------------------------------------------------------------------------
# prompt_choice case-insensitive matching
# --------------------------------------------------------------------------


def test_prompt_choice_lowercase_input_lowercase_choices(monkeypatch):
    # Baseline: existing callers passing lowercase choices and lowercase
    # input must keep working unchanged.
    monkeypatch.setattr("builtins.input", lambda _: "ollama")
    assert m.prompt_choice("Pick", ["ollama", "other"]) == "ollama"


def test_prompt_choice_uppercase_input_lowercase_choices(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "OLLAMA")
    assert m.prompt_choice("Pick", ["ollama", "other"]) == "ollama"


def test_prompt_choice_mixed_case_input_lowercase_choices(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Ollama")
    assert m.prompt_choice("Pick", ["ollama", "other"]) == "ollama"


def test_prompt_choice_mixed_case_choice_matched_case_insensitively(monkeypatch):
    # The original-cased choice must be returned, not the casefolded input.
    monkeypatch.setattr("builtins.input", lambda _: "newprovider")
    assert m.prompt_choice("Pick", ["NewProvider", "other"]) == "NewProvider"


def test_prompt_choice_mixed_case_choice_exact_case_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "NewProvider")
    assert m.prompt_choice("Pick", ["NewProvider", "other"]) == "NewProvider"


def test_prompt_choice_prompt_preserves_original_casing(monkeypatch):
    # Display casing is a separate concern from matching: the prompt shown to
    # the user must list the original-cased choices, not the casefolded ones.
    captured = {}

    def fake_input(prompt):
        captured["prompt"] = prompt
        return "newprovider"

    monkeypatch.setattr("builtins.input", fake_input)
    m.prompt_choice("Pick", ["NewProvider", "other"])
    assert "NewProvider" in captured["prompt"]
    assert "newprovider" not in captured["prompt"]


def test_prompt_choice_reprompts_on_invalid_input(monkeypatch, capsys):
    # prompt_choice loops until it gets a valid answer; it does NOT fall back
    # to the default on invalid (non-empty) input. Feed one invalid value
    # followed by a valid one, and assert the error message lists the
    # original-cased choices.
    inputs = iter(["not-a-choice", "newprovider"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    result = m.prompt_choice("Pick", ["NewProvider", "other"])
    assert result == "NewProvider"
    out = capsys.readouterr().out
    assert "NewProvider" in out
    assert "newprovider" not in out


def test_prompt_choice_empty_input_returns_default(monkeypatch):
    # The default is only used when the user submits an empty answer.
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert m.prompt_choice("Pick", ["NewProvider", "other"], default="other") == "other"


# --------------------------------------------------------------------------
# Confirmation of manually-entered electricity / exchange rates
# --------------------------------------------------------------------------


def test_prompt_yes_no_accepts_yes_and_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert m.prompt_yes_no("ok?") is True
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert m.prompt_yes_no("ok?") is False


def test_prompt_yes_no_empty_uses_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert m.prompt_yes_no("ok?", default=True) is True
    assert m.prompt_yes_no("ok?", default=False) is False


# --------------------------------------------------------------------------
# Non-interactive config validation
# --------------------------------------------------------------------------


def _write_pricing(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                # Mirrors the shipped pricing.json, which carries as_of.
                "as_of": "2026-01-01",
                "providers": {
                    "claude": {
                        "models": {
                            "opus-5": {
                                "display_name": "Claude Opus 5",
                                "input_per_million": 5.0,
                                "output_per_million": 25.0,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_run_non_interactive_rent_mode(tmp_path: Path, capsys):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Local (rented cloud GPU)" in out


def test_run_non_interactive_rent_mode_missing_hourly_rate_raises_config_error(
    tmp_path: Path,
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40},  # hourly_rate omitted
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="hourly_rate"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_missing_workload_field_raises_config_error(tmp_path: Path):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
        },  # avg_output_tokens missing
        "local": {
            "mode": "own",
            "tokens_per_sec": 40,
            "hardware_cost": 1600,
            "lifetime_years": 3,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="avg_output_tokens"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_unknown_mode_raises_config_error(tmp_path: Path):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "bogus", "tokens_per_sec": 40},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="own.*rent"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_resolves_relative_pricing_file_against_config_dir(
    tmp_path: Path, capsys
):
    # pricing.json lives next to the config file, not the process cwd.
    subdir = tmp_path / "configs"
    subdir.mkdir()
    _write_pricing(subdir / "pricing.json")

    config_path = subdir / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": "pricing.json",  # relative — must resolve against config_path.parent
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "Claude Opus 5" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main() CLI export-flag handling
# --------------------------------------------------------------------------


def test_main_version_flag_prints_version_and_exits_zero(capsys, monkeypatch):
    # argparse's "version" action prints to stdout and raises SystemExit(0).
    # %(prog)s is derived from sys.argv[0], which under pytest is the test
    # runner rather than this script, so pin it. Asserting only the version
    # suffix would let a reordered or truncated format string through, which
    # is exactly the regression this smoke test exists to catch.
    monkeypatch.setattr(sys, "argv", ["llm_cost_comparison.py", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        m.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"llm_cost_comparison.py {m.VERSION}\n"


def test_main_non_interactive_export_without_path_defaults(
    tmp_path: Path, monkeypatch, capsys
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    exit_code = m.main(
        ["--non-interactive", "--config", str(config_path), "--export", "csv"]
    )
    assert exit_code == 0
    assert (tmp_path / "cost_comparison.csv").exists()


def test_version_attribute_is_an_alias_not_a_second_literal():
    # "is a non-empty string" would pass for any hardcoded value, which is
    # exactly what must not be here: a literal alongside pyproject.toml
    # goes stale the first time someone releases without editing both.
    # Identity, not equality, so the two names cannot drift apart.
    assert m.__version__ is m.VERSION
    assert isinstance(m.__version__, str) and m.__version__


def test_version_comes_from_package_metadata_or_the_source_sentinel():
    # Either the installed distribution's version, or the sentinel used
    # when running from a checkout. Anything else means someone reinstated
    # a literal.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        expected = pkg_version("llm-cost-comparison")
    except PackageNotFoundError:
        expected = "0.0.0+unknown"
    assert m.VERSION == expected


def test_main_version_flag_prints_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        m.main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # argparse's %(prog)s prefix names the program alongside the number. A
    # bare version string is ambiguous the moment it is pasted into a bug
    # report, which is the use case the issue names. prog is derived from
    # sys.argv[0], so it is pinned against that rather than hardcoded —
    # under pytest it is the runner's name, not the script's.
    expected_prog = os.path.basename(sys.argv[0])
    assert out.strip() == f"{expected_prog} {m.__version__}"
    assert out.strip() != m.__version__


def test_main_help_flag_still_works(capsys):
    with pytest.raises(SystemExit) as excinfo:
        m.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # Existing flags must still be advertised.
    assert "--version" in out
    assert "--non-interactive" in out
    assert "--update-pricing" in out


def test_main_non_interactive_config_error_reports_and_exits_nonzero(
    tmp_path: Path, capsys
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40},  # missing hourly_rate
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.main(["--non-interactive", "--config", str(config_path)])
    assert exit_code == 1
    assert "hourly_rate" in capsys.readouterr().err


def test_main_non_interactive_missing_pricing_file_reports_and_exits_nonzero(
    tmp_path: Path, capsys
):
    """Issue #33 end-to-end: a missing pricing.json in --non-interactive mode
    must print a clear, user-facing message (not a raw traceback) and exit
    non-zero -- via main()'s existing ConfigError handler, since
    run_non_interactive itself re-raises rather than swallowing it."""
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": "does_not_exist.json",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.main(["--non-interactive", "--config", str(config_path)])
    assert exit_code == 1
    assert "pricing file not found" in capsys.readouterr().err


def test_run_non_interactive_missing_pricing_file_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": "does_not_exist.json",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="pricing file not found"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_missing_config_file_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "does_not_exist.json"
    with pytest.raises(m.ConfigError, match="config file not found"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_invalid_json_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(m.ConfigError, match="not valid JSON"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_own_mode_rejects_non_numeric_field(tmp_path: Path):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {
            "mode": "own",
            "tokens_per_sec": 40,
            "hardware_cost": "1600",
            "lifetime_years": 3,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(m.ConfigError, match="hardware_cost"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize("bad_value", [0, -5, "fast", True, 0.0009, 0.0001])
def test_run_non_interactive_rejects_invalid_tokens_per_sec(tmp_path: Path, bad_value):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": bad_value, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="tokens_per_sec"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_static_currency_converts_table_and_export(
    tmp_path: Path, capsys
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
        "currency": "GBP",
        "static_fx_rate": 0.8,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    export_path = tmp_path / "out.json"
    exit_code = m.run_non_interactive(
        config_path, export_fmt="json", export_path=export_path
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "£" in out
    assert "$" not in out

    # The symbols alone prove nothing about the arithmetic. With
    # static_fx_rate 0.8 meaning "0.8 GBP per USD", the Claude Opus 5 row
    # costs $300/month in USD (1000 req/day * 500 in + 300 out tokens,
    # priced at $5/$25 per million), so it must read £240 — not £375,
    # which is what dividing by the rate instead of multiplying produced.
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert all("monthly_cost_gbp" in row for row in data)
    assert all("monthly_cost_usd" not in row for row in data)
    hosted = next(row for row in data if "Opus" in row["option"])
    assert hosted["monthly_cost_gbp"] == pytest.approx(240.0)
    assert "£240.00" in out
    # A GBP figure must be smaller than the USD one it came from, since a
    # pound buys more than a dollar. An inverted rate makes it larger,
    # which is the sanity check that catches the direction regardless of
    # the exact numbers.
    assert hosted["monthly_cost_gbp"] < 300.0


@pytest.mark.parametrize("rate", [True, False, 0, -1, "0.8", None, [0.8]])
def test_run_non_interactive_rejects_a_non_positive_or_non_numeric_rate(
    tmp_path: Path, rate
):
    # True is the interesting one: bool subclasses int, so without the
    # explicit guard "static_fx_rate": true is accepted as a rate of 1.0
    # and the table silently claims GBP figures that are really USD.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload": {
                    "requests_per_day": 1000,
                    "avg_input_tokens": 500,
                    "avg_output_tokens": 300,
                },
                "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
                "pricing_file": str(pricing_path),
                "currency": "GBP",
                "static_fx_rate": rate,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(m.ConfigError, match="static_fx_rate must be a positive number"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_static_fx_rate_is_units_of_currency_per_usd(tmp_path: Path, capsys):
    # The direction, pinned on its own so it cannot be lost in a test that
    # is also checking exports. Same config priced at two rates: doubling
    # the rate must double the displayed figure.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)

    def run(rate):
        config_path = tmp_path / f"config-{rate}.json"
        config_path.write_text(
            json.dumps(
                {
                    "workload": {
                        "requests_per_day": 1000,
                        "avg_input_tokens": 500,
                        "avg_output_tokens": 300,
                    },
                    "local": {
                        "mode": "rent",
                        "tokens_per_sec": 40,
                        "hourly_rate": 2.5,
                    },
                    "pricing_file": str(pricing_path),
                    "currency": "GBP",
                    "static_fx_rate": rate,
                }
            ),
            encoding="utf-8",
        )
        export_path = tmp_path / f"out-{rate}.json"
        m.run_non_interactive(config_path, export_fmt="json", export_path=export_path)
        capsys.readouterr()
        data = json.loads(export_path.read_text(encoding="utf-8"))
        return next(r for r in data if "Opus" in r["option"])["monthly_cost_gbp"]

    assert run(1.0) == pytest.approx(300.0)  # parity with USD
    assert run(2.0) == pytest.approx(600.0)  # twice as many units per dollar
    assert run(0.5) == pytest.approx(150.0)


@pytest.mark.parametrize(
    "currency", ["", "pounds", "£", "US", "USDD", 42, None, ["GBP"]]
)
def test_run_non_interactive_rejects_a_non_currency_code(tmp_path: Path, currency):
    # render_table prints an unrecognised code verbatim as its own symbol,
    # so "pounds 12.34" would render happily rather than fail. The code has
    # to be rejected at config time.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload": {
                    "requests_per_day": 1000,
                    "avg_input_tokens": 500,
                    "avg_output_tokens": 300,
                },
                "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
                "pricing_file": str(pricing_path),
                "currency": currency,
                "static_fx_rate": 0.8,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(m.ConfigError, match="three-letter currency code"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_accepts_a_currency_without_a_symbol(
    tmp_path: Path, capsys
):
    # EUR has no entry in CURRENCY_SYMBOLS. render_table falls back to
    # printing the code, which is a reasonable answer, so a valid code
    # must not be rejected just because no symbol is defined for it.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload": {
                    "requests_per_day": 1000,
                    "avg_input_tokens": 500,
                    "avg_output_tokens": 300,
                },
                "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
                "pricing_file": str(pricing_path),
                "currency": "eur",
                "static_fx_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    assert m.run_non_interactive(config_path, export_fmt=None, export_path=None) == 0
    out = capsys.readouterr().out
    # Lower case in the config, upper case in the table.
    assert "EUR 150.00" in out


def test_run_non_interactive_currency_without_static_rate_raises_config_error(
    tmp_path: Path,
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
        "currency": "GBP",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="static_fx_rate"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_defaults_to_usd_without_static_rate(
    tmp_path: Path, capsys
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "$" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["own", "existing", "rent"])
@pytest.mark.parametrize("bad_value", [0, 0.0009, 0.0001])
def test_run_non_interactive_rejects_below_minimum_tokens_per_sec_in_every_mode(
    tmp_path: Path, mode, bad_value
):
    # The interactive prompt enforces a minimum of 0.001 on tokens/sec
    # regardless of which hardware mode was chosen; the non-interactive path
    # must reject the same bad values in every mode, not just the one
    # exercised above.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    local_cfg = {"mode": mode, "tokens_per_sec": bad_value}
    if mode == "own":
        local_cfg.update(
            {
                "hardware_cost": 1600,
                "lifetime_years": 3,
                "power_watts": 450,
                "electricity_rate_per_kwh": 0.15,
            }
        )
    elif mode == "existing":
        local_cfg.update({"power_watts": 450, "electricity_rate_per_kwh": 0.15})
    else:  # rent
        local_cfg.update({"hourly_rate": 2.5})
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": local_cfg,
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="tokens_per_sec"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize("mode", ["own", "existing", "rent"])
def test_run_non_interactive_accepts_minimum_tokens_per_sec_in_every_mode(
    tmp_path: Path, mode, capsys
):
    # 0.001 is the exact boundary the interactive prompt accepts (its
    # `minimum=0.001` check is inclusive), so the non-interactive path must
    # accept it too — otherwise the two modes would disagree on the same
    # value.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    local_cfg = {"mode": mode, "tokens_per_sec": 0.001}
    if mode == "own":
        local_cfg.update(
            {
                "hardware_cost": 1600,
                "lifetime_years": 3,
                "power_watts": 450,
                "electricity_rate_per_kwh": 0.15,
            }
        )
    elif mode == "existing":
        local_cfg.update({"power_watts": 450, "electricity_rate_per_kwh": 0.15})
    else:  # rent
        local_cfg.update({"hourly_rate": 2.5})
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": local_cfg,
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "Claude Opus 5" in capsys.readouterr().out


def test_run_non_interactive_rejects_zero_total_workload_tokens(tmp_path: Path):
    # A workload with no input and no output used to surface as the vague
    # "zero total tokens" error. It is now caught by the per-field rule, which
    # names the offending field (issue #36).
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        m.ConfigError, match=r"workload\.avg_input_tokens must be a positive number"
    ):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("requests_per_day", 0),
        ("requests_per_day", -1),
        ("avg_input_tokens", 0),
        ("avg_input_tokens", -5),
        # avg_output_tokens == 0 is deliberately NOT included here — it's a
        # legitimate value (classification-only workload), covered by
        # test_run_non_interactive_allows_zero_output_tokens_for_input_only_workload
        # below. Only a negative value is invalid.
        ("avg_output_tokens", -3),
    ],
)
def test_run_non_interactive_rejects_nonpositive_workload_field(
    tmp_path: Path, field, bad_value
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    workload = {
        "requests_per_day": 1000,
        "avg_input_tokens": 500,
        "avg_output_tokens": 300,
    }
    workload[field] = bad_value
    config = {
        "workload": workload,
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    if bad_value < 0:
        # Negative values are rejected earlier, by the per-field type/range
        # check in _resolve_workload_scenarios, which uses the unprefixed
        # "{field} must be ..." wording (see
        # test_run_non_interactive_rejects_bad_workload_field).
        with pytest.raises(
            m.ConfigError,
            match=rf"^{field} must be a non-negative number, got {bad_value!r}$",
        ):
            m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    else:
        # Zero passes the non-negative check above but is still rejected by
        # _validate_workload's stricter positivity rule for these two
        # fields, which keeps the "workload."-prefixed wording.
        with pytest.raises(
            m.ConfigError, match=rf"workload\.{field} must be a positive number"
        ):
            m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_all_shipped_presets_pass_validation():
    # The positivity rules apply to preset shapes too, so a shipped preset with
    # a zero/negative field would start failing at config-resolution time.
    for preset in m.WORKLOAD_PRESETS:
        m._validate_workload(preset.to_workload())


@pytest.mark.parametrize("shape", ["workload_preset", "workload_presets"])
def test_resolve_workload_scenarios_validates_preset_shapes(monkeypatch, shape):
    # The reason validation lives in _resolve_workload_scenarios rather than
    # run_non_interactive is that it then covers the preset shapes as well.
    bad = m.WorkloadPreset(
        key="broken",
        label="Broken",
        description="Preset with no input tokens.",
        requests_per_day=100,
        avg_input_tokens=0,
        avg_output_tokens=300,
    )
    monkeypatch.setattr(m, "WORKLOAD_PRESETS", (bad,))
    config = {shape: "broken" if shape == "workload_preset" else ["broken"]}

    with pytest.raises(
        m.ConfigError, match=r"workload\.avg_input_tokens must be a positive number"
    ):
        m._resolve_workload_scenarios(config)


@pytest.mark.parametrize(
    "field", ["requests_per_day", "avg_input_tokens", "avg_output_tokens"]
)
def test_run_non_interactive_rejects_bool_workload_field(tmp_path: Path, field):
    # bool is a subclass of int, so True/False would otherwise pass the numeric
    # type check and be silently treated as 1/0.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    workload = {
        "requests_per_day": 1000,
        "avg_input_tokens": 500,
        "avg_output_tokens": 300,
    }
    workload[field] = True
    config = {
        "workload": workload,
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match=rf"workload\.{field} must be a number"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize("bad_value", [-1, "many"])
@pytest.mark.parametrize(
    "field_name", ["requests_per_day", "avg_input_tokens", "avg_output_tokens"]
)
def test_run_non_interactive_rejects_bad_workload_field(
    tmp_path: Path, field_name, bad_value
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    workload = {
        "requests_per_day": 1000,
        "avg_input_tokens": 500,
        "avg_output_tokens": 300,
    }
    workload[field_name] = bad_value
    config = {
        "workload": workload,
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Per-field wording (no "workload." prefix) — pins the exact message
    # shape, including the field name and the offending value, so the
    # format can't silently drift back to a prefixed or value-less variant
    # for any of the three fields, not just the one originally exercised.
    with pytest.raises(
        m.ConfigError,
        match=rf"^{field_name} must be a non-negative number, got {bad_value!r}$",
    ):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


def test_run_non_interactive_allows_zero_output_tokens_for_input_only_workload(
    tmp_path: Path, capsys
):
    # avg_output_tokens == 0 alone is legitimate (e.g. a classification-only
    # workload) as long as total tokens/month is still positive.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 0,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "Claude Opus 5" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Workload presets
# --------------------------------------------------------------------------


def test_get_preset_returns_matching_preset():
    preset = m.get_preset("casual")
    assert preset.key == "casual"
    assert preset.to_workload().monthly_total_tokens > 0


def test_get_preset_raises_config_error_for_unknown_key():
    with pytest.raises(m.ConfigError, match="unknown workload preset"):
        m.get_preset("does_not_exist")


def test_every_preset_has_positive_total_tokens():
    for preset in m.WORKLOAD_PRESETS:
        assert preset.to_workload().monthly_total_tokens > 0


# --------------------------------------------------------------------------
# interactive_workload scenario menu
# --------------------------------------------------------------------------

_NUM_PRESETS = len(m.WORKLOAD_PRESETS)
_ALL_OPTION = str(_NUM_PRESETS + 1)
_CUSTOM_OPTION = str(_NUM_PRESETS + 2)


def test_interactive_workload_selects_single_preset(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    scenarios = m.interactive_workload()
    assert len(scenarios) == 1
    key, label, workload = scenarios[0]
    assert key == m.WORKLOAD_PRESETS[0].key
    assert label == m.WORKLOAD_PRESETS[0].label
    assert workload.monthly_total_tokens > 0


def test_interactive_workload_compare_all_returns_every_preset(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: _ALL_OPTION)
    scenarios = m.interactive_workload()
    assert [key for key, _, _ in scenarios] == [p.key for p in m.WORKLOAD_PRESETS]


def test_interactive_workload_custom_reprompts_on_zero_total_tokens(
    monkeypatch, capsys
):
    # Select "custom", then: requests_per_day=0 -> zero total tokens (reprompt),
    # then valid answers.
    answers = iter([_CUSTOM_OPTION, "0", "500", "300", "1000", "500", "300"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    scenarios = m.interactive_workload()
    assert len(scenarios) == 1
    key, label, workload = scenarios[0]
    assert key == "custom"
    assert workload.requests_per_day == 1000
    assert workload.monthly_total_tokens > 0
    assert "zero total tokens" in capsys.readouterr().out


def test_interactive_workload_custom_accepts_zero_output_tokens_alone(monkeypatch):
    # avg_output_tokens == 0 alone is fine as long as total tokens is positive.
    answers = iter([_CUSTOM_OPTION, "1000", "500", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    scenarios = m.interactive_workload()
    _key, _label, workload = scenarios[0]
    assert workload.avg_output_tokens == 0
    assert workload.monthly_total_tokens > 0


def test_render_table_omits_multiple_line_for_single_row():
    rows = [m.ComparisonRow("Only option", 42.0, 1.0)]
    table = m.render_table(rows)
    assert "Only option" in table
    assert "most expensive option is" not in table


# --------------------------------------------------------------------------
# interactive_local_setup — "existing hardware" branch
# --------------------------------------------------------------------------


def _stub_gpu_info(name="NVIDIA GeForce RTX 3090"):
    return {
        "name": name,
        "memory_total_mib": 24576.0,
        "power_draw_w": 200.0,
        "power_limit_w": 350.0,
    }


# Answers for a full "existing hardware" pass. Matched against prompt text
# rather than fed positionally: a positional list fails as a bare
# StopIteration naming neither the prompt nor the drift, which is how the
# first version of these tests failed.
_LOCAL_SETUP_ANSWERS = {
    # Not skipped, so GPU detection and the benchmark question are both asked.
    "Skip benchmark": "n",
    "auto-detect your GPU": "y",
    "benchmark a running local model endpoint": "n",
    "Measured or estimated tokens/sec": "40",
    "Hardware mode": "existing",
    # Keep the run offline: decline the live Octopus lookup, and pay in USD so
    # no FX conversion is attempted.
    "Look up your current unit rate live": "n",
    "Do you pay for electricity in GBP": "n",
    "Electricity rate": "0.15",
    # The two power prompts take the defaults derived from GPU detection —
    # which is exactly what the detected/undetected assertions below compare.
    "Extra power draw while generating": "",
    "Total system power draw while running": "",
    # Accept the rate as entered. The decline path is exercised separately
    # by the _local_setup_with tests below.
    "Use this electricity rate?": "",
}


def _local_setup(monkeypatch, gpu_info, answers=None):
    """Run interactive_local_setup offline with GPU detection stubbed.

    Both benchmark entry points are replaced with stubs that fail the test if
    called: the "existing hardware" branch must never reach them.
    """
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: gpu_info)

    def _no_benchmark(*args, **kwargs):
        raise AssertionError("benchmark must not run in the existing-hardware branch")

    monkeypatch.setattr(m, "benchmark_ollama", _no_benchmark)
    monkeypatch.setattr(m, "benchmark_openai_compatible", _no_benchmark)
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: pytest.fail("network call"))

    script = {**_LOCAL_SETUP_ANSWERS, **(answers or {})}
    pending = {
        fragment: list(value) if isinstance(value, list) else None
        for fragment, value in script.items()
    }

    def fake_input(prompt: str = "") -> str:
        for fragment, answer in script.items():
            if fragment in prompt:
                queued = pending[fragment]
                if queued is None:
                    return answer
                if not queued:
                    pytest.fail(f"ran out of scripted answers for prompt: {prompt!r}")
                return queued.pop(0)
        pytest.fail(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    return m.interactive_local_setup()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://0.0.0.0:8000/v1",
        "http://[::1]:8000/v1",
        # urlsplit normalises the host to lower case, so no explicit
        # fold is needed in the helper. Pinned here because removing
        # one that is not needed should not be able to break this.
        "http://LOCALHOST:8000/v1",
    ],
)
def test_no_wall_clock_caveat_for_a_loopback_endpoint(base_url):
    # The README's own position: on loopback there is no real network hop,
    # so wall-clock is a fair proxy. Warning anyway is noise, and noise
    # trains people to skip the warning that does matter.
    assert m.wall_clock_benchmark_caveat(base_url) is None


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.com/v1",
        "http://192.168.1.50:8000/v1",
        "http://gpu-box.lan:11434/v1",
    ],
)
def test_wall_clock_caveat_for_a_remote_endpoint(base_url):
    caveat = m.wall_clock_benchmark_caveat(base_url)
    assert caveat is not None
    assert "network latency" in caveat


def test_wall_clock_caveat_handles_an_unparseable_url():
    # The caveat is decoration on a benchmark that already succeeded, so a
    # URL it cannot parse must not take the run down with it.
    assert m.wall_clock_benchmark_caveat("http://[") is not None


def test_benchmark_openai_compatible_does_not_print(monkeypatch, capsys):
    # The caveat belongs to the caller that displays the number. A library
    # function that prints cannot be reused by anything that formats its
    # own output, and printing before returning put the caveat above the
    # figure it qualifies.
    body = json.dumps(
        {
            "choices": [{"message": {"content": "one two three"}}],
            "usage": {"completion_tokens": 3},
        }
    ).encode("utf-8")
    monkeypatch.setattr(
        m.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(body)
    )
    rate = m.benchmark_openai_compatible("https://api.example.com/v1", "gpt-x")
    assert rate > 0
    assert capsys.readouterr().out == ""


def _benchmark_setup(monkeypatch, base_url, backend="openai"):
    """interactive_local_setup driven through the endpoint-benchmark branch."""
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(m, "detect_gpu_wmi", lambda: None)
    monkeypatch.setattr(m, "average_gpu_power_w", lambda *a, **k: None)
    monkeypatch.setattr(m, "measure_gpu_power_during", lambda fn: (fn(), None))
    monkeypatch.setattr(m, "benchmark_openai_compatible", lambda *a, **k: 37.0)
    monkeypatch.setattr(m, "benchmark_ollama", lambda *a, **k: 37.0)
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: pytest.fail("network call"))
    answers = {
        "Skip benchmark": "n",
        "auto-detect your GPU": "n",
        "benchmark a running local model endpoint": "y",
        "Backend": backend,
        "Base URL": base_url,
        "Model name as served locally": "some-model",
        "Hardware mode": "existing",
        "Look up your current unit rate live": "n",
        "Do you pay for electricity in GBP": "n",
        "Electricity rate": "0.15",
        "Use this electricity rate?": "y",
        "Extra power draw while generating": "",
        "Total system power draw while running": "",
    }

    def fake_input(prompt: str = "") -> str:
        for fragment, answer in answers.items():
            if fragment in prompt:
                return answer
        pytest.fail(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    return m.interactive_local_setup()


def _local_setup_with_gpu_source(monkeypatch, nvidia, wmi):
    """interactive_local_setup with both detection paths stubbed."""
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: nvidia)
    monkeypatch.setattr(m, "detect_gpu_wmi", lambda: wmi)
    monkeypatch.setattr(
        m, "average_gpu_power_w", lambda *a, **k: pytest.fail("nvidia-smi polled")
    )
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: pytest.fail("network call"))

    def fake_input(prompt: str = "") -> str:
        for fragment, answer in _LOCAL_SETUP_ANSWERS.items():
            if fragment in prompt:
                return answer
        pytest.fail(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    return m.interactive_local_setup()


def test_interactive_setup_uses_a_wmi_detected_card(monkeypatch, capsys):
    # The point of issue #62. detect_gpu existed but nothing called it, so
    # an AMD card on Windows was still invisible to the only code path a
    # user reaches. Stub nvidia-smi as absent and WMI as present.
    amd = {
        "name": "AMD Radeon RX 7900 XTX",
        "vendor": "Advanced Micro Devices, Inc.",
        "driver_version": "31.0.24027.1012",
    }
    _local_setup_with_gpu_source(monkeypatch, nvidia=None, wmi=amd)
    out = capsys.readouterr().out
    assert "AMD Radeon RX 7900 XTX" in out
    assert "No GPU detected" not in out


def test_wmi_detected_card_does_not_trigger_nvidia_smi_power_polling(
    monkeypatch, capsys
):
    # average_gpu_power_w shells out to nvidia-smi. A WMI card has no power
    # telemetry to average, so polling would spawn a binary that is not
    # there. The stub above fails the test if it is called at all.
    _local_setup_with_gpu_source(
        monkeypatch,
        nvidia=None,
        wmi={"name": "Intel Arc A770", "vendor": "Intel", "driver_version": "1.0"},
    )
    # Reaching here means average_gpu_power_w was never called. The card is
    # still reported, with the fields WMI cannot supply marked unknown.
    out = capsys.readouterr().out
    assert "Intel Arc A770" in out
    assert "VRAM unknown" in out
    assert "power draw unknown" in out


def test_no_gpu_message_names_both_detection_paths(monkeypatch, capsys):
    _local_setup_with_gpu_source(monkeypatch, nvidia=None, wmi=None)
    out = capsys.readouterr().out
    assert "nvidia-smi and WMI both returned nothing" in out


def test_detect_gpu_wmi_handles_commas_inside_wmic_fields(monkeypatch):
    # AdapterCompatibility for an AMD card is "Advanced Micro Devices,
    # Inc." — and wmic does not quote it. A CSV row therefore has more
    # commas than columns, and no split recovers the fields: splitting
    # unlimited mis-assigns them, splitting with a limit hands the leftover
    # to the last column. /format:list sidesteps it entirely.
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "wmi", None)

    class _Result:
        returncode = 0
        stdout = (
            "\r\n"
            "AdapterCompatibility=Advanced Micro Devices, Inc.\r\n"
            "DriverVersion=31.0.24027.1012\r\n"
            "Name=AMD Radeon RX 7900 XTX\r\n"
            "\r\n"
        )

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Result())
    assert m.detect_gpu_wmi() == {
        "name": "AMD Radeon RX 7900 XTX",
        "vendor": "Advanced Micro Devices, Inc.",
        "driver_version": "31.0.24027.1012",
    }


def test_detect_gpu_wmi_returns_the_first_named_adapter(monkeypatch):
    # A laptop typically lists the integrated chip and the discrete card.
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "wmi", None)

    class _Result:
        returncode = 0
        stdout = (
            "AdapterCompatibility=Intel Corporation\r\n"
            "DriverVersion=31.0.101\r\n"
            "Name=Intel(R) UHD Graphics\r\n"
            "\r\n"
            "AdapterCompatibility=Advanced Micro Devices, Inc.\r\n"
            "DriverVersion=31.0.24027\r\n"
            "Name=AMD Radeon RX 7900 XTX\r\n"
            "\r\n"
        )

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Result())
    assert m.detect_gpu_wmi()["name"] == "Intel(R) UHD Graphics"


def test_detect_gpu_wmi_returns_none_when_wmic_names_nothing(monkeypatch):
    monkeypatch.setattr(m.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "wmi", None)

    class _Result:
        returncode = 0
        stdout = "AdapterCompatibility=Intel\r\nDriverVersion=1.0\r\n\r\n"

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Result())
    assert m.detect_gpu_wmi() is None


def test_gpu_detection_prompt_matches_what_detect_gpu_tries():
    # The prompt said "an NVIDIA GPU via nvidia-smi" while a WMI fallback
    # was added underneath it, which would have told Windows AMD users the
    # question did not apply to them.
    assert "nvidia-smi" in m.GPU_DETECTION_PROMPT
    assert "WMI" in m.GPU_DETECTION_PROMPT
    assert "an NVIDIA GPU" not in m.GPU_DETECTION_PROMPT


def test_caveat_is_printed_after_the_throughput_for_a_remote_endpoint(
    monkeypatch, capsys
):
    # The caveat has to reach the user, not merely exist as a helper, and
    # it has to come after the number it qualifies.
    _benchmark_setup(monkeypatch, "https://api.example.com/v1")
    out = capsys.readouterr().out
    assert "Measured throughput: 37.0 tokens/sec" in out
    assert "network latency" in out
    assert out.index("Measured throughput") < out.index("network latency")


def test_no_caveat_printed_for_a_loopback_endpoint(monkeypatch, capsys):
    _benchmark_setup(monkeypatch, "http://localhost:11434/v1")
    out = capsys.readouterr().out
    assert "Measured throughput: 37.0 tokens/sec" in out
    assert "network latency" not in out


def test_no_caveat_printed_for_the_ollama_backend(monkeypatch, capsys):
    # benchmark_ollama reports generation-only time from eval_duration, so
    # the wall-clock caveat does not apply to it at all — remote or not.
    _benchmark_setup(monkeypatch, "https://ollama.example.com", backend="ollama")
    out = capsys.readouterr().out
    assert "Measured throughput: 37.0 tokens/sec" in out
    assert "network latency" not in out


def test_interactive_local_setup_existing_hardware_branch(monkeypatch, capsys):
    row_builder, display_currency, usd_per_gbp, tokens_per_sec, settings = _local_setup(
        monkeypatch, _stub_gpu_info()
    )

    # The documented return shape: (row_builder, display_currency, usd_per_gbp,
    # tokens_per_sec, settings). "existing" is carried in settings, not as a
    # bare tuple element.
    assert callable(row_builder)
    assert display_currency == "USD"
    assert usd_per_gbp is None
    assert tokens_per_sec == 40.0
    assert settings["mode"] == "existing"
    assert settings["tokens_per_sec"] == 40.0
    assert settings["electricity_rate_per_kwh"] == 0.15

    # The detected card is surfaced to the user and drives the power defaults.
    out = capsys.readouterr().out
    assert "RTX 3090" in out
    assert settings["power_watts_extra"] == 150
    assert settings["power_watts_total"] == 250.0


def test_interactive_local_setup_existing_hardware_with_no_gpu_detected(
    monkeypatch, capsys
):
    _, _, _, tokens_per_sec, settings = _local_setup(monkeypatch, None)

    assert settings["mode"] == "existing"
    assert tokens_per_sec == 40.0

    # With nothing detected the user is told so, and the power defaults fall
    # back to the generic figures rather than the card-derived ones above.
    out = capsys.readouterr().out
    assert "No GPU detected" in out
    assert settings["power_watts_extra"] == 250.0
    assert settings["power_watts_total"] == 350.0


def test_declining_the_usd_rate_confirmation_reprompts_and_uses_the_new_value(
    monkeypatch, capsys
):
    # The point of issue #54: a mistyped rate must be correctable before it
    # reaches the table. Enter $99/kWh, decline, enter $0.15, accept.
    _, _, _, _, settings = _local_setup(
        monkeypatch,
        _stub_gpu_info(),
        answers={
            "Electricity rate": ["99", "0.15"],
            "Use this electricity rate?": ["n", "y"],
        },
    )
    assert settings["electricity_rate_per_kwh"] == 0.15
    out = capsys.readouterr().out
    # Both the rejected and the accepted value were shown back for review;
    # confirming a value the user never saw would be no confirmation at all.
    assert "$99.0000/kWh" in out
    assert "$0.1500/kWh" in out
    assert "Re-entering the value." in out


def test_accepting_the_usd_rate_confirmation_asks_exactly_once(monkeypatch, capsys):
    # The happy path must not become a two-step flow for users with nothing
    # to correct: Enter accepts, and the summary is printed once.
    _, _, _, _, settings = _local_setup(monkeypatch, _stub_gpu_info())
    assert settings["electricity_rate_per_kwh"] == 0.15
    assert capsys.readouterr().out.count("Electricity rate: $") == 1


def _gbp_setup(monkeypatch, answers):
    """interactive_local_setup down the GBP branch, with FX stubbed."""
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: _stub_gpu_info())
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: 1.30)
    script = {
        "Skip benchmark": "y",
        "auto-detect your GPU": "y",
        "Measured or estimated tokens/sec": "40",
        "Hardware mode": "existing",
        "Look up your current unit rate live": "n",
        "Do you pay for electricity in GBP": "y",
        "Extra power draw while generating": "",
        "Total system power draw while running": "",
        **answers,
    }
    pending = {f: list(v) if isinstance(v, list) else None for f, v in script.items()}

    def fake_input(prompt: str = "") -> str:
        for fragment, answer in script.items():
            if fragment in prompt:
                queued = pending[fragment]
                if queued is None:
                    return answer
                if not queued:
                    pytest.fail(f"ran out of scripted answers for prompt: {prompt!r}")
                return queued.pop(0)
        pytest.fail(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    return m.interactive_local_setup()


def test_declining_the_gbp_confirmation_reprompts_both_interdependent_values(
    monkeypatch, capsys
):
    # The displayed electricity rate is gbp_rate * usd_per_gbp, so the two
    # values cannot be corrected independently — declining must re-ask both.
    _, display_currency, usd_per_gbp, _, settings = _gbp_setup(
        monkeypatch,
        {
            "Electricity rate (GBP/kWh)": ["0.30", "0.20"],
            "GBP→USD exchange rate": ["9.99", "1.25"],
            "Use this electricity rate and exchange rate?": ["n", "y"],
        },
    )
    assert display_currency == "GBP"
    assert usd_per_gbp == 1.25
    assert settings["electricity_rate_per_kwh"] == pytest.approx(0.20 * 1.25)
    assert "Re-entering both values." in capsys.readouterr().out


def test_declining_the_gbp_confirmation_does_not_default_to_the_rejected_rate(
    monkeypatch,
):
    # Pressing Enter at the re-prompt must not hand back the value just
    # rejected. The default offered second time is the same one offered
    # first time (0.2483), not the 0.30 the user turned down.
    prompts = []
    real_prompt_float = m.prompt_float

    def recording_prompt_float(prompt, default=None, minimum=None):
        prompts.append((prompt, default))
        return real_prompt_float(prompt, default=default, minimum=minimum)

    monkeypatch.setattr(m, "prompt_float", recording_prompt_float)
    _, _, _, _, settings = _gbp_setup(
        monkeypatch,
        {
            # "" on the second pass takes whatever default is offered.
            "Electricity rate (GBP/kWh)": ["0.30", ""],
            "GBP→USD exchange rate": ["1.25", "1.25"],
            "Use this electricity rate and exchange rate?": ["n", "y"],
        },
    )
    rate_defaults = [d for p, d in prompts if "Electricity rate (GBP/kWh)" in p]
    assert rate_defaults == [0.2483, 0.2483]
    assert settings["electricity_rate_per_kwh"] == pytest.approx(0.2483 * 1.25)


# --------------------------------------------------------------------------
# interactive_provider_selection input validation
# --------------------------------------------------------------------------


def _sample_pricing() -> dict:
    return {
        "providers": {
            "claude": {
                "models": {
                    "opus-5": {
                        "display_name": "Claude Opus 5",
                        "input_per_million": 5.0,
                        "output_per_million": 25.0,
                    },
                    "haiku-4.5": {
                        "display_name": "Claude Haiku 4.5",
                        "input_per_million": 1.0,
                        "output_per_million": 5.0,
                    },
                }
            },
            "deepseek": {
                "models": {
                    "deepseek-v3": {
                        "display_name": "DeepSeek-V3",
                        "input_per_million": 0.27,
                        "output_per_million": 1.10,
                    }
                }
            },
        }
    }


def test_interactive_provider_selection_all_returns_none(monkeypatch):
    # Answering "yes" to "compare against all" short-circuits to None.
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert m.interactive_provider_selection(_sample_pricing()) is None


def test_interactive_provider_selection_accepts_valid_keys(monkeypatch):
    # "no" to all, then a valid comma-separated list.
    answers = iter(["n", "claude/opus-5, deepseek/deepseek-v3"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    assert selected == {"claude/opus-5", "deepseek/deepseek-v3"}


def test_interactive_provider_selection_is_case_insensitive(monkeypatch):
    answers = iter(["n", "Claude/Opus-5"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    # Canonical (lowercase) key is returned, not the user's casing.
    assert selected == {"claude/opus-5"}


def test_interactive_provider_selection_blank_returns_empty_set(monkeypatch):
    answers = iter(["n", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert m.interactive_provider_selection(_sample_pricing()) == set()


def test_interactive_provider_selection_warns_and_reprompts_on_unknown_key(
    monkeypatch, capsys
):
    # First attempt has a typo; user chooses to re-enter, then gives a valid key.
    answers = iter(["n", "claude/opus-6", "y", "claude/opus-5"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    assert selected == {"claude/opus-5"}
    out = capsys.readouterr().out
    assert "Unrecognized key" in out
    assert "claude/opus-6" in out
    # The valid keys must be listed *with the warning*, not merely somewhere
    # in the output: every key already appears in the menu printed before
    # the prompt, so asserting "claude/opus-5" in out would pass with the
    # listing deleted. Anchor on the header and check what follows it.
    listing = out.split("Valid keys are:")[1]
    assert "claude/opus-5" in listing
    assert "claude/haiku-4.5" in listing
    assert "deepseek/deepseek-v3" in listing


def test_interactive_provider_selection_keeps_recognized_keys_on_decline(
    monkeypatch, capsys
):
    # Mixed valid + invalid; user declines to re-enter, so only the valid key
    # is kept (and the invalid one is reported).
    answers = iter(["n", "claude/opus-5, not-a-real-key", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    assert selected == {"claude/opus-5"}
    out = capsys.readouterr().out
    assert "not-a-real-key" in out
    assert "Unrecognized key" in out


def test_interactive_provider_selection_all_unknown_declines_returns_empty(
    monkeypatch, capsys
):
    answers = iter(["n", "bogus/one, bogus/two", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    assert selected == set()
    out = capsys.readouterr().out
    assert "bogus/one" in out and "bogus/two" in out
    # An empty set is indistinguishable from "leave blank for local only",
    # so the user is told that is what they are getting.
    assert "comparing local options only" in out


def test_interactive_provider_selection_does_not_silently_keep_unknown_keys(
    monkeypatch,
):
    # The behaviour issue #56 is about. Previously every entered string went
    # into the returned set verbatim, so a typo produced a selection that
    # matched no model and a comparison quietly missing it. The unknown key
    # must not survive into the result under any answer to the re-prompt.
    answers = iter(["n", "claude/opus-5, claude/opus-6", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    selected = m.interactive_provider_selection(_sample_pricing())
    assert "claude/opus-6" not in selected


def test_interactive_provider_selection_tolerates_spacing_and_duplicates(monkeypatch):
    answers = iter(["n", "  claude/opus-5 ,, claude/opus-5 , deepseek/deepseek-v3  "])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert m.interactive_provider_selection(_sample_pricing()) == {
        "claude/opus-5",
        "deepseek/deepseek-v3",
    }


def test_interactive_provider_selection_reprompt_can_be_declined_after_retrying(
    monkeypatch,
):
    # Two bad attempts in a row: the loop must keep asking rather than give
    # up after one round, and the second decline must still return what was
    # recognized on that final attempt.
    answers = iter(["n", "bogus/one", "y", "claude/opus-5, bogus/two", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert m.interactive_provider_selection(_sample_pricing()) == {"claude/opus-5"}


# --------------------------------------------------------------------------
# "Already own the hardware" cost mode (electricity only, no amortization)
# --------------------------------------------------------------------------


def test_local_monthly_cost_existing_hardware_is_electricity_only():
    # 1 kW * $0.10/hr * 10 hr = $1.00, with zero fixed/amortized cost
    cost = m.local_monthly_cost_existing_hardware(
        power_watts=1000, electricity_rate_per_kwh=0.10, hours_needed_per_month=10
    )
    assert cost == pytest.approx(1.0)


def test_local_monthly_cost_existing_hardware_zero_hours_is_free():
    cost = m.local_monthly_cost_existing_hardware(
        power_watts=450, electricity_rate_per_kwh=0.15, hours_needed_per_month=0
    )
    assert cost == 0.0


def test_build_local_row_existing_hardware():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(
        w,
        tokens_per_sec=100,
        mode="existing",
        power_watts=450,
        electricity_rate_per_kwh=0.15,
    )
    assert row.name == "Local (already-on PC)"
    assert row.monthly_cost > 0
    # No amortization component: cheaper than the "buying" mode for the same power/rate.
    buying_row = m.build_local_row(
        w,
        tokens_per_sec=100,
        mode="own",
        hardware_cost=1600,
        lifetime_years=3,
        power_watts=450,
        electricity_rate_per_kwh=0.15,
    )
    assert row.monthly_cost < buying_row.monthly_cost


def test_build_local_row_existing_hardware_accepts_name_override():
    w = m.Workload(requests_per_day=100, avg_input_tokens=500, avg_output_tokens=500)
    row = m.build_local_row(
        w,
        tokens_per_sec=100,
        mode="existing",
        power_watts=450,
        electricity_rate_per_kwh=0.15,
        name="Local (custom label)",
    )
    assert row.name == "Local (custom label)"


def test_run_non_interactive_existing_mode(tmp_path: Path, capsys):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {
            "mode": "existing",
            "tokens_per_sec": 40,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "already-on PC" in capsys.readouterr().out


def test_run_non_interactive_existing_mode_missing_fields_raises_config_error(
    tmp_path: Path,
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {
            "mode": "existing",
            "tokens_per_sec": 40,
        },  # missing power_watts, rate
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="power_watts"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


# --------------------------------------------------------------------------
# GPU price/power lookup
# --------------------------------------------------------------------------


def test_lookup_gpu_defaults_matches_known_card():
    result = m.lookup_gpu_defaults("NVIDIA GeForce RTX 4090")
    assert result is not None
    cost, power = result
    assert cost > 0
    assert power > 0


def test_lookup_gpu_defaults_returns_none_for_unknown_card():
    assert m.lookup_gpu_defaults("Some Future GPU Nobody Has Heard Of") is None


def test_lookup_gpu_defaults_case_insensitive():
    assert m.lookup_gpu_defaults("nvidia geforce rtx 4090") is not None


def test_load_gpu_defaults_reads_shipped_json_file():
    defaults = m.load_gpu_defaults()
    assert len(defaults) > 0
    labels = [label for label, _cost, _power in defaults]
    assert "RTX 4090" in labels
    for label, cost, power in defaults:
        assert isinstance(label, str) and label
        assert cost > 0
        assert power > 0


def test_load_gpu_defaults_falls_back_when_file_missing(tmp_path: Path, capsys):
    defaults = m.load_gpu_defaults(tmp_path / "does_not_exist.json")
    assert defaults == m._FALLBACK_GPU_COST_POWER_DEFAULTS
    # An absent file is the out-of-the-box state, not a user mistake, so it
    # must stay silent — unlike every other unusable-file case below.
    assert capsys.readouterr().err == ""


def test_load_gpu_defaults_falls_back_on_invalid_json(tmp_path: Path, capsys):
    bad_path = tmp_path / "gpu_power_defaults.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    assert m.load_gpu_defaults(bad_path) == m._FALLBACK_GPU_COST_POWER_DEFAULTS
    assert "using built-in GPU defaults" in capsys.readouterr().err


@pytest.mark.parametrize("encoding", ["utf-16", "latin-1"])
def test_load_gpu_defaults_falls_back_on_non_utf8_file(
    tmp_path: Path, capsys, encoding
):
    # The file is opened as UTF-8, so any other encoding raises
    # UnicodeDecodeError — a ValueError, not an OSError, and therefore not
    # caught by the obvious `except (json.JSONDecodeError, OSError)`.
    # "Present but unusable" must warn and fall back, not traceback.
    path = tmp_path / "gpu_power_defaults.json"
    path.write_bytes(
        '{"gpus": [{"label": "RTX 4090", "cost_usd": 1600.0, "power_watts": 450.0}]}'
        "\n// caf\u00e9".encode(encoding)
    )
    assert m.load_gpu_defaults(path) == m._FALLBACK_GPU_COST_POWER_DEFAULTS
    assert "could not be read" in capsys.readouterr().err


@pytest.mark.parametrize(
    "contents, expected_problem",
    [
        # Valid JSON, but not a JSON object. The original code called
        # ``.get`` on the parsed value, so these raised AttributeError
        # instead of falling back as the docstring promised.
        ("[1, 2, 3]", "must contain a JSON object, got list"),
        ('"hello"', "must contain a JSON object, got str"),
        ("42", "must contain a JSON object, got int"),
        ("null", "must contain a JSON object, got NoneType"),
        # Object, but "gpus" is not a list.
        ('{"gpus": {"label": "RTX 4090"}}', '"gpus" must be a list, got dict'),
        # Object with no usable entries at all.
        ('{"gpus": []}', "listed no usable GPU entries"),
    ],
)
def test_load_gpu_defaults_falls_back_on_wrong_shape(
    tmp_path: Path, capsys, contents, expected_problem
):
    path = tmp_path / "gpu_power_defaults.json"
    path.write_text(contents, encoding="utf-8")
    assert m.load_gpu_defaults(path) == m._FALLBACK_GPU_COST_POWER_DEFAULTS
    assert expected_problem in capsys.readouterr().err


@pytest.mark.parametrize(
    "entry",
    [
        {"cost_usd": 100.0, "power_watts": 50.0},  # no label
        {"label": "X", "power_watts": 50.0},  # no cost
        {"label": "X", "cost_usd": 100.0},  # no power
        {"label": "   ", "cost_usd": 100.0, "power_watts": 50.0},  # blank label
        {"label": "X", "cost_usd": "not a number", "power_watts": 50.0},
        {"label": "X", "cost_usd": None, "power_watts": 50.0},
        {"label": "X", "cost_usd": True, "power_watts": 50.0},  # bool is not 1.0
        {"label": "X", "cost_usd": 0, "power_watts": 50.0},  # free GPU
        {"label": "X", "cost_usd": -100.0, "power_watts": 50.0},
        {"label": "X", "cost_usd": 100.0, "power_watts": 0},  # zero draw
        {"label": "X", "cost_usd": 100.0, "power_watts": -50.0},
        "not an object",
    ],
)
def test_load_gpu_defaults_skips_bad_entry_but_keeps_the_rest(
    tmp_path: Path, capsys, entry
):
    path = tmp_path / "gpu_power_defaults.json"
    good = {"label": "GOOD CARD", "cost_usd": 200.0, "power_watts": 60.0}
    path.write_text(json.dumps({"gpus": [entry, good]}), encoding="utf-8")

    # The rest of the file survives — one typo does not discard the user's
    # whole customisation — but the skip is reported, so they are never
    # left wondering why their card stopped matching.
    assert m.load_gpu_defaults(path) == (("GOOD CARD", 200.0, 60.0),)
    assert "skipping gpus[0]" in capsys.readouterr().err


def test_load_gpu_defaults_upper_cases_user_supplied_labels(tmp_path: Path):
    # lookup_gpu_defaults upper-cases the detected card name, so a
    # lower-case label in the user's file could never match before. The
    # README documents this matching as case-insensitive.
    path = tmp_path / "gpu_power_defaults.json"
    path.write_text(
        json.dumps(
            {"gpus": [{"label": "rtx 5090", "cost_usd": 2.0, "power_watts": 3.0}]}
        ),
        encoding="utf-8",
    )
    defaults = m.load_gpu_defaults(path)
    assert defaults == (("RTX 5090", 2.0, 3.0),)
    assert m.lookup_gpu_defaults("NVIDIA GeForce RTX 5090", defaults=defaults) == (
        2.0,
        3.0,
    )


def test_fallback_gpu_defaults_match_shipped_json():
    # The fallback tuple duplicates the shipped JSON so the script still
    # works if the file goes missing. Duplication is only safe while the
    # two agree, so this asserts they do.
    assert m.load_gpu_defaults() == m._FALLBACK_GPU_COST_POWER_DEFAULTS


def test_gpu_cost_power_defaults_constant_reflects_shipped_json():
    # The public constant kept its name across the move to a config file,
    # and now carries what the file says rather than a hardcoded copy.
    assert m.GPU_COST_POWER_DEFAULTS == m.load_gpu_defaults()
    assert m.lookup_gpu_defaults("NVIDIA GeForce RTX 4090") == (1600.0, 450.0)


def test_load_gpu_defaults_reads_custom_file(tmp_path: Path):
    custom_path = tmp_path / "gpu_power_defaults.json"
    custom_path.write_text(
        json.dumps(
            {
                "as_of": "2026-07-28",
                "note": "custom",
                "gpus": [
                    {"label": "MY CUSTOM GPU", "cost_usd": 123.0, "power_watts": 45.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    defaults = m.load_gpu_defaults(custom_path)
    assert defaults == (("MY CUSTOM GPU", 123.0, 45.0),)
    assert m.lookup_gpu_defaults("my custom gpu", defaults=defaults) == (123.0, 45.0)


def test_lookup_gpu_defaults_accepts_explicit_defaults_tuple():
    custom = (("FAKE CARD", 999.0, 111.0),)
    assert m.lookup_gpu_defaults("Fake Card 9000", defaults=custom) == (999.0, 111.0)
    assert m.lookup_gpu_defaults("Something Else", defaults=custom) is None


# --------------------------------------------------------------------------
# Rest-of-system power allowance (laptop vs desktop)
# --------------------------------------------------------------------------


def test_rest_of_system_allowance_uses_laptop_default_for_laptop_gpu():
    gpu_info = {"name": "NVIDIA GeForce RTX 5070 Laptop GPU"}
    assert m.rest_of_system_allowance_w(gpu_info) == m.LAPTOP_REST_OF_SYSTEM_W


def test_rest_of_system_allowance_uses_desktop_default_for_desktop_gpu():
    gpu_info = {"name": "NVIDIA GeForce RTX 4090"}
    assert m.rest_of_system_allowance_w(gpu_info) == m.DESKTOP_REST_OF_SYSTEM_W


def test_rest_of_system_allowance_uses_desktop_default_when_no_gpu_detected():
    assert m.rest_of_system_allowance_w(None) == m.DESKTOP_REST_OF_SYSTEM_W


def test_rest_of_system_allowance_is_case_insensitive():
    gpu_info = {"name": "nvidia geforce rtx 5070 laptop gpu"}
    assert m.rest_of_system_allowance_w(gpu_info) == m.LAPTOP_REST_OF_SYSTEM_W


# --------------------------------------------------------------------------
# Local model discovery (mocked urllib — no real server needed)
# --------------------------------------------------------------------------


def test_list_ollama_models_parses_tags_response(monkeypatch):
    body = json.dumps(
        {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.list_ollama_models("http://localhost:11434") == ["llama3:8b", "mistral:7b"]


def test_list_running_ollama_models_parses_ps_response(monkeypatch):
    body = json.dumps({"models": [{"name": "llama3:8b"}]}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.list_running_ollama_models("http://localhost:11434") == ["llama3:8b"]


def test_list_openai_compatible_models_parses_models_response(monkeypatch):
    body = json.dumps(
        {"data": [{"id": "local-model-a"}, {"id": "local-model-b"}]}
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.list_openai_compatible_models("http://localhost:1234") == [
        "local-model-a",
        "local-model-b",
    ]


def test_discover_local_models_prefers_running_over_installed_for_ollama(monkeypatch):
    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/ps"):
            return _FakeHTTPResponse(
                json.dumps({"models": [{"name": "running-model"}]}).encode()
            )
        return _FakeHTTPResponse(
            json.dumps(
                {"models": [{"name": "installed-a"}, {"name": "installed-b"}]}
            ).encode()
        )

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.discover_local_models("ollama", "http://localhost:11434") == [
        "running-model"
    ]


def test_discover_local_models_falls_back_to_installed_when_none_running(monkeypatch):
    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/api/ps"):
            return _FakeHTTPResponse(json.dumps({"models": []}).encode())
        return _FakeHTTPResponse(
            json.dumps({"models": [{"name": "installed-a"}]}).encode()
        )

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.discover_local_models("ollama", "http://localhost:11434") == [
        "installed-a"
    ]


def test_discover_local_models_returns_empty_list_on_failure(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    assert m.discover_local_models("ollama", "http://localhost:11434") == []
    assert m.discover_local_models("openai", "http://localhost:1234") == []


# --------------------------------------------------------------------------
# _resolve_workload_scenarios
# --------------------------------------------------------------------------


def test_resolve_workload_scenarios_explicit_workload():
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        }
    }
    scenarios = m._resolve_workload_scenarios(config)
    assert len(scenarios) == 1
    assert scenarios[0][0] == "custom"


def test_resolve_workload_scenarios_single_preset():
    scenarios = m._resolve_workload_scenarios({"workload_preset": "coding_agent"})
    assert len(scenarios) == 1
    assert scenarios[0][0] == "coding_agent"


def test_resolve_workload_scenarios_multiple_presets():
    scenarios = m._resolve_workload_scenarios(
        {"workload_presets": ["casual", "team_tool"]}
    )
    assert [key for key, _, _ in scenarios] == ["casual", "team_tool"]


def test_resolve_workload_scenarios_requires_exactly_one_source():
    with pytest.raises(m.ConfigError, match="must include one of"):
        m._resolve_workload_scenarios({})
    with pytest.raises(m.ConfigError, match="must include only one of"):
        m._resolve_workload_scenarios(
            {
                "workload": {
                    "requests_per_day": 1,
                    "avg_input_tokens": 1,
                    "avg_output_tokens": 1,
                },
                "workload_preset": "casual",
            }
        )


def test_resolve_workload_scenarios_unknown_preset_key_raises():
    with pytest.raises(m.ConfigError, match="unknown workload preset"):
        m._resolve_workload_scenarios({"workload_preset": "not_a_real_preset"})


def test_resolve_workload_scenarios_always_returns_workload_objects():
    # Contract relied on by run_non_interactive's validation loop: the third
    # element of every scenario tuple must be a Workload instance (never a
    # dict or other mapping), so attribute access is safe.
    for config in (
        {
            "workload": {
                "requests_per_day": 1000,
                "avg_input_tokens": 500,
                "avg_output_tokens": 300,
            }
        },
        {"workload_preset": "casual"},
        {"workload_presets": ["casual", "coding_agent"]},
    ):
        scenarios = m._resolve_workload_scenarios(config)
        assert scenarios, f"expected at least one scenario for {config!r}"
        for _key, _label, workload in scenarios:
            assert isinstance(workload, m.Workload), (
                f"scenario workload for {config!r} is "
                f"{type(workload).__name__}, not Workload"
            )
            # The three fields the validation loop reads must be present.
            assert hasattr(workload, "requests_per_day")
            assert hasattr(workload, "avg_input_tokens")
            assert hasattr(workload, "avg_output_tokens")


def test_run_non_interactive_raises_config_error_for_mapping_shaped_workload(
    tmp_path: Path, monkeypatch
):
    # Regression test: if _resolve_workload_scenarios ever returned a
    # mapping-shaped workload (e.g. a dict) instead of a Workload object,
    # the validation loop in run_non_interactive must raise ConfigError
    # (a clear, user-facing error) rather than AttributeError (an opaque
    # internal failure). Simulate that by monkeypatching the resolver.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_resolver(_config):
        # Deliberately wrong shape: a dict instead of a Workload.
        return [
            (
                "custom",
                "Custom",
                {
                    "requests_per_day": 1000,
                    "avg_input_tokens": 500,
                    "avg_output_tokens": 300,
                },
            )
        ]

    monkeypatch.setattr(m, "_resolve_workload_scenarios", fake_resolver)

    with pytest.raises(m.ConfigError, match="not a Workload"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


# --------------------------------------------------------------------------
# run_non_interactive with multiple preset scenarios (per-scenario export)
# --------------------------------------------------------------------------


def test_run_non_interactive_currency_flag_converts_rows_and_export(
    tmp_path: Path, monkeypatch, capsys
):
    # A non-USD --currency should fetch an FX rate and convert every row
    # (local and hosted alike) before rendering/exporting, mirroring the
    # interactive GBP path. The rate is mocked so the test doesn't depend
    # on a live FX provider.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Record the arguments: the direction of the lookup is the whole
    # correctness question here, and a stub that ignores them cannot
    # distinguish a right answer from an inverted one.
    fx_calls = []

    def fake_fx(from_currency, to_currency, timeout=5.0):
        fx_calls.append((from_currency, to_currency))
        return 1.25  # 1 GBP = 1.25 USD

    monkeypatch.setattr(m, "fetch_fx_rate", fake_fx)

    export_path = tmp_path / "out.json"
    exit_code = m.run_non_interactive(
        config_path,
        export_fmt="json",
        export_path=export_path,
        currency="GBP",
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "£" in out

    # convert_rows_currency divides by USD-per-target, so the lookup has to
    # be GBP->USD, not USD->GBP. Asking the other way round returns ~0.79
    # and scales costs *up*.
    assert fx_calls == [("GBP", "USD")]

    # The Claude Opus 5 row is $300/month in USD (1000 req/day, 500 in +
    # 300 out tokens, at $5/$25 per million). At 1 GBP = 1.25 USD that is
    # £240 — and it must be smaller than the dollar figure, because a
    # pound buys more than a dollar.
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert all("monthly_cost_gbp" in row for row in data)
    assert all("monthly_cost_usd" not in row for row in data)
    hosted = next(row for row in data if "Opus" in row["option"])
    assert hosted["monthly_cost_gbp"] == pytest.approx(240.0)
    assert hosted["monthly_cost_gbp"] < 300.0
    assert "£240.00" in out


def test_currency_flag_fetches_the_rate_once_for_a_multi_scenario_run(
    tmp_path: Path, monkeypatch, capsys
):
    # The FX lookup is a network call. Resolving it before the scenario
    # loop is the stated design; three presets must not mean three
    # requests.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload_presets": ["casual", "coding_agent", "team_tool"],
                "local": {
                    "mode": "existing",
                    "tokens_per_sec": 40,
                    "power_watts": 450,
                    "electricity_rate_per_kwh": 0.15,
                },
                "pricing_file": str(pricing_path),
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_fx(from_currency, to_currency, timeout=5.0):
        calls.append((from_currency, to_currency))
        return 1.25

    monkeypatch.setattr(m, "fetch_fx_rate", fake_fx)
    assert (
        m.run_non_interactive(
            config_path, export_fmt=None, export_path=None, currency="GBP"
        )
        == 0
    )
    capsys.readouterr()
    assert len(calls) == 1


@pytest.mark.parametrize("bad_rate", [None, 0, -1.0])
def test_currency_flag_falls_back_on_a_non_positive_rate(
    tmp_path: Path, monkeypatch, capsys, bad_rate
):
    # A zero or negative rate is as unusable as no rate at all, and
    # convert_rows_currency would raise on it. Falling back keeps the run
    # alive with correct numbers in the wrong unit.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workload": {
                    "requests_per_day": 1000,
                    "avg_input_tokens": 500,
                    "avg_output_tokens": 300,
                },
                "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
                "pricing_file": str(pricing_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda f, t, timeout=5.0: bad_rate)
    assert (
        m.run_non_interactive(
            config_path, export_fmt=None, export_path=None, currency="GBP"
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "falling back to USD" in captured.err
    assert "$300.00" in captured.out
    assert "£" not in captured.out


def test_run_non_interactive_currency_flag_falls_back_to_usd_on_fx_failure(
    tmp_path: Path, monkeypatch, capsys
):
    # If the FX lookup fails, the run should still succeed — just in USD —
    # with a warning on stderr, rather than raising or producing nonsense.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(m, "fetch_fx_rate", lambda f, t, timeout=5.0: None)

    export_path = tmp_path / "out.json"
    exit_code = m.run_non_interactive(
        config_path,
        export_fmt="json",
        export_path=export_path,
        currency="GBP",
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "falling back to USD" in captured.err
    assert "$" in captured.out

    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert all("monthly_cost_usd" in row for row in data)


def test_run_non_interactive_default_currency_is_usd(
    tmp_path: Path, monkeypatch, capsys
):
    # No --currency (or USD) must not trigger an FX lookup at all.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fetch_fx_rate should not be called for USD")

    monkeypatch.setattr(m, "fetch_fx_rate", fail_if_called)

    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    assert "$" in capsys.readouterr().out


def test_main_non_interactive_currency_flag_is_forwarded(
    tmp_path: Path, monkeypatch, capsys
):
    # The CLI flag must actually reach run_non_interactive — a common
    # regression when a new argument is added but not threaded through.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": 1000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(m, "fetch_fx_rate", lambda f, t, timeout=5.0: 0.8)

    exit_code = m.main(
        [
            "--non-interactive",
            "--config",
            str(config_path),
            "--currency",
            "GBP",
        ]
    )
    assert exit_code == 0
    assert "£" in capsys.readouterr().out


# --------------------------------------------------------------------------
# --use-defaults fast-path re-runs GPU detection / throughput benchmark
# --------------------------------------------------------------------------


_SAVED_TARGET = {
    "backend": "ollama",
    "base_url": "http://localhost:11434",
    "model": "llama3",
}


def _no_stdin(monkeypatch):
    """Fail the test if anything reads stdin.

    --use-defaults is the fast path. A refresh that asks which backend,
    which URL and which model is four questions the flag exists to avoid,
    so "does not prompt" is part of the contract, not an incidental.
    """
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": pytest.fail(f"--use-defaults prompted: {prompt!r}"),
    )


def test_refresh_measurements_for_defaults_uses_fresh_benchmark(monkeypatch):
    # The saved tokens_per_sec is stale; the fresh benchmark must win.
    _no_stdin(monkeypatch)
    settings = {"tokens_per_sec": 5.0, "benchmark_target": dict(_SAVED_TARGET)}
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: _stub_gpu_info())
    monkeypatch.setattr(m, "average_gpu_power_w", lambda *a, **k: 40.0)
    monkeypatch.setattr(m, "benchmark_ollama", lambda base_url, model: 123.0)
    monkeypatch.setattr(
        m, "measure_gpu_power_during", lambda func, **k: (func(), 380.0)
    )

    tokens_per_sec, gpu_info, measured_load_power_w = (
        m._refresh_measurements_for_defaults(settings)
    )
    assert tokens_per_sec == pytest.approx(123.0)
    assert gpu_info is not None
    assert measured_load_power_w == pytest.approx(380.0)


def test_refresh_measurements_reuses_the_saved_endpoint(monkeypatch):
    # The endpoint comes from the saved settings, not from the user. If it
    # did not, the fast path would have to ask for it.
    _no_stdin(monkeypatch)
    seen = {}
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(
        m,
        "benchmark_openai_compatible",
        lambda base_url, model: seen.update(url=base_url, model=model) or 50.0,
    )
    monkeypatch.setattr(
        m, "benchmark_ollama", lambda *a, **k: pytest.fail("wrong backend")
    )
    monkeypatch.setattr(m, "measure_gpu_power_during", lambda func, **k: (func(), None))

    m._refresh_measurements_for_defaults(
        {
            "tokens_per_sec": 5.0,
            "benchmark_target": {
                "backend": "openai",
                "base_url": "http://gpu-box:8000/v1",
                "model": "qwen",
            },
        }
    )
    assert seen == {"url": "http://gpu-box:8000/v1", "model": "qwen"}


def test_refresh_measurements_keeps_a_hand_entered_throughput(monkeypatch, capsys):
    # No saved endpoint means the previous run never benchmarked one — the
    # user declined, or typed the figure in. A hand-entered number is not a
    # stale measurement, so re-measuring is neither possible nor wanted.
    _no_stdin(monkeypatch)
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(
        m, "benchmark_ollama", lambda *a, **k: pytest.fail("nothing to benchmark")
    )
    tokens_per_sec, gpu_info, power = m._refresh_measurements_for_defaults(
        {"tokens_per_sec": 7.5}
    )
    assert tokens_per_sec == pytest.approx(7.5)
    assert gpu_info is None and power is None
    assert "No saved benchmark endpoint" in capsys.readouterr().out


def test_refresh_measurements_falls_back_to_saved_on_benchmark_failure(
    monkeypatch, capsys
):
    # When the endpoint is saved but unreachable, the saved value is used
    # and the script says so rather than silently replaying it.
    _no_stdin(monkeypatch)
    settings = {"tokens_per_sec": 7.5, "benchmark_target": dict(_SAVED_TARGET)}
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(
        m,
        "benchmark_ollama",
        lambda base_url, model: (_ for _ in ()).throw(OSError("connection refused")),
    )
    monkeypatch.setattr(m, "measure_gpu_power_during", lambda func, **k: (func(), None))

    tokens_per_sec, gpu_info, measured_load_power_w = (
        m._refresh_measurements_for_defaults(settings)
    )
    assert tokens_per_sec == pytest.approx(7.5)
    assert gpu_info is None
    assert measured_load_power_w is None
    out = capsys.readouterr().out
    assert "connection refused" in out
    assert "using saved throughput" in out.lower()


def test_run_interactive_use_defaults_reruns_benchmark(
    tmp_path: Path, monkeypatch, capsys
):
    # End-to-end: --use-defaults must not simply replay the saved
    # tokens_per_sec — it must re-run the benchmark and use the fresh
    # value, without asking anything.
    saved = {
        "mode": "rent",
        "tokens_per_sec": 5.0,
        "hourly_rate": 2.5,
        "benchmark_target": dict(_SAVED_TARGET),
        "workload_preset": "casual",
        "selected_models": None,
        "last_run_at": "2020-01-01T00:00:00+00:00",
    }
    last_run_path = tmp_path / ".last_run.json"
    last_run_path.write_text(json.dumps(saved), encoding="utf-8")
    # load_last_run's path default is bound at definition, so rebinding
    # m.DEFAULT_LAST_RUN_PATH does nothing — the original test did that and
    # therefore never entered the fast path at all, falling through to the
    # full interactive flow and passing on its output instead.
    monkeypatch.setattr(
        m,
        "load_last_run",
        lambda *a, **k: json.loads(last_run_path.read_text(encoding="utf-8")),
    )
    _no_stdin(monkeypatch)
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(m, "benchmark_ollama", lambda base_url, model: 99.0)
    monkeypatch.setattr(m, "measure_gpu_power_during", lambda func, **k: (func(), None))
    monkeypatch.setattr(m, "prompt_yes_no", lambda prompt, default=True: False)

    exit_code = m.run_interactive(use_defaults=True)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Measured throughput: 99.0 tokens/sec" in out
    # The stale saved figure must not be what got used.
    assert "5.0 tokens/sec" not in out


@pytest.mark.parametrize(
    "mode_answer, expected_mode, extra_answers",
    [
        ("rent", "rent", {"Rented GPU hourly rate": "2.5"}),
        # The prompt offers existing/buying/rent; "buying" is stored as
        # mode "own". Answering "own" loops the prompt forever.
        (
            "buying",
            "own",
            {
                "Hardware cost (USD)": "3600",
                "Expected hardware lifetime": "3",
                "Power draw under load": "450",
            },
        ),
        ("existing", "existing", {}),
    ],
)
def test_interactive_setup_records_the_benchmark_target(
    monkeypatch, capsys, mode_answer, expected_mode, extra_answers
):
    # --use-defaults can only re-benchmark an endpoint the previous run
    # wrote down, and nothing persisted backend/base_url/model before.
    # Every hardware mode builds its own settings dict, so all three have
    # to carry it — dropping it from one would otherwise go unnoticed.
    monkeypatch.setattr(m, "detect_nvidia_gpu", lambda runner=None: None)
    monkeypatch.setattr(m, "average_gpu_power_w", lambda *a, **k: None)
    monkeypatch.setattr(m, "discover_local_models", lambda backend, url: ["llama3"])
    monkeypatch.setattr(m, "benchmark_ollama", lambda base_url, model: 42.0)
    monkeypatch.setattr(m, "measure_gpu_power_during", lambda func, **k: (func(), None))
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: pytest.fail("network call"))
    answers = {
        "Skip benchmark": "n",
        "auto-detect an NVIDIA GPU": "n",
        "benchmark a running local model endpoint": "y",
        "Backend": "ollama",
        "Base URL": "http://gpu-box:11434",
        "Model name as served locally": "llama3",
        "Hardware mode": mode_answer,
        "Look up your current unit rate live": "n",
        "Do you pay for electricity in GBP": "n",
        "Electricity rate": "0.15",
        "Use this electricity rate?": "y",
        "Extra power draw while generating": "",
        "Total system power draw while running": "",
        **extra_answers,
    }

    def fake_input(prompt: str = "") -> str:
        for fragment, answer in answers.items():
            if fragment in prompt:
                return answer
        pytest.fail(f"unscripted prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    *_, settings = m.interactive_local_setup()
    assert settings["mode"] == expected_mode
    assert settings["benchmark_target"] == {
        "backend": "ollama",
        "base_url": "http://gpu-box:11434",
        "model": "llama3",
    }


def test_run_non_interactive_accepts_valid_multi_scenario_config(
    tmp_path: Path, capsys
):
    # Workload validation applies to every resolved scenario, not just the
    # primary one, so a config naming several presets has to survive it
    # intact — each scenario priced and printed, no ConfigError.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload_presets": ["casual", "coding_agent"],
        "local": {
            "mode": "existing",
            "tokens_per_sec": 40,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Must not raise ConfigError.
    exit_code = m.run_non_interactive(config_path, export_fmt=None, export_path=None)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Casual personal use" in out
    assert "Autonomous coding agent" in out


def test_run_non_interactive_multiple_presets_prints_one_combined_table_and_exports_one_file(
    tmp_path: Path, capsys
):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload_presets": ["casual", "coding_agent"],
        "local": {
            "mode": "existing",
            "tokens_per_sec": 40,
            "power_watts": 450,
            "electricity_rate_per_kwh": 0.15,
        },
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    export_path = tmp_path / "out.json"
    exit_code = m.run_non_interactive(
        config_path, export_fmt="json", export_path=export_path
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Casual personal use" in out
    assert "Autonomous coding agent" in out
    assert export_path.exists()
    data = json.loads(export_path.read_text(encoding="utf-8"))
    scenarios_seen = {row["scenario"] for row in data}
    assert scenarios_seen == {"Casual personal use", "Autonomous coding agent"}


# --------------------------------------------------------------------------
# Full interactive end-to-end session (monkeypatched input())
# --------------------------------------------------------------------------


_PRICING_FIXTURE = {
    "as_of": "2026-01-01",
    "note": "test fixture",
    "providers": {
        "claude": {
            "models": {
                "opus-5": {
                    "display_name": "Claude Opus 5",
                    "input_per_million": 5.0,
                    "output_per_million": 25.0,
                }
            }
        }
    },
}


def _interactive_session(monkeypatch, answers: dict):
    """Drive run_interactive() by matching prompts rather than counting them.

    A positional list of answers breaks the moment a prompt is added, removed
    or reordered, and fails as an opaque StopIteration that says nothing about
    which prompt drifted. Matching on a substring of the prompt keeps these
    tests readable and pins each answer to the question it belongs to.

    Every prompt must be matched by exactly one fragment. Falling back to ""
    would be worse than the StopIteration it replaces: an empty string is
    usually accepted as the shown default, so a reworded prompt would let the
    test keep passing while silently no longer exercising the intended path.
    Prompts whose default is genuinely what we want are listed explicitly in
    `_ACCEPT_DEFAULT` rather than left to fall through.

    An answer may be a string, or a list of strings to return on successive
    matches (used to feed an invalid answer followed by a valid one).
    """
    pending = {
        fragment: list(value) if isinstance(value, list) else None
        for fragment, value in answers.items()
    }

    def fake_input(prompt: str = "") -> str:
        for fragment, value in answers.items():
            if fragment in prompt:
                queued = pending[fragment]
                if queued is None:
                    return value
                if not queued:
                    pytest.fail(f"ran out of scripted answers for prompt: {prompt!r}")
                return queued.pop(0)
        pytest.fail(
            f"unscripted prompt: {prompt!r}\n"
            "Add a fragment of it to the test's answers, or to _ACCEPT_DEFAULT."
        )

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(m, "load_pricing", lambda *a, **k: dict(_PRICING_FIXTURE))
    # Belt and braces: fail loudly rather than reach the network if a future
    # prompt change routes past the explicit answers below.
    monkeypatch.setattr(
        m, "fetch_octopus_agile_rate", lambda *a, **k: pytest.fail("network call")
    )
    monkeypatch.setattr(m, "fetch_fx_rate", lambda *a, **k: pytest.fail("network call"))


# Prompts whose shown default is what these tests want. Listed explicitly so
# that an unrecognised prompt is an error rather than a silent default.
_ACCEPT_DEFAULT = {
    "Extra power draw while generating": "",
    "Total system power draw while running": "",
    "Export results to a file?": "",
    "Save these settings as defaults": "",
    # The confirmation added for issue #54. The trailing "?" keeps this
    # distinct from the GBP branch's "...rate and exchange rate?" prompt,
    # which the first-match responder would otherwise swallow.
    "Use this electricity rate?": "",
}


# Answers shared by every interactive test: skip all hardware probing and
# supply the electricity rate by hand so nothing touches the network.
_OFFLINE_ANSWERS = {
    "auto-detect your GPU": "n",
    "benchmark a running local model endpoint": "n",
    "Look up your current unit rate live": "n",
    # Declining the Octopus lookup still leaves the currency as GBP, which
    # triggers a live FX conversion. Paying in USD keeps the run offline.
    "Do you pay for electricity in GBP": "n",
    "Electricity rate": "0.15",
    # Pinned rather than defaulted so the local cost column is deterministic.
    "Measured or estimated tokens/sec": "40",
    **_ACCEPT_DEFAULT,
}


def test_run_interactive_end_to_end_preset(monkeypatch, capsys):
    # Pick preset #1 ("casual") and the default "existing" hardware mode, then
    # compare against every hosted model.
    _interactive_session(
        monkeypatch,
        {
            "Scenario [1-7]": "1",
            "Skip benchmark": "y",
            "Hardware mode": "existing",
            "Compare against all of the above?": "y",
            **_OFFLINE_ANSWERS,
        },
    )

    exit_code = m.run_interactive()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Casual personal use" in out
    assert "Claude Opus 5" in out


def test_run_interactive_end_to_end_custom_workload(monkeypatch, capsys):
    # The "custom numbers" branch is the last menu entry, after the presets
    # and the "compare all" option.
    custom_option = str(len(m.WORKLOAD_PRESETS) + 2)
    _interactive_session(
        monkeypatch,
        {
            "Scenario [1-7]": custom_option,
            "Expected requests/day": "1000",
            "Average input tokens/request": "500",
            "Average output tokens/request": "300",
            "Skip benchmark": "y",
            "Hardware mode": "buying",
            # "buying" swaps the two power-basis prompts for an up-front
            # hardware cost, a lifetime to amortise it over, and a single
            # under-load power figure.
            "Hardware cost (USD)": "1600",
            "Expected hardware lifetime (years)": "3",
            "Power draw under load (W)": "450",
            "Compare against all of the above?": "y",
            **_OFFLINE_ANSWERS,
        },
    )

    exit_code = m.run_interactive()

    out = capsys.readouterr().out
    assert exit_code == 0
    # "buying" amortises the hardware cost, so the local row is labelled
    # differently from the "existing" (electricity-only) row above.
    assert "Local (buy hardware)" in out
    assert "40.0 tok/s" in out
    assert "Claude Opus 5" in out


def test_run_interactive_end_to_end_compare_all_presets(monkeypatch, capsys):
    # "Compare all scenarios" sits directly after the presets and produces one
    # row group per preset.
    all_option = str(len(m.WORKLOAD_PRESETS) + 1)
    _interactive_session(
        monkeypatch,
        {
            "Scenario [1-7]": all_option,
            "Skip benchmark": "y",
            "Hardware mode": "existing",
            "Compare against all of the above?": "y",
            **_OFFLINE_ANSWERS,
        },
    )

    exit_code = m.run_interactive()

    out = capsys.readouterr().out
    assert exit_code == 0
    for preset in m.WORKLOAD_PRESETS:
        assert preset.label in out
    assert "Claude Opus 5" in out


def test_run_interactive_reprompts_on_invalid_scenario(monkeypatch, capsys):
    # Out-of-range and non-numeric menu answers must both be rejected with a
    # readable message and re-prompted, not crash or silently pick a default.
    _interactive_session(
        monkeypatch,
        {
            "Scenario [1-7]": ["99", "abc", "1"],
            "Skip benchmark": "y",
            "Hardware mode": "existing",
            "Compare against all of the above?": "y",
            **_OFFLINE_ANSWERS,
        },
    )

    exit_code = m.run_interactive()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.count("Please enter a number from 1 to 7.") == 2
    # Having recovered, the run still completes against the chosen preset.
    assert "Casual personal use" in out


def test_run_interactive_falls_back_to_manual_throughput(monkeypatch, capsys):
    # With the benchmark not skipped but both GPU auto-detect and the local
    # endpoint declined, throughput has to come from the manual prompt. This is
    # the fallback path that has no hardware to measure from.
    _interactive_session(
        monkeypatch,
        {
            "Scenario [1-7]": "1",
            "Skip benchmark": "n",
            "Hardware mode": "existing",
            "Compare against all of the above?": "y",
            **_OFFLINE_ANSWERS,
        },
    )

    exit_code = m.run_interactive()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Claude Opus 5" in out
