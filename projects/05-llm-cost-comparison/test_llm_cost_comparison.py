"""Tests for llm_cost_comparison.py.

Covers the pure cost-calculation functions, pricing/config loading, GPU
detection parsing (with a mocked subprocess runner), and export helpers.
Interactive input() flows are intentionally not exercised here — the
interactive functions are thin wrappers over the tested pure functions.
"""

from __future__ import annotations

import json
import subprocess
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
    # exist in a month (720). The cost is still real — it's what running
    # flat-out, 24/7, all month would cost — but the notes must say plainly
    # that this only covers part of the workload rather than implying the
    # full requested volume was delivered for that price.
    w = m.Workload(requests_per_day=50000, avg_input_tokens=500, avg_output_tokens=300)
    row = m.build_local_row(
        w,
        tokens_per_sec=7.7,
        mode="existing",
        power_watts=100,
        electricity_rate_per_kwh=0.15,
    )
    assert "covers only ~" in row.notes
    assert "x this throughput" in row.notes
    assert row.feasible is False
    assert row.monthly_cost > 0


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
    m._validate_http_url("http://localhost:11434")
    m._validate_http_url("https://example.com")


def test_validate_http_url_rejects_other_schemes():
    with pytest.raises(ValueError):
        m._validate_http_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        m._validate_http_url("not-a-url")


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

    def fake_urlopen(url, timeout=None):
        if "standard-unit-rates" in url:
            return _FakeHTTPResponse(rates_body)
        return _FakeHTTPResponse(products_body)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    rate = m.fetch_octopus_agile_rate("C")
    assert rate == pytest.approx(0.2483)


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
# Non-interactive config validation
# --------------------------------------------------------------------------


def _write_pricing(path: Path) -> None:
    path.write_text(
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


@pytest.mark.parametrize("bad_value", [0, -5, "fast", True])
def test_run_non_interactive_rejects_nonpositive_tokens_per_sec(
    tmp_path: Path, bad_value
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
        "local": {"mode": "rent", "tokens_per_sec": bad_value, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="tokens_per_sec"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize("mode", ["own", "existing", "rent"])
def test_run_non_interactive_rejects_zero_tokens_per_sec_in_every_mode(
    tmp_path: Path, mode
):
    # The interactive prompt enforces a minimum on tokens/sec regardless of
    # which hardware mode was chosen; the non-interactive path must reject
    # the same bad value in every mode, not just the one exercised above.
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    local_cfg = {"mode": mode, "tokens_per_sec": 0}
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


def test_run_non_interactive_rejects_zero_total_workload_tokens(tmp_path: Path):
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

    with pytest.raises(m.ConfigError, match="zero total tokens"):
        m.run_non_interactive(config_path, export_fmt=None, export_path=None)


@pytest.mark.parametrize("bad_value", [-1, "many"])
def test_run_non_interactive_rejects_bad_workload_field(tmp_path: Path, bad_value):
    pricing_path = tmp_path / "pricing.json"
    _write_pricing(pricing_path)
    config_path = tmp_path / "config.json"
    config = {
        "workload": {
            "requests_per_day": bad_value,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        },
        "local": {"mode": "rent", "tokens_per_sec": 40, "hourly_rate": 2.5},
        "pricing_file": str(pricing_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(m.ConfigError, match="requests_per_day"):
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


# --------------------------------------------------------------------------
# run_non_interactive with multiple preset scenarios (per-scenario export)
# --------------------------------------------------------------------------


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
