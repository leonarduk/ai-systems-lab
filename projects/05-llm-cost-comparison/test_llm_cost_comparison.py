"""Tests for llm_cost_comparison.py.

Covers the pure cost-calculation functions, pricing/config loading, GPU
detection parsing (with a mocked subprocess runner), and export helpers.
Interactive input() flows are intentionally not exercised here — the
interactive functions are thin wrappers over the tested pure functions.
"""

from __future__ import annotations

import json
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

    expected = "non-negative" if field == "avg_output_tokens" else "positive"
    with pytest.raises(
        m.ConfigError, match=rf"workload\.{field} must be a {expected} number"
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
    "auto-detect an NVIDIA GPU": "y",
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
        "auto-detect an NVIDIA GPU": "y",
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


# --------------------------------------------------------------------------
# run_non_interactive with multiple preset scenarios (per-scenario export)
# --------------------------------------------------------------------------


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
    "auto-detect an NVIDIA GPU": "n",
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
