#!/usr/bin/env python3
"""Interactive LLM cost comparison: local inference vs. hosted APIs.

Walks the user through a series of prompts to estimate the effective cost of
running an LLM locally (owned hardware or rented cloud GPU) and compares it
against hosted providers (DeepSeek, Claude) using pricing stored in
``pricing.json``.

Design goals:
  * Zero required third-party dependencies (stdlib only) so it runs on a
    plain Windows Python install with no ``pip install`` step.
  * All cost math lives in small, pure, unit-testable functions. The
    interactive layer is a thin wrapper around them so the logic can be
    exercised in tests without mocking ``input()``.
  * Best-effort Windows/NVIDIA GPU detection via ``nvidia-smi`` and an
    optional local-endpoint throughput benchmark (Ollama or an
    OpenAI-compatible ``/v1/chat/completions`` server). Both degrade
    gracefully to manual input if unavailable.

Usage:
    python llm_cost_comparison.py                # interactive
    python llm_cost_comparison.py --non-interactive --config example.json
    python llm_cost_comparison.py --export json --export-path out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Callable, Optional

# Resolved from installed package metadata so pyproject.toml is the single
# source of truth for the version. Falls back to an obviously-not-a-release
# sentinel when running from a source checkout where the package isn't
# installed (e.g. `python llm_cost_comparison.py --version`).
try:
    VERSION = _pkg_version("llm-cost-comparison")
except PackageNotFoundError:
    VERSION = "0.0.0+unknown"

DEFAULT_PRICING_PATH = Path(__file__).parent / "pricing.json"
DEFAULT_GPU_DEFAULTS_PATH = Path(__file__).parent / "gpu_power_defaults.json"
DEFAULT_LAST_RUN_PATH = Path(__file__).parent / ".last_run.json"
DAYS_PER_MONTH = 30
HOURS_PER_MONTH = DAYS_PER_MONTH * 24


class ConfigError(ValueError):
    """Raised for a malformed --non-interactive config, or an unknown preset key.

    Deliberately distinct from a bare ``KeyError``/``TypeError`` traceback:
    this is user-facing config, so a missing or misspelled field should say
    exactly what's missing rather than dumping a Python stack trace.
    """


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def load_pricing(
    path: Path = DEFAULT_PRICING_PATH, *, try_refresh: bool = False
) -> dict:
    """Load hosted-provider pricing from a JSON config file.

    By default this only reads the file and never mutates it. Passing
    ``try_refresh=True`` performs the same best-effort refresh used by the
    explicit ``--update-pricing`` command, and only for the shipped default
    pricing path so custom/user-edited config files are never overwritten
    implicitly.
    """
    if try_refresh and path == DEFAULT_PRICING_PATH:
        fetch_deepseek_pricing(path)
        fetch_bedrock_pricing(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"pricing file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"pricing file {path} is not valid JSON: {exc}") from exc


def iter_models(pricing: dict):
    """Yield (provider_key, model_key, model_info) for every priced model."""
    for provider_key, provider in pricing.get("providers", {}).items():
        for model_key, model_info in provider.get("models", {}).items():
            yield provider_key, model_key, model_info


DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"


def fetch_deepseek_pricing(
    path: Path = DEFAULT_PRICING_PATH, timeout: float = 10.0
) -> bool:
    """Fetch DeepSeek pricing from the official API docs and update ``path``.

    Returns ``True`` if the file was updated, ``False`` if the fetch failed
    or the prices were unchanged.  Only the DeepSeek section is touched;
    other providers (Claude, etc.) are preserved as-is.
    """
    import re

    try:
        req = urllib.request.Request(
            DEEPSEEK_PRICING_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return False

    # Extract the pricing rows from the HTML.  The page renders a table with
    # Flash and Pro prices side-by-side in the same row, so the Pro patterns
    # skip the first dollar amount (Flash) and capture the second (Pro).
    flash_cache_hit = _extract_price(
        text, r"deepseek-v4-flash.*?cache hit.*?\$([\d.]+)"
    )
    flash_cache_miss = _extract_price(
        text, r"deepseek-v4-flash.*?cache miss.*?\$([\d.]+)"
    )
    flash_output = _extract_price(
        text, r"deepseek-v4-flash.*?output tokens.*?\$([\d.]+)"
    )
    pro_cache_hit = _extract_price(
        text, r"deepseek-v4-pro.*?cache hit.*?\$[\d.]+\s*\$([\d.]+)"
    )
    pro_cache_miss = _extract_price(
        text, r"deepseek-v4-pro.*?cache miss.*?\$[\d.]+\s*\$([\d.]+)"
    )
    pro_output = _extract_price(
        text, r"deepseek-v4-pro.*?output tokens.*?\$[\d.]+\s*\$([\d.]+)"
    )

    if None in (flash_cache_miss, flash_output, pro_cache_miss, pro_output):
        return False

    # Read existing file directly (not via load_pricing, to avoid recursion).
    try:
        with open(path, "r", encoding="utf-8") as f:
            pricing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pricing = {}
    deepseek_models = {
        "deepseek-v4-flash-cache-miss": {
            "display_name": "DeepSeek Flash",
            "input_per_million": flash_cache_miss,
            "output_per_million": flash_output,
        },
        "deepseek-v4-pro-cache-miss": {
            "display_name": "DeepSeek Pro",
            "input_per_million": pro_cache_miss,
            "output_per_million": pro_output,
        },
    }
    if flash_cache_hit is not None:
        deepseek_models["deepseek-v4-flash-cache-hit"] = {
            "display_name": "DeepSeek Flash (cached)",
            "input_per_million": flash_cache_hit,
            "output_per_million": flash_output,
        }
    if pro_cache_hit is not None:
        deepseek_models["deepseek-v4-pro-cache-hit"] = {
            "display_name": "DeepSeek Pro (cached)",
            "input_per_million": pro_cache_hit,
            "output_per_million": pro_output,
        }

    pricing.setdefault("providers", {})["deepseek"] = {
        "display_name": "DeepSeek",
        "models": deepseek_models,
    }
    pricing["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pricing["source_deepseek"] = DEEPSEEK_PRICING_URL

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pricing, f, indent=2)
    return True


def _extract_price(text: str, pattern: str) -> Optional[float]:
    """Try a regex; return the first captured float or None."""
    import re

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


BEDROCK_PRICING_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
    "AmazonBedrock/current/region_index.json"
)


def fetch_bedrock_pricing(
    path: Path = DEFAULT_PRICING_PATH, timeout: float = 10.0
) -> bool:
    """Fetch Bedrock on-demand per-token prices from the AWS Price List API.

    Returns ``True`` if ``path`` was updated, ``False`` on failure.
    Existing Bedrock entries are replaced; other providers are untouched.
    """
    try:
        # 1. Get the current version URL for us-east-1
        with urllib.request.urlopen(BEDROCK_PRICING_URL, timeout=timeout) as resp:
            region_index = json.loads(resp.read())
        version_url = region_index["regions"]["us-east-1"]["currentVersionUrl"]
        full_url = f"https://pricing.us-east-1.amazonaws.com{version_url}"

        # 2. Fetch the per-SKU pricing data
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return False

    models = {}
    for sku, product in data.get("products", {}).items():
        attrs = product.get("attributes", {})
        model = attrs.get("model", "").strip()
        if not model:
            continue
        # Only standard on-demand pricing (skip flex / batch / priority tiers).
        tier = (attrs.get("service_tier") or "").lower()
        if tier and tier not in ("standard", "on-demand"):
            continue
        inference = attrs.get("inferenceType", "").lower()
        terms = data.get("terms", {}).get("OnDemand", {}).get(sku, {})
        for term in terms.values():
            for rate in term.get("priceDimensions", {}).values():
                try:
                    price = float(rate.get("pricePerUnit", {}).get("USD", 0))
                except (ValueError, TypeError):
                    continue
                if price <= 0:
                    continue
                # Bedrock pricePerUnit is per 1 000 tokens
                per_million = price * 1000
                models.setdefault(model, {})
                if "input" in inference:
                    models[model]["input_per_million"] = round(per_million, 6)
                elif "output" in inference:
                    models[model]["output_per_million"] = round(per_million, 6)

    if not models:
        return False

    pricing = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            pricing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Merge into existing aws models — only overwrite entries the API found.
    existing = (
        pricing.setdefault("providers", {})
        .setdefault("aws", {})
        .setdefault("models", {})
    )
    for model_name, prices in models.items():
        key = model_name.lower().replace(" ", "-").replace(".", "")
        if "input_per_million" not in prices or "output_per_million" not in prices:
            continue
        existing[key] = {
            "display_name": f"Bedrock {model_name}",
            "input_per_million": prices["input_per_million"],
            "output_per_million": prices["output_per_million"],
        }

    pricing["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pricing["source_bedrock"] = BEDROCK_PRICING_URL

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pricing, f, indent=2)
    return True


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


@dataclass
class Workload:
    """Describes the traffic the comparison should be sized against."""

    requests_per_day: float
    avg_input_tokens: float
    avg_output_tokens: float

    @property
    def monthly_input_tokens(self) -> float:
        return self.requests_per_day * self.avg_input_tokens * DAYS_PER_MONTH

    @property
    def monthly_output_tokens(self) -> float:
        return self.requests_per_day * self.avg_output_tokens * DAYS_PER_MONTH

    @property
    def monthly_total_tokens(self) -> float:
        return self.monthly_input_tokens + self.monthly_output_tokens


@dataclass(frozen=True)
class WorkloadPreset:
    """A named, plain-language traffic scenario.

    Guessing "requests per day" and "average input tokens" cold is a bad
    starting point for someone who has never measured their own usage.
    Presets give a menu of scenarios described in terms people actually
    reason in ("a live app serving many users"), with the underlying numbers
    filled in — while still leaving room for a fully custom entry.
    """

    key: str
    label: str
    description: str
    requests_per_day: float
    avg_input_tokens: float
    avg_output_tokens: float

    def to_workload(self) -> Workload:
        return Workload(
            self.requests_per_day, self.avg_input_tokens, self.avg_output_tokens
        )


WORKLOAD_PRESETS: tuple = (
    WorkloadPreset(
        key="casual",
        label="Casual personal use",
        description=(
            "A handful of questions a day, similar to using it instead of a search engine."
        ),
        requests_per_day=20,
        avg_input_tokens=300,
        avg_output_tokens=250,
    ),
    WorkloadPreset(
        key="daily_assistant",
        label="Daily work assistant",
        description=(
            "Used on and off throughout the workday for drafting, research, and quick "
            "coding help."
        ),
        requests_per_day=150,
        avg_input_tokens=800,
        avg_output_tokens=500,
    ),
    WorkloadPreset(
        key="coding_agent",
        label="Autonomous coding agent",
        description=(
            "An agent that reads files and runs commands on its own (like an AI coding "
            "assistant). Each turn re-sends a lot of file/context content, so input "
            "tokens are large relative to output."
        ),
        requests_per_day=500,
        avg_input_tokens=4000,
        avg_output_tokens=800,
    ),
    WorkloadPreset(
        key="team_tool",
        label="Small team internal tool",
        description="A shared assistant used by a small team (roughly 5-20 people) all day.",
        requests_per_day=2000,
        avg_input_tokens=600,
        avg_output_tokens=400,
    ),
    WorkloadPreset(
        key="production_app",
        label="Production customer-facing app",
        description="A live app serving many users' requests around the clock.",
        requests_per_day=50000,
        avg_input_tokens=500,
        avg_output_tokens=300,
    ),
)


def get_preset(key: str) -> WorkloadPreset:
    """Look up a preset by key, raising ``ConfigError`` with valid keys listed."""
    for preset in WORKLOAD_PRESETS:
        if preset.key == key:
            return preset
    valid = ", ".join(p.key for p in WORKLOAD_PRESETS)
    raise ConfigError(f"unknown workload preset {key!r}; valid presets: {valid}")


# --------------------------------------------------------------------------
# Core cost math (pure functions — unit test these directly)
# --------------------------------------------------------------------------


def hosted_monthly_cost(
    workload: Workload, input_per_million: float, output_per_million: float
) -> float:
    """Monthly cost for a hosted API at the given per-million-token rates."""
    return (
        workload.monthly_input_tokens / 1_000_000 * input_per_million
        + workload.monthly_output_tokens / 1_000_000 * output_per_million
    )


def hours_needed_for_workload(
    total_monthly_tokens: float, tokens_per_sec: float
) -> float:
    """Compute-hours per month required to generate the workload's tokens.

    Assumes the local/rented GPU only needs to run while actively producing
    tokens for this workload (i.e. it can idle/power down otherwise). This is
    the right basis for a rented cloud GPU, and the variable-cost basis for
    owned hardware (see ``local_monthly_cost_owned``).
    """
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be > 0")
    tokens_per_hour = tokens_per_sec * 3600
    return total_monthly_tokens / tokens_per_hour


def scale_workload_to_local_capacity(
    workload: Workload, tokens_per_sec: float
) -> tuple:
    """Scale a workload down to what local throughput can produce in a real month.

    Pricing hosted providers against a workload's full requested volume
    while the local option can only ever produce a fraction of it (running
    flat-out, 24/7, all month — see ``build_local_row``) isn't an
    apples-to-apples comparison: it prices cloud for more work than the
    local machine could ever do, so the two monthly-cost columns aren't
    answering the same question. Scaling every option — local and hosted
    alike — to the same, real, achievable token volume makes every option's
    monthly figure directly comparable.

    Returns ``(effective_workload, feasible, coverage_pct)``. If
    ``tokens_per_sec`` can produce the workload's full monthly token total
    within ``HOURS_PER_MONTH``, returns the workload unchanged with
    ``feasible=True`` and ``coverage_pct=100.0``. Otherwise scales
    ``requests_per_day`` down (keeping the same average input/output token
    ratio per request, so it's still "the same kind of workload", just
    fewer requests/day) to what the hardware could actually produce, and
    returns ``feasible=False`` with the resulting coverage percentage.
    """
    hours_needed = hours_needed_for_workload(
        workload.monthly_total_tokens, tokens_per_sec
    )
    if hours_needed <= HOURS_PER_MONTH:
        return workload, True, 100.0
    coverage = HOURS_PER_MONTH / hours_needed
    scaled = Workload(
        requests_per_day=workload.requests_per_day * coverage,
        avg_input_tokens=workload.avg_input_tokens,
        avg_output_tokens=workload.avg_output_tokens,
    )
    return scaled, False, coverage * 100.0


# Default input:output token ratio assumed when sizing the "local at maximum
# capacity" comparison — matches the customary defaults used elsewhere for a
# generic request (e.g. the old free-form workload entry). Only the ratio
# matters here, not the absolute numbers: the comparison is sized from local
# throughput, not from a request count.
DEFAULT_AVG_INPUT_TOKENS = 500.0
DEFAULT_AVG_OUTPUT_TOKENS = 300.0


def local_max_capacity_workload(
    tokens_per_sec: float,
    avg_input_tokens: float = DEFAULT_AVG_INPUT_TOKENS,
    avg_output_tokens: float = DEFAULT_AVG_OUTPUT_TOKENS,
) -> Workload:
    """Build the workload representing local hardware running flat-out, 24/7.

    There's no "requests/day" to guess here — the comparison is sized
    backwards from what the hardware can physically produce
    (``tokens_per_sec`` for a full real month, ``HOURS_PER_MONTH``), not
    from an assumed traffic level. An input:output token ratio is still
    needed only because hosted providers price those at different rates,
    so ``requests_per_day`` is solved for whatever value makes the
    resulting ``Workload`` add up to that same total token count.
    """
    max_tokens_per_month = tokens_per_sec * 3600 * HOURS_PER_MONTH
    tokens_per_request = avg_input_tokens + avg_output_tokens
    requests_per_day = max_tokens_per_month / (DAYS_PER_MONTH * tokens_per_request)
    return Workload(requests_per_day, avg_input_tokens, avg_output_tokens)


def local_monthly_cost_owned(
    hardware_cost: float,
    lifetime_years: float,
    power_watts: float,
    electricity_rate_per_kwh: float,
    hours_needed_per_month: float,
) -> float:
    """Monthly cost for owned local hardware.

    Splits into two components, which matters because they behave
    differently:
      * Hardware amortization is a *fixed* monthly cost — the card ages and
        eventually needs replacing whether it's busy or idle.
      * Electricity is a *variable* cost that only accrues while it's
        actually generating tokens for this workload.
    """
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be > 0")
    lifetime_months = lifetime_years * 12
    hardware_monthly = hardware_cost / lifetime_months
    power_kw = power_watts / 1000
    variable_monthly = power_kw * electricity_rate_per_kwh * hours_needed_per_month
    return hardware_monthly + variable_monthly


def local_monthly_cost_rented(
    hourly_rate: float, hours_needed_per_month: float
) -> float:
    """Monthly cost for a rented cloud GPU billed per hour of usage."""
    return hourly_rate * hours_needed_per_month


def local_monthly_cost_existing_hardware(
    power_watts: float, electricity_rate_per_kwh: float, hours_needed_per_month: float
) -> float:
    """Monthly cost when the PC/GPU is already owned for other reasons.

    No hardware amortization at all — unlike ``local_monthly_cost_owned``,
    this assumes the machine exists (and its cost is sunk) regardless of
    whether it's ever used for a local LLM. The only cost attributable to
    that use is the extra electricity consumed while it runs. ``power_watts``
    should reflect whichever basis applies:
      * If the machine is already on for other reasons, use the *extra*
        draw the GPU/CPU pull under load, above idle.
      * If it's only powered on to run the local LLM, use the *whole
        system's* draw while running (GPU plus CPU, RAM, storage, etc.) —
        the entire session's electricity is attributable to this use.
    """
    power_kw = power_watts / 1000
    return power_kw * electricity_rate_per_kwh * hours_needed_per_month


def local_monthly_cost_always_on(
    idle_watts: float,
    extra_watts: float,
    electricity_rate_per_kwh: float,
    hours_needed_per_month: float,
) -> float:
    """Monthly electricity cost when the PC runs as a 24/7 server.

    Unlike ``local_monthly_cost_existing_hardware`` (which assumes the
    machine powers down between sessions), this charges for *all* 720 hours:
      * idle draw for every hour the machine is on but not generating, plus
      * the extra GPU/CPU load for the hours spent actively generating.

    ``idle_watts`` is the system's baseline draw (GPU + CPU + RAM + fans,
    just sitting there).  ``extra_watts`` is the additional draw pulled
    *above idle* while generating tokens.  Both are needed because hosted
    serverless (e.g. AWS Lambda, DeepSeek) charges zero when idle — the
    always-on PC does not.
    """
    idle_kw = idle_watts / 1000
    extra_kw = extra_watts / 1000
    idle_cost = idle_kw * electricity_rate_per_kwh * HOURS_PER_MONTH
    generation_cost = extra_kw * electricity_rate_per_kwh * hours_needed_per_month
    return idle_cost + generation_cost


def cost_per_million_tokens(monthly_cost: float, monthly_total_tokens: float) -> float:
    """Blended $/1M tokens (input+output combined) for a given monthly cost.

    Blending input and output into one number makes local and hosted costs
    directly comparable even though hosted providers price them separately.
    """
    if monthly_total_tokens <= 0:
        raise ValueError("monthly_total_tokens must be > 0")
    return monthly_cost / monthly_total_tokens * 1_000_000


# --------------------------------------------------------------------------
# Comparison rows / rendering
# --------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    name: str
    monthly_cost: float
    cost_per_million_tokens: float
    notes: str = ""
    # False for a local option whose measured/assumed throughput can't
    # actually deliver the workload in real time (see build_local_row) — its
    # monthly_cost is real arithmetic but not a real-world option, so it must
    # never rank as "cheapest" against options that can actually do the job.
    feasible: bool = True


def build_local_row(
    workload: Workload,
    tokens_per_sec: float,
    mode: str,
    *,
    hardware_cost: float = 0.0,
    lifetime_years: float = 0.0,
    power_watts: float = 0.0,
    electricity_rate_per_kwh: float = 0.0,
    hourly_rate: float = 0.0,
    idle_watts: float = 0.0,
    extra_watts: float = 0.0,
    name: Optional[str] = None,
) -> ComparisonRow:
    """Build a local-option row, costed for what the hardware can actually do.

    ``hours_needed`` is how long this workload's tokens would take to
    generate at ``tokens_per_sec``. When that exceeds ``HOURS_PER_MONTH``
    (a real month only has 720 hours), the hardware physically cannot
    produce the whole workload in real time — but running it flat-out,
    24/7, all month, *is* a real, payable scenario (the machine simply
    isn't idle), so the cost is computed for ``effective_hours`` (capped at
    ``HOURS_PER_MONTH``) rather than for the uncapped ``hours_needed``.
    Capping avoids a straight-line extrapolation past hours that don't
    exist in a month, which would otherwise read as a real bill for
    something physically impossible (e.g. "costs more per month than the
    hardware itself would cost to buy"). ``cost_per_million_tokens`` is
    computed against the tokens actually produced in ``effective_hours``,
    not the workload's full requested total, so the $/1M rate stays the
    same real, hours-independent per-token figure either way — it's only
    the "does this option fully replace hosted for this workload" question
    that ``feasible`` still answers, and infeasible rows are still never
    ranked as "cheapest" (see ``render_table``).
    """
    hours_needed = hours_needed_for_workload(
        workload.monthly_total_tokens, tokens_per_sec
    )
    feasible = hours_needed <= HOURS_PER_MONTH
    effective_hours = min(hours_needed, HOURS_PER_MONTH)
    if mode == "own":
        monthly_cost = local_monthly_cost_owned(
            hardware_cost,
            lifetime_years,
            power_watts,
            electricity_rate_per_kwh,
            effective_hours,
        )
        name = name or "Local (buy hardware)"
    elif mode == "existing":
        monthly_cost = local_monthly_cost_existing_hardware(
            power_watts, electricity_rate_per_kwh, effective_hours
        )
        name = name or "Local (already-on PC)"
    elif mode == "rent":
        monthly_cost = local_monthly_cost_rented(hourly_rate, effective_hours)
        name = name or "Local (rented cloud GPU)"
    elif mode == "always_on":
        monthly_cost = local_monthly_cost_always_on(
            idle_watts, extra_watts, electricity_rate_per_kwh, effective_hours
        )
        name = name or "Local (24/7 server)"
    else:
        raise ValueError(f"Unknown local cost mode: {mode!r}")
    tokens_produced = tokens_per_sec * 3600 * effective_hours
    per_million = cost_per_million_tokens(monthly_cost, tokens_produced)
    if feasible:
        notes = (
            f"~{effective_hours:.1f} compute-hrs/month at {tokens_per_sec:.1f} tok/s"
        )
    else:
        coverage_pct = effective_hours / hours_needed * 100
        parallel_needed = hours_needed / HOURS_PER_MONTH
        notes = (
            f"running 24/7 all month at {tokens_per_sec:.1f} tok/s covers only "
            f"~{coverage_pct:.0f}% of this workload's tokens — would need "
            f"~{parallel_needed:.1f}x this throughput to fully replace hosted"
        )
    return ComparisonRow(name, monthly_cost, per_million, notes, feasible=feasible)


def _validate_pricing_model(model_info: dict, full_key: str) -> None:
    """Validate a single pricing model's per-million-token fields.

    Shared by ``load_pricing`` (via ``_validate_pricing``) and
    ``build_hosted_rows`` so both paths enforce identical semantics and
    raise the same ``ConfigError`` message. A field must be a real number
    (not a ``bool``, which is an ``int`` subclass), finite, and strictly
    positive — a zero or negative price would silently produce a
    nonsensical cost, and ``NaN``/``inf`` would poison every downstream
    figure.
    """
    import math

    for field in ("input_per_million", "output_per_million"):
        value = model_info.get(field)
        ok = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )
        if not ok:
            raise ConfigError(
                f"pricing model {full_key!r} is missing a numeric {field}"
            )


def build_hosted_rows(
    workload: Workload, pricing: dict, selected: Optional[set] = None
) -> list:
    """Build a ComparisonRow for each hosted model in ``pricing``.

    ``selected`` is an optional set of ``"provider/model"`` keys to restrict
    the comparison to; if None, every model in the pricing file is included.

    Pricing values are validated here (via ``_validate_pricing_model``)
    rather than assumed pre-validated, so a hand-constructed ``pricing``
    dict that bypassed ``load_pricing`` still fails loudly with a
    ``ConfigError`` instead of silently computing costs from a missing,
    zero, negative, non-numeric, ``NaN``, or ``inf`` price. The check is
    cheap (two field lookups per model) and reuses the same helper as
    ``load_pricing``, so error messages and semantics stay identical.
    """
    rows = []
    for provider_key, model_key, model_info in iter_models(pricing):
        full_key = f"{provider_key}/{model_key}"
        if selected is not None and full_key not in selected:
            continue
        _validate_pricing_model(model_info, full_key)
        monthly_cost = hosted_monthly_cost(
            workload, model_info["input_per_million"], model_info["output_per_million"]
        )
        per_million = cost_per_million_tokens(
            monthly_cost, workload.monthly_total_tokens
        )
        display = model_info.get("display_name", full_key)
        rows.append(ComparisonRow(display, monthly_cost, per_million))
    return rows


CURRENCY_SYMBOLS = {"USD": "$", "GBP": "£"}


def convert_rows_currency(rows: list, usd_per_gbp: float) -> list:
    """Convert every row's dollar figures to GBP for display.

    All cost math internally is in USD (hosted pricing is USD-denominated),
    so a GBP-preferring user's table is produced by converting the final
    figures once at display time rather than threading a currency through
    every cost function.
    """
    if usd_per_gbp <= 0:
        raise ValueError("usd_per_gbp must be > 0")
    return [
        ComparisonRow(
            r.name,
            r.monthly_cost / usd_per_gbp,
            r.cost_per_million_tokens / usd_per_gbp,
            r.notes,
            feasible=r.feasible,
        )
        for r in rows
    ]


def render_table(rows: list, currency: str = "USD") -> str:
    """Render comparison rows as a plain-text table, cheapest first.

    Infeasible rows (``feasible=False`` — a local option whose throughput
    can't produce this workload's full token volume within a real month)
    are sorted to the bottom and never picked as "cheapest": their cost is
    real (see ``build_local_row`` — it's the cost of running flat-out, 24/7,
    all month), but it only covers part of the workload, so ranking it
    against options that fully cover the workload would be misleading.
    """
    if not rows:
        return "(no rows to display)"
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    rows_sorted = sorted(rows, key=lambda r: (not r.feasible, r.monthly_cost))

    def cost_cell(r: ComparisonRow) -> str:
        return f"{symbol}{r.monthly_cost:,.2f}"

    name_w = max(len("Option"), max(len(r.name) for r in rows_sorted))
    cost_w = max(len("Monthly cost"), max(len(cost_cell(r)) for r in rows_sorted))
    per_m_w = max(
        len(f"{symbol}/1M tokens"),
        max(len(f"{symbol}{r.cost_per_million_tokens:,.2f}") for r in rows_sorted),
    )
    header = (
        f"{'Option':<{name_w}}  {'Monthly cost':>{cost_w}}  "
        f"{symbol + '/1M tokens':>{per_m_w}}  Notes"
    )
    lines = [header, "-" * len(header)]
    for r in rows_sorted:
        cost_s = cost_cell(r)
        per_m_s = f"{symbol}{r.cost_per_million_tokens:,.2f}"
        lines.append(
            f"{r.name:<{name_w}}  {cost_s:>{cost_w}}  {per_m_s:>{per_m_w}}  {r.notes}"
        )
    feasible_rows = [r for r in rows_sorted if r.feasible]
    if len(feasible_rows) > 1:
        cheapest = min(feasible_rows, key=lambda r: r.monthly_cost)
        most_expensive = max(feasible_rows, key=lambda r: r.monthly_cost)
        if most_expensive.monthly_cost > 0 and cheapest.monthly_cost > 0:
            multiple = most_expensive.monthly_cost / cheapest.monthly_cost
            lines.append("")
            lines.append(
                f"Cheapest: {cheapest.name} — most expensive option is {multiple:.1f}x its cost."
            )
    elif len(feasible_rows) == 1 and len(rows_sorted) > 1:
        lines.append("")
        lines.append(f"Cheapest: {feasible_rows[0].name} (only feasible option).")
    return "\n".join(lines)


def render_combined_table(scenario_rows: list, currency: str = "USD") -> str:
    """Render several scenarios as one matrix: one row per option, one
    column per scenario's monthly cost, plus a single "$/1M tokens" column.

    A separate mini-table per scenario (tried first) got unreadable past two
    or three scenarios — the same options repeated in full every time, just
    to show different monthly totals. A flat table with a repeated
    "Scenario" column (tried before that) was worse still: long scenario and
    option names forced every column to widen to fit the longest repeated
    value. Both approaches also obscured a fact worth surfacing: an option's
    $/1M-tokens rate does not vary with workload size (monthly cost scales
    with tokens, so the ratio is constant) — only the monthly total does.
    A single "$/1M tokens" column plus one cost column per scenario shows
    that directly instead of repeating the same rate in every section.

    Every cost here is real: a scenario whose local throughput can't keep up
    with the requested traffic should already have had its workload scaled
    down to what the hardware can actually produce running 24/7 (see
    ``scale_workload_to_local_capacity``) before its rows were built here,
    so local and hosted costs both reflect the same achievable token volume.

    ``scenario_rows`` is a list of ``(scenario_label, rows)`` pairs. Rows are
    matched across scenarios by name; row order (and $/1M tokens value) is
    taken from whichever scenario each name first appears in, then the
    matrix is sorted by that rate ascending.
    """
    if not scenario_rows:
        return "(no rows to display)"
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    row_order = []
    per_million_by_name = {}
    cost_by_name_and_scenario = {}
    for label, rows in scenario_rows:
        for r in rows:
            if r.name not in per_million_by_name:
                row_order.append(r.name)
                per_million_by_name[r.name] = r.cost_per_million_tokens
            cost_by_name_and_scenario[(r.name, label)] = r.monthly_cost

    row_order.sort(key=lambda name: per_million_by_name[name])
    scenario_labels = [label for label, _rows in scenario_rows]

    def fmt(value: float) -> str:
        return f"{symbol}{value:,.2f}"

    def cell_str(name: str, label: str) -> str:
        key = (name, label)
        if key not in cost_by_name_and_scenario:
            return "-"
        return fmt(cost_by_name_and_scenario[key])

    name_w = max(len("Option"), max(len(n) for n in row_order))
    per_m_header = symbol + "/1M tokens"
    per_m_w = max(
        len(per_m_header),
        max(len(fmt(per_million_by_name[n])) for n in row_order),
    )
    col_widths = {
        label: max([len(label)] + [len(cell_str(n, label)) for n in row_order])
        for label in scenario_labels
    }

    header = f"{'Option':<{name_w}}  {per_m_header:>{per_m_w}}"
    for label in scenario_labels:
        header += f"  {label:>{col_widths[label]}}"
    lines = [header, "-" * len(header)]

    for name in row_order:
        line = f"{name:<{name_w}}  {fmt(per_million_by_name[name]):>{per_m_w}}"
        for label in scenario_labels:
            line += f"  {cell_str(name, label):>{col_widths[label]}}"
        lines.append(line)

    return "\n".join(lines)


def export_csv(rows: list, path: Path, currency: str = "USD") -> None:
    suffix = currency.lower()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "option",
                f"monthly_cost_{suffix}",
                f"cost_per_million_tokens_{suffix}",
                "notes",
            ]
        )
        for r in sorted(rows, key=lambda r: (not r.feasible, r.monthly_cost)):
            writer.writerow(
                [
                    r.name,
                    f"{r.monthly_cost:.4f}",
                    f"{r.cost_per_million_tokens:.4f}",
                    r.notes,
                ]
            )


def export_json(rows: list, path: Path, currency: str = "USD") -> None:
    suffix = currency.lower()
    data = [
        {
            "option": r.name,
            f"monthly_cost_{suffix}": round(r.monthly_cost, 4),
            f"cost_per_million_tokens_{suffix}": round(r.cost_per_million_tokens, 4),
            "notes": r.notes,
        }
        for r in sorted(rows, key=lambda r: (not r.feasible, r.monthly_cost))
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_combined_csv(scenario_rows: list, path: Path, currency: str = "USD") -> None:
    """Export several scenarios' rows to one CSV with a leading scenario column."""
    suffix = currency.lower()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "option",
                f"monthly_cost_{suffix}",
                f"cost_per_million_tokens_{suffix}",
                "notes",
            ]
        )
        for label, rows in scenario_rows:
            for r in sorted(rows, key=lambda r: (not r.feasible, r.monthly_cost)):
                writer.writerow(
                    [
                        label,
                        r.name,
                        f"{r.monthly_cost:.4f}",
                        f"{r.cost_per_million_tokens:.4f}",
                        r.notes,
                    ]
                )


def export_combined_json(
    scenario_rows: list, path: Path, currency: str = "USD"
) -> None:
    """Export several scenarios' rows to one JSON file with a scenario field."""
    suffix = currency.lower()
    data = [
        {
            "scenario": label,
            "option": r.name,
            f"monthly_cost_{suffix}": round(r.monthly_cost, 4),
            f"cost_per_million_tokens_{suffix}": round(r.cost_per_million_tokens, 4),
            "notes": r.notes,
        }
        for label, rows in scenario_rows
        for r in sorted(rows, key=lambda r: (not r.feasible, r.monthly_cost))
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------
# Windows / NVIDIA GPU detection (best-effort, non-fatal)
# --------------------------------------------------------------------------


def _safe_float(value: str) -> Optional[float]:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None


def detect_nvidia_gpu(runner: Callable = subprocess.run) -> Optional[dict]:
    """Detect an NVIDIA GPU via ``nvidia-smi`` (works on Windows and Linux).

    Returns a dict with name/memory/power info, or None if ``nvidia-smi``
    isn't installed, isn't on PATH, or returns no usable data. Callers must
    treat None as "detection unavailable" and fall back to manual input.
    """
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first_line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 4:
        return None
    name, mem_total, power_draw, power_limit = parts[:4]
    return {
        "name": name,
        "memory_total_mib": _safe_float(mem_total),
        "power_draw_w": _safe_float(power_draw),
        "power_limit_w": _safe_float(power_limit),
    }


def average_gpu_power_w(
    runner: Callable = subprocess.run, samples: int = 3, interval: float = 0.3
) -> Optional[float]:
    """Average a handful of quick ``power.draw`` readings instead of trusting one.

    A single ``nvidia-smi`` query can be noisy — especially on a laptop GPU,
    which tends to report power in coarser steps and shift with whatever
    else the OS/driver is doing at that exact instant — so one instantaneous
    reading used as an "idle" baseline can vary a lot between runs of the
    same hardware. Averaging a few readings a fraction of a second apart
    smooths that out. Returns None if no reading was ever available.
    """
    readings = []
    for i in range(samples):
        info = detect_nvidia_gpu(runner)
        if info and info.get("power_draw_w") is not None:
            readings.append(info["power_draw_w"])
        if i < samples - 1:
            time.sleep(interval)
    return sum(readings) / len(readings) if readings else None


def measure_gpu_power_during(
    func: Callable, runner: Callable = subprocess.run, poll_interval: float = 0.5
) -> tuple:
    """Run ``func()`` while polling GPU power draw; return ``(result, avg_watts)``.

    ``avg_watts`` is the average of every ``power.draw`` reading seen while
    ``func`` was running, or None if no reading was available (no GPU, or
    ``func`` finished before the first poll). A single peak reading is one
    noisy driver sample away from being an outlier; averaging across the
    whole benchmark is a far more stable estimate of sustained draw under
    load — which is what a monthly electricity estimate actually needs,
    since real usage is however long generation actually runs, not one
    instantaneous spike.
    """
    readings = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            info = detect_nvidia_gpu(runner)
            if info and info.get("power_draw_w") is not None:
                readings.append(info["power_draw_w"])
            stop.wait(poll_interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        result = func()
    finally:
        stop.set()
        thread.join(timeout=poll_interval * 4)
    avg_watts = sum(readings) / len(readings) if readings else None
    return result, avg_watts


OCTOPUS_PRODUCTS_URL = "https://api.octopus.energy/v1/products/"


def fetch_octopus_agile_rate(
    region_letter: str = "C", timeout: float = 5.0
) -> Optional[float]:
    """Best-effort fetch of the current Octopus Agile unit rate (GBP/kWh).

    Octopus Agile product codes roll over every few months, so this looks
    the current one up rather than hardcoding it, then finds the
    half-hourly rate slot covering right now. Returns None on any failure
    (network, no matching product, parsing) so callers fall back to manual
    entry — this is a convenience lookup, not a requirement.
    """
    try:
        with urllib.request.urlopen(OCTOPUS_PRODUCTS_URL, timeout=timeout) as resp:
            products = json.loads(resp.read()).get("results", [])
        agile_codes = [
            p["code"] for p in products if "AGILE" in p.get("code", "").upper()
        ]
        if not agile_codes:
            return None
        product_code = agile_codes[0]
        tariff_code = f"E-1R-{product_code}-{region_letter}"
        rates_url = (
            f"https://api.octopus.energy/v1/products/{product_code}/"
            f"electricity-tariffs/{tariff_code}/standard-unit-rates/"
        )
        with urllib.request.urlopen(rates_url, timeout=timeout) as resp:
            rates = json.loads(resp.read()).get("results", [])
        now = datetime.now(timezone.utc)
        for rate in rates:
            valid_from = datetime.fromisoformat(
                rate["valid_from"].replace("Z", "+00:00")
            )
            valid_to_raw = rate.get("valid_to")
            valid_to = (
                datetime.fromisoformat(valid_to_raw.replace("Z", "+00:00"))
                if valid_to_raw
                else None
            )
            if valid_from <= now and (valid_to is None or now < valid_to):
                return rate["value_inc_vat"] / 100.0
        return None
    except Exception:  # noqa: BLE001 - best-effort, any failure just falls back
        return None


# Tried in order; each is a free FX source. Frankfurter has moved domains
# before (frankfurter.app -> frankfurter.dev), and any single provider can be
# down or blocked on a given network, so falling through to the next one is
# more robust than depending on exactly one host.
FX_RATE_URL_TEMPLATES: tuple = (
    "https://api.frankfurter.dev/v1/latest?from={from_currency}&to={to_currency}",
    "https://api.frankfurter.app/v1/latest?from={from_currency}&to={to_currency}",
    "https://api.exchangerate.host/latest?base={from_currency}&symbols={to_currency}",
)


def _fetch_yahoo_fx_rate(from_currency: str, to_currency: str, timeout: float) -> float:
    """Last-resort fallback via Yahoo Finance's unofficial chart endpoint.

    There's no official public Yahoo Finance API (it was retired in 2017),
    so this undocumented endpoint can change or start blocking requests
    without notice — that's why it's tried last, after the FX-specific
    providers above, rather than first.
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{from_currency}{to_currency}=X"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["chart"]["result"][0]["meta"]["regularMarketPrice"]


def fetch_fx_rate(
    from_currency: str, to_currency: str, timeout: float = 5.0
) -> Optional[float]:
    """Best-effort live exchange rate, trying each ``FX_RATE_URL_TEMPLATES``
    provider and finally Yahoo Finance.

    Returns None only if every provider fails (network, unknown currency,
    parsing) so callers fall back to manual entry rather than hardcoding a
    rate that goes stale.
    """
    for template in FX_RATE_URL_TEMPLATES:
        url = template.format(from_currency=from_currency, to_currency=to_currency)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data["rates"][to_currency]
        except Exception:  # noqa: BLE001 - best-effort, try the next provider
            continue
    try:
        return _fetch_yahoo_fx_rate(from_currency, to_currency, timeout)
    except Exception:  # noqa: BLE001 - best-effort, every provider failed
        return None


# Fallback used only if ``gpu_power_defaults.json`` cannot be read, so the
# script still prefills sensible figures out of the box. The shipped JSON
# file is the source of truth users are expected to edit; this mirrors its
# contents, and ``test_fallback_gpu_defaults_match_shipped_json`` fails if
# the two ever drift apart.
_FALLBACK_GPU_COST_POWER_DEFAULTS: tuple = (
    ("RTX 4090", 1600.0, 450.0),
    ("RTX 4080 SUPER", 1000.0, 320.0),
    ("RTX 4080", 1000.0, 320.0),
    ("RTX 4070 TI SUPER", 800.0, 285.0),
    ("RTX 4070", 550.0, 200.0),
    ("RTX 3090 TI", 900.0, 450.0),
    ("RTX 3090", 800.0, 350.0),
    ("RTX 3080", 500.0, 320.0),
    ("RTX 3070", 400.0, 220.0),
    ("RTX 6000 ADA", 6800.0, 300.0),
    ("A100", 10000.0, 400.0),
    ("H100", 25000.0, 700.0),
)


def load_gpu_defaults(path: Path = DEFAULT_GPU_DEFAULTS_PATH) -> tuple:
    """Load GPU price/power defaults from a JSON config file.

    Returns a tuple of ``(LABEL, cost_usd, power_watts)`` entries in the
    order the file lists them, so a more specific label can be listed
    before a more general one and still match first. Labels are upper-cased
    here because ``lookup_gpu_defaults`` matches against an upper-cased
    card name — a user who writes ``"rtx 5090"`` in the file gets the
    case-insensitive matching the README promises.

    Unlike ``load_pricing``, an unreadable file is not fatal: without
    pricing nothing can be computed, whereas these values only prefill
    prompts the user can override, so the run should continue. The file
    simply being absent falls back silently. Anything else — unparseable
    JSON, a non-object top level, an entry missing or mistyping a field,
    an empty list — means the user edited the file and got it wrong, so it
    warns on stderr before falling back. Silently ignoring a typo would
    leave them wondering why their card never matches.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _FALLBACK_GPU_COST_POWER_DEFAULTS
    except (json.JSONDecodeError, OSError) as exc:
        return _warn_bad_gpu_defaults(path, f"could not be read ({exc})")
    if not isinstance(data, dict):
        return _warn_bad_gpu_defaults(
            path, f"must contain a JSON object, got {type(data).__name__}"
        )
    raw_entries = data.get("gpus", [])
    if not isinstance(raw_entries, list):
        return _warn_bad_gpu_defaults(
            path, f'"gpus" must be a list, got {type(raw_entries).__name__}'
        )
    entries = []
    for index, item in enumerate(raw_entries):
        try:
            label = str(item["label"]).strip().upper()
            cost = _gpu_default_number(item["cost_usd"])
            power = _gpu_default_number(item["power_watts"])
            if not label:
                raise ValueError("label is empty")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            print(
                f"Warning: skipping gpus[{index}] in {path}: {exc}",
                file=sys.stderr,
            )
            continue
        entries.append((label, cost, power))
    if not entries:
        return _warn_bad_gpu_defaults(path, "listed no usable GPU entries")
    return tuple(entries)


def _warn_bad_gpu_defaults(path: Path, problem: str) -> tuple:
    """Warn that ``path`` is unusable and return the built-in defaults."""
    print(
        f"Warning: {path} {problem}; using built-in GPU defaults.",
        file=sys.stderr,
    )
    return _FALLBACK_GPU_COST_POWER_DEFAULTS


def _gpu_default_number(value: object) -> float:
    """Coerce a GPU cost/power field to a positive float.

    ``bool`` is rejected explicitly: it subclasses ``int``, so ``true``
    would otherwise be read as a cost of 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected a number, got {value!r}")
    number = float(value)
    if not number > 0:
        raise ValueError(f"expected a positive number, got {value!r}")
    return number


# The GPU price/power defaults in force for this run, read from
# ``gpu_power_defaults.json`` at import. Kept under its original public
# name so existing callers are unaffected by the move to a config file.
GPU_COST_POWER_DEFAULTS: tuple = load_gpu_defaults()


def lookup_gpu_defaults(
    gpu_name: str, defaults: Optional[tuple] = None
) -> Optional[tuple]:
    """Best-effort ``(cost_usd, power_watts)`` defaults for a detected GPU.

    Matches by substring against ``GPU_COST_POWER_DEFAULTS`` (loaded from
    ``gpu_power_defaults.json``) so a detected card pre-fills a realistic
    price/power pair instead of a generic default unrelated to the actual
    hardware. Returns None on no match — callers fall back to a generic
    default and the prompt still lets the user override.
    """
    if defaults is None:
        defaults = GPU_COST_POWER_DEFAULTS
    name = gpu_name.upper()
    for label, cost, power in defaults:
        if label in name:
            return cost, power
    return None


DESKTOP_REST_OF_SYSTEM_W = 100.0
LAPTOP_REST_OF_SYSTEM_W = 30.0

# User-facing prompt text for the two power-draw questions in
# ``interactive_local_setup``. Hoisted to module scope so the wording lives
# in one discoverable place (and can be localized later) instead of being
# buried inline in the function body.
EXTRA_POWER_DRAW_PROMPT = (
    "Extra power draw while generating — GPU/CPU load above idle (W), "
    "for when the machine is already on for other reasons "
    "(the additional watts the GPU/CPU pull when active, on top of "
    "the idle system draw)"
)
TOTAL_SYSTEM_DRAW_PROMPT = (
    "Total system power draw while running — GPU plus the rest of the PC "
    "(W), for when it's only powered on to run this "
    "(the entire system's power consumption while the GPU is under load, "
    "including the extra draw above)"
)


def rest_of_system_allowance_w(gpu_info: Optional[dict]) -> float:
    """Rough default allowance (W) for everything except the GPU itself —
    CPU, RAM, storage, and, for a desktop, a separate PSU/motherboard/fans.

    A laptop integrates all of that far more efficiently than a full-size
    desktop tower (no discrete PSU, far lower-power CPU/board), so a single
    flat number badly overshoots one form factor or undershoots the other.
    ``nvidia-smi`` reports "... Laptop GPU" in the name for mobile parts,
    which is the only signal available to tell them apart without extra
    tooling — still just a starting point to override, not a measurement.
    """
    if gpu_info and "laptop" in gpu_info.get("name", "").lower():
        return LAPTOP_REST_OF_SYSTEM_W
    return DESKTOP_REST_OF_SYSTEM_W


def format_gpu_summary(gpu_info: dict) -> str:
    """Render a detected GPU's info for display.

    ``nvidia-smi`` reports ``[N/A]`` for some fields on certain cards/drivers
    (``_safe_float`` turns that into ``None``); numeric formatting on a bare
    ``None`` raises ``TypeError``, so each field is guarded independently
    rather than formatted unconditionally.
    """
    mem = gpu_info.get("memory_total_mib")
    mem_s = f"{mem:.0f} MiB VRAM" if mem is not None else "VRAM unknown"
    draw = gpu_info.get("power_draw_w")
    draw_s = f"{draw:.0f} W draw" if draw is not None else "power draw unknown"
    limit = gpu_info.get("power_limit_w")
    limit_s = f"{limit:.0f} W limit" if limit is not None else "power limit unknown"
    return f"{gpu_info['name']} ({mem_s}, {draw_s} / {limit_s})"


# --------------------------------------------------------------------------
# Local model discovery (best-effort, non-fatal)
# --------------------------------------------------------------------------


def list_ollama_models(base_url: str) -> list:
    """List every model pulled into the local Ollama install (``GET /api/tags``)."""
    base_url = _validate_http_url(base_url)
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", []) if "name" in m]


def list_running_ollama_models(base_url: str) -> list:
    """List models Ollama currently has loaded in memory (``GET /api/ps``).

    This is "what's actually loaded right now", which is the model any
    benchmark request will hit — a better default than the full installed
    list from ``list_ollama_models`` when both are available.
    """
    base_url = _validate_http_url(base_url)
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/ps", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", []) if "name" in m]


def list_openai_compatible_models(base_url: str, api_key: Optional[str] = None) -> list:
    """List models an OpenAI-compatible server reports as available (``GET /v1/models``)."""
    base_url = _validate_http_url(base_url)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models", headers=headers, method="GET"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["id"] for m in data.get("data", []) if "id" in m]


def discover_local_models(backend: str, base_url: str) -> list:
    """Best-effort list of model names/ids available on a local endpoint.

    For Ollama, currently-loaded models (if any) are preferred over the full
    installed list, since that's what a benchmark request will actually hit.
    Returns an empty list on any failure (unreachable endpoint, unexpected
    response shape, etc.) — callers should treat that as "ask the user
    manually", not as an error worth surfacing.
    """
    try:
        if backend == "ollama":
            running = list_running_ollama_models(base_url)
            return running or list_ollama_models(base_url)
        return list_openai_compatible_models(base_url)
    except Exception:  # noqa: BLE001 - best-effort, any failure just falls back
        return []


# --------------------------------------------------------------------------
# Optional local throughput benchmark (best-effort, non-fatal)
# --------------------------------------------------------------------------

BENCHMARK_PROMPT = (
    "Write a short, three-sentence paragraph describing the weather in an "
    "imaginary coastal town."
)


def _validate_http_url(base_url: str) -> str:
    """Normalize and validate a base URL before building a request from it.

    ``base_url`` comes straight from free-form user input. Users commonly
    type a bare ``host[:port]`` (e.g. ``localhost:11434``) without a scheme,
    so a missing scheme is silently filled in with ``http://`` — the tool
    only ever makes HTTP requests to local endpoints, so that's the right
    default. Any other scheme (``file://``, ``ftp://``, etc.) is still
    rejected, since passing those through to ``urllib.request`` unchecked
    would be unsafe.

    Returns the normalized URL so callers can use the scheme-prefixed form.
    """
    import re

    url = base_url.strip()
    # urlsplit can't be used to detect the scheme here: it reads the "host" of
    # a bare "localhost:11434" as a scheme, which is exactly the input this
    # normalizes. Check for an explicit "<scheme>://" prefix instead.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    scheme = url.split("://", 1)[0].lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"base_url must start with http:// or https:// (got {base_url!r})"
        )
    # Without this, empty or whitespace-only input normalizes to a bare
    # "http://" and is accepted, then fails much later inside urllib with a
    # confusing URLError. Reject it here with a clear message instead. This
    # also catches malformed input like "://x", which normalizes to
    # "http://://x" (netloc ":", no host).
    if not urllib.parse.urlsplit(url).hostname:
        raise ValueError(f"base_url must include a host (got {base_url!r})")
    return url


def benchmark_ollama(base_url: str, model: str, num_predict: int = 200) -> float:
    """Measure tokens/sec against a local Ollama server.

    Uses Ollama's ``eval_count``/``eval_duration`` fields, which measure
    generation only (excludes prompt processing) — the same basis this
    script uses elsewhere for local throughput. Not directly comparable to
    ``benchmark_openai_compatible``'s wall-clock measurement (see its
    docstring).
    """
    base_url = _validate_http_url(base_url)
    payload = json.dumps(
        {
            "model": model,
            "prompt": BENCHMARK_PROMPT,
            "stream": False,
            "options": {"num_predict": num_predict},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    eval_count = data.get("eval_count")
    eval_duration_ns = data.get("eval_duration")
    if not eval_count or not eval_duration_ns:
        raise ValueError("Ollama response missing eval_count/eval_duration")
    return eval_count / (eval_duration_ns / 1e9)


def benchmark_openai_compatible(
    base_url: str, model: str, api_key: Optional[str] = None, max_tokens: int = 200
) -> float:
    """Measure tokens/sec against a local OpenAI-compatible chat endpoint.

    Timing is wall-clock — it includes prompt processing and network
    round-trip, not generation alone — and falls back to a word-count
    approximation if the response has no ``usage.completion_tokens`` field.
    Both make this systematically lower and not directly comparable to
    ``benchmark_ollama``'s generation-only ``eval_duration`` measurement.
    """
    base_url = _validate_http_url(base_url)
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    completion_tokens = (data.get("usage") or {}).get("completion_tokens")
    if not completion_tokens:
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        completion_tokens = max(1, len(text.split()))
    if elapsed <= 0:
        raise ValueError("Non-positive elapsed time measuring benchmark")
    return completion_tokens / elapsed


# --------------------------------------------------------------------------
# Last-run persistence (save/restore interactive defaults)
# --------------------------------------------------------------------------


def save_last_run(settings: dict, path: Path = DEFAULT_LAST_RUN_PATH) -> None:
    """Persist interactive run settings so the next run can reuse them as defaults.

    The saved JSON is a flat dict of the raw values entered (before they get
    closed into a row-builder lambda), annotated with a ``last_run_at``
    timestamp.
    """
    payload = dict(settings)
    payload["last_run_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_last_run(path: Path = DEFAULT_LAST_RUN_PATH) -> Optional[dict]:
    """Return the settings saved by the previous interactive run, or ``None``.

    Returns ``None`` when the file does not exist or is unreadable (a
    corrupted last-run file shouldn't block the script).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------
# Interactive prompt helpers (thin I/O layer over the pure functions above)
# --------------------------------------------------------------------------


def prompt_float(
    prompt: str, default: Optional[float] = None, *, minimum: Optional[float] = None
) -> float:
    """Prompt for a float, re-asking on non-numeric input.

    ``minimum`` (exclusive-or-equal, i.e. ``value >= minimum``) rejects
    values that would blow up downstream math (e.g. a tokens/sec or
    lifetime-years of 0 raises ``ValueError`` deep in the cost calculation,
    discarding every answer the user already gave).
    """
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Please enter a number >= {minimum}.")
            continue
        return value


def prompt_choice(prompt: str, choices: list, default: Optional[str] = None) -> str:
    choice_str = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    # Normalize choices for case-insensitive comparison while preserving the
    # original casing for display. A future mixed-case entry (e.g. "Claude")
    # would otherwise be silently rejected when the user types "claude".
    #
    # Build the lookup explicitly rather than via a dict comprehension so a
    # casefold collision (e.g. ["Claude", "claude"]) fails loudly instead of
    # silently dropping one entry — the dropped choice would otherwise be
    # unreachable even though it was explicitly offered. This is a no-op for
    # the current all-lowercase-ASCII callers, where casefold is identity.
    normalized_choices = {}
    for c in choices:
        key = c.casefold()
        if key in normalized_choices:
            raise ValueError(
                f"choices contain casefold-collision: "
                f"{normalized_choices[key]!r} and {c!r} both normalize to {key!r}"
            )
        normalized_choices[key] = c
    while True:
        raw = input(f"{prompt} ({choice_str}){suffix}: ").strip().casefold()
        if not raw and default:
            return default
        if raw in normalized_choices:
            return normalized_choices[raw]
        print(f"  Please enter one of: {choice_str}")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _prompt_custom_workload() -> Workload:
    """Prompt until the user enters a workload with a positive token total."""
    while True:
        requests_per_day = prompt_float(
            "Expected requests/day", default=1000.0, minimum=0
        )
        avg_input_tokens = prompt_float(
            "Average input tokens/request", default=500.0, minimum=0
        )
        avg_output_tokens = prompt_float(
            "Average output tokens/request", default=300.0, minimum=0
        )
        workload = Workload(requests_per_day, avg_input_tokens, avg_output_tokens)
        if workload.monthly_total_tokens > 0:
            return workload
        print(
            "  Workload produces zero total tokens/month — set requests/day and "
            "at least one token field above zero."
        )


def interactive_workload() -> list:
    """Prompt for one or more workload scenarios to compare."""
    print("\n== Workload ==")
    print("Choose the traffic scenario to size the comparison against:")
    for index, preset in enumerate(WORKLOAD_PRESETS, start=1):
        print(f"  {index}. {preset.label} — {preset.description}")
    all_index = len(WORKLOAD_PRESETS) + 1
    custom_index = len(WORKLOAD_PRESETS) + 2
    print(f"  {all_index}. Compare all scenarios")
    print(f"  {custom_index}. Custom numbers")

    while True:
        raw = input(f"Scenario [1-{custom_index}] [2]: ").strip() or "2"
        try:
            choice = int(raw)
        except ValueError:
            print(f"  Please enter a number from 1 to {custom_index}.")
            continue
        if 1 <= choice <= len(WORKLOAD_PRESETS):
            preset = WORKLOAD_PRESETS[choice - 1]
            return [(preset.key, preset.label, preset.to_workload())]
        if choice == all_index:
            return [(p.key, p.label, p.to_workload()) for p in WORKLOAD_PRESETS]
        if choice == custom_index:
            return [("custom", "Custom", _prompt_custom_workload())]
        print(f"  Please enter a number from 1 to {custom_index}.")


def interactive_local_setup() -> tuple:
    """Returns ``(row_builder, display_currency, usd_per_gbp, tokens_per_sec, settings)``.

    ``display_currency`` is ``"GBP"`` only when the user chose to pay
    electricity in GBP in ``"existing"`` mode; ``usd_per_gbp`` is the rate to
    convert the whole comparison table to GBP for display (None otherwise).
    ``tokens_per_sec`` is returned alongside the builder so the caller can
    scale each scenario's workload to what this throughput can actually
    produce (see ``scale_workload_to_local_capacity``) before pricing both
    local and hosted rows against it.
    ``settings`` is a dict of the raw values entered, suitable for passing to
    ``save_last_run()`` so the next run can reuse them as defaults.
    """
    print("\n== Local setup ==")
    # A single combined prompt replaces the previous two separate yes/no
    # questions (GPU detection, then throughput benchmark). Answering "y"
    # skips both steps; "n" or Enter falls through to the original
    # two-question flow so each can still be controlled independently.
    skip_benchmark = prompt_yes_no(
        "Skip benchmark (GPU detection + throughput)?", default=False
    )
    gpu_detection_enabled = not skip_benchmark
    benchmark_enabled = not skip_benchmark

    gpu_info = None
    if gpu_detection_enabled and prompt_yes_no(
        "Attempt to auto-detect an NVIDIA GPU via nvidia-smi?", default=True
    ):
        gpu_info = detect_nvidia_gpu()
        if gpu_info:
            # A single power.draw sample is noisy (especially on a laptop
            # GPU) — average a few quick readings for a steadier idle
            # baseline instead of trusting the one snapshot from detection.
            idle_avg = average_gpu_power_w()
            if idle_avg is not None:
                gpu_info["power_draw_w"] = idle_avg
            print(f"  Detected: {format_gpu_summary(gpu_info)}")
        else:
            print(
                "  No GPU detected (nvidia-smi not found or returned no data) — enter manually."
            )

    tokens_per_sec = None
    measured_load_power_w = None
    if benchmark_enabled and prompt_yes_no(
        "Attempt to benchmark a running local model endpoint (Ollama or OpenAI-compatible)?",
        default=True,
    ):
        backend = prompt_choice("Backend", ["ollama", "openai"], default="ollama")
        base_url = (
            input("Base URL [http://localhost:11434]: ").strip()
            or "http://localhost:11434"
        )

        detected_models = discover_local_models(backend, base_url)
        default_model = detected_models[0] if detected_models else None
        if detected_models:
            print(f"  Detected models: {', '.join(detected_models)}")
        prompt_label = "Model name as served locally"
        if default_model:
            model = (
                input(f"{prompt_label} [{default_model}]: ").strip() or default_model
            )
        else:
            model = input(f"{prompt_label}: ").strip()
        try:
            if backend == "ollama":
                tokens_per_sec, measured_load_power_w = measure_gpu_power_during(
                    lambda: benchmark_ollama(base_url, model)
                )
            else:
                tokens_per_sec, measured_load_power_w = measure_gpu_power_during(
                    lambda: benchmark_openai_compatible(base_url, model)
                )
            print(f"  Measured throughput: {tokens_per_sec:.1f} tokens/sec")
        except (
            Exception
        ) as exc:  # noqa: BLE001 - best-effort, any failure just falls back
            print(f"  Benchmark failed ({exc}) — enter throughput manually.")
            tokens_per_sec = None

    if tokens_per_sec is None:
        tokens_per_sec = prompt_float(
            "Measured or estimated tokens/sec", default=40.0, minimum=0.001
        )

    print(
        "\n  How should the cost of the hardware itself count?\n"
        "    existing — You already have this PC/GPU for other reasons; only the extra\n"
        "               electricity it uses while running counts (most people, most of\n"
        "               the time).\n"
        "    buying   — You're weighing whether to buy hardware specifically for this.\n"
        "    rent     — You'd rent GPU time in the cloud instead of using your own machine.\n"
    )
    mode = prompt_choice(
        "Hardware mode", ["existing", "buying", "rent"], default="existing"
    )

    if mode == "buying":
        gpu_defaults = lookup_gpu_defaults(gpu_info["name"]) if gpu_info else None
        if gpu_defaults:
            default_cost, default_power = gpu_defaults
            print(
                f"  Using typical price/power for {gpu_info['name']}: "
                f"${default_cost:,.0f}, {default_power:.0f} W — override below if yours differs."
            )
        else:
            default_cost = 1600.0
            default_power = (
                gpu_info["power_limit_w"]
                if gpu_info and gpu_info.get("power_limit_w")
                else 450.0
            )
        hardware_cost = prompt_float(
            "Hardware cost (USD)", default=default_cost, minimum=0
        )
        lifetime_years = prompt_float(
            "Expected hardware lifetime (years)", default=3.0, minimum=0.001
        )
        power_watts = prompt_float(
            "Power draw under load (W)", default=default_power, minimum=0
        )
        electricity_rate = prompt_float(
            "Electricity rate (USD/kWh)", default=0.15, minimum=0
        )
        settings = {
            "mode": "own",
            "tokens_per_sec": tokens_per_sec,
            "hardware_cost": hardware_cost,
            "lifetime_years": lifetime_years,
            "power_watts": power_watts,
            "electricity_rate_per_kwh": electricity_rate,
        }
        return (
            lambda workload: [
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "own",
                    hardware_cost=hardware_cost,
                    lifetime_years=lifetime_years,
                    power_watts=power_watts,
                    electricity_rate_per_kwh=electricity_rate,
                )
            ],
            "USD",
            None,
            tokens_per_sec,
            settings,
        )
    elif mode == "existing":
        idle_draw = gpu_info.get("power_draw_w") if gpu_info else None
        extra_default = None
        if measured_load_power_w is not None and idle_draw is not None:
            extra_default = round(max(measured_load_power_w - idle_draw, 0.0))
            print(
                f"  Extra power draw estimate: {extra_default:.0f} W (measured: "
                f"average load {measured_load_power_w:.0f} W minus average idle "
                f"{idle_draw:.0f} W)"
            )
        elif gpu_info:
            gpu_defaults = lookup_gpu_defaults(gpu_info["name"])
            if gpu_defaults:
                _, typical_load_power = gpu_defaults
                baseline_idle = idle_draw if idle_draw is not None else 0.0
                extra_default = round(max(typical_load_power - baseline_idle, 0.0))
                print(
                    f"  Extra power draw estimate: {extra_default:.0f} W (calculated: "
                    f"{gpu_info['name']}'s typical load power {typical_load_power:.0f} W "
                    f"minus its current idle draw {baseline_idle:.0f} W — run the "
                    f"benchmark for a measured value instead of this estimate)"
                )
        if extra_default is None:
            extra_default = 250.0
            print(
                f"  Extra power draw estimate: {extra_default:.0f} W (generic fallback — "
                "no GPU detected and no benchmark run to measure from)"
            )

        print(
            "\n  Both cost bases are shown below, since which applies depends on why the\n"
            "  machine is on — you don't have to pick one up front.\n"
        )
        power_watts_extra = prompt_float(
            EXTRA_POWER_DRAW_PROMPT,
            default=extra_default,
            minimum=0,
        )
        # Rough allowance for the rest of the system (CPU, RAM, storage,
        # motherboard) beyond just the GPU, for when the whole machine's
        # power is attributable to this use because it wouldn't otherwise be
        # on. A laptop draws far less here than a desktop tower, so the
        # allowance depends on the detected GPU's form factor rather than
        # one flat number for both.
        rest_of_system_w = rest_of_system_allowance_w(gpu_info)
        form_factor = (
            "laptop"
            if gpu_info and "laptop" in gpu_info["name"].lower()
            else "desktop (no GPU detected)" if not gpu_info else "desktop"
        )
        print(
            f"  Assuming a {form_factor} rest-of-system allowance of "
            f"{rest_of_system_w:.0f} W (CPU/RAM/storage, plus PSU/motherboard "
            "for a desktop) — override below if yours differs."
        )
        power_watts_total = prompt_float(
            TOTAL_SYSTEM_DRAW_PROMPT,
            default=power_watts_extra + rest_of_system_w,
            minimum=0,
        )

        # Octopus Agile is a UK-only tariff, so choosing to look it up already
        # answers "do you pay in GBP?" — asking that separately first was a
        # redundant question for anyone who was about to say yes to Agile.
        electricity_rate = None
        display_currency = "USD"
        usd_per_gbp = None
        gbp_rate = None
        if prompt_yes_no(
            "Look up your current unit rate live from Octopus Agile? "
            "(UK-only — implies you pay in GBP)",
            default=True,
        ):
            display_currency = "GBP"
            region = (
                input("Octopus Energy region letter (A-P; C = London) [C]: ")
                .strip()
                .upper()
                or "C"
            )
            gbp_rate = fetch_octopus_agile_rate(region)
            if gbp_rate is not None:
                print(f"  Current Octopus Agile unit rate: £{gbp_rate:.4f}/kWh")
            else:
                print("  Could not fetch a live Octopus Agile rate — enter manually.")
                gbp_rate = prompt_float(
                    "Electricity rate (GBP/kWh)", default=0.2483, minimum=0
                )
        elif prompt_yes_no(
            "Do you pay for electricity in GBP (e.g. UK)?", default=True
        ):
            display_currency = "GBP"
            gbp_rate = prompt_float(
                "Electricity rate (GBP/kWh)", default=0.2483, minimum=0
            )

        if display_currency == "GBP":
            live_usd_per_gbp = fetch_fx_rate("GBP", "USD")
            if live_usd_per_gbp is not None:
                print(f"  Current GBP→USD exchange rate: {live_usd_per_gbp:.4f}")
            else:
                print("  Could not fetch a live exchange rate — enter manually.")
            usd_per_gbp = prompt_float(
                "GBP→USD exchange rate (used internally to keep local and hosted "
                "costs comparable; the table itself is shown in GBP)",
                default=live_usd_per_gbp if live_usd_per_gbp is not None else 1.27,
                minimum=0.001,
            )
            electricity_rate = gbp_rate * usd_per_gbp
        else:
            electricity_rate = prompt_float(
                "Electricity rate (USD/kWh)", default=0.15, minimum=0
            )

        def build_existing_rows(workload: Workload) -> list:
            return [
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "existing",
                    power_watts=power_watts_extra,
                    electricity_rate_per_kwh=electricity_rate,
                    name="Local (already-on PC)",
                ),
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "always_on",
                    idle_watts=power_watts_total - power_watts_extra,
                    extra_watts=power_watts_extra,
                    electricity_rate_per_kwh=electricity_rate,
                    name="Local (kept on 24/7)",
                ),
            ]

        settings = {
            "mode": "existing",
            "tokens_per_sec": tokens_per_sec,
            "power_watts_extra": power_watts_extra,
            "power_watts_total": power_watts_total,
            "electricity_rate_per_kwh": electricity_rate,
            "display_currency": display_currency,
            "usd_per_gbp": usd_per_gbp,
        }
        return (
            build_existing_rows,
            display_currency,
            usd_per_gbp,
            tokens_per_sec,
            settings,
        )
    else:
        hourly_rate = prompt_float(
            "Rented GPU hourly rate (USD/hr)", default=2.50, minimum=0
        )
        settings = {
            "mode": "rent",
            "tokens_per_sec": tokens_per_sec,
            "hourly_rate": hourly_rate,
        }
        return (
            lambda workload: [
                build_local_row(
                    workload, tokens_per_sec, "rent", hourly_rate=hourly_rate
                )
            ],
            "USD",
            None,
            tokens_per_sec,
            settings,
        )


def interactive_provider_selection(pricing: dict) -> Optional[set]:
    """Prompt the user to choose which hosted models to compare against.

    Returns ``None`` to mean "compare against every model in ``pricing``",
    a (possibly empty) set of canonical keys to restrict the comparison to
    those models, or an empty set for "local only".

    ``all_keys`` is the set of canonical keys accepted by this prompt. It
    contains full ``provider/model`` keys only (e.g. ``"claude/opus-5"``,
    ``"deepseek/deepseek-v3"``) — provider-only input such as ``"claude"``
    is *not* a member of ``all_keys`` and is therefore treated as unknown.

    User input is matched case-insensitively via ``canonical_by_lower``
    (a ``{lowercased_key: canonical_key}`` mapping), so ``"Claude/Opus-5"``
    resolves to the canonical ``"claude/opus-5"``. Any input that does not
    match a canonical key after case-folding is treated as unknown and
    routed through the warning/re-prompt flow rather than being silently
    dropped or accepted.
    """
    print("\n== Hosted providers ==")
    print("Available models:")
    all_keys = []
    for provider_key, model_key, model_info in iter_models(pricing):
        full_key = f"{provider_key}/{model_key}"
        all_keys.append(full_key)
        print(f"  {full_key}: {model_info.get('display_name', full_key)}")
    if prompt_yes_no("Compare against all of the above?", default=True):
        return None
    raw = input(
        "Enter comma-separated keys to include (leave blank for local only): "
    ).strip()
    selected = {k.strip() for k in raw.split(",") if k.strip()}
    return selected


def run_interactive(use_defaults: bool = False) -> int:
    """Compare local hardware vs hosted providers for user-selected workloads.

    The interactive path asks for a workload preset (or custom traffic
    numbers), local hardware setup, and hosted model selection before
    rendering the cost table.

    When ``use_defaults`` is True, saved settings from a previous run are
    used without prompting (equivalent to passing ``--use-defaults`` on the
    command line).  If no saved file exists the flag is a no-op and the
    normal interactive flow runs instead.
    """
    print("LLM Cost Comparison — local vs hosted APIs")
    print("=" * 60)
    pricing = load_pricing()
    as_of = pricing.get("as_of", "unknown date")
    note = pricing.get("note", "")
    print(f"Hosted pricing as of {as_of}. {note}\n")

    defaults = None
    saved = load_last_run()
    if saved is not None:
        saved_at = saved.get("last_run_at", "unknown")
        if use_defaults:
            print(f"Using saved settings from last run ({saved_at}).")
            defaults = saved
        else:
            print(f"Saved settings from last run found ({saved_at}).")
            if prompt_yes_no("Use them as defaults?", default=True):
                defaults = saved
        print()

    if defaults is not None:
        # Fast path: replay saved settings — no prompts at all.
        settings = dict(defaults)
        tokens_per_sec = settings["tokens_per_sec"]
        mode = settings["mode"]
        if any(
            k in settings for k in ("workload", "workload_preset", "workload_presets")
        ):
            scenarios = _resolve_workload_scenarios(settings)
        else:
            # Backward-compatible fallback for last-run files saved before
            # workload choices were persisted.
            legacy = Workload(
                requests_per_day=96,
                avg_input_tokens=DEFAULT_AVG_INPUT_TOKENS,
                avg_output_tokens=DEFAULT_AVG_OUTPUT_TOKENS,
            )
            scenarios = [("legacy", "Saved default workload", legacy)]

        if mode == "own":
            local_row_builder = lambda workload: [
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "own",
                    hardware_cost=settings["hardware_cost"],
                    lifetime_years=settings["lifetime_years"],
                    power_watts=settings["power_watts"],
                    electricity_rate_per_kwh=settings["electricity_rate_per_kwh"],
                )
            ]
            display_currency = "USD"
            usd_per_gbp = None
        elif mode == "existing":
            display_currency = settings.get("display_currency", "USD")
            usd_per_gbp = settings.get("usd_per_gbp")
            electricity_rate = settings["electricity_rate_per_kwh"]

            # Saved electricity / FX rates go stale quickly — re-fetch when
            # the original run paid in GBP so the numbers stay current.
            if display_currency == "GBP":
                fresh_gbp = fetch_octopus_agile_rate()
                fresh_fx = fetch_fx_rate("GBP", "USD")
                if fresh_gbp is not None and fresh_fx is not None:
                    electricity_rate = fresh_gbp * fresh_fx
                    usd_per_gbp = fresh_fx
                    settings["electricity_rate_per_kwh"] = electricity_rate
                    settings["usd_per_gbp"] = fresh_fx
                    print(
                        f"  Fresh Octopus rate: £{fresh_gbp:.4f}/kWh, "
                        f"GBP→USD: {fresh_fx:.4f}"
                    )
                elif fresh_gbp is not None:
                    # FX fetch failed — convert with saved rate so at least
                    # the electricity rate is current.
                    electricity_rate = fresh_gbp * (usd_per_gbp or 1.27)
                    settings["electricity_rate_per_kwh"] = electricity_rate
                    print(
                        f"  Fresh Octopus rate: £{fresh_gbp:.4f}/kWh "
                        f"(FX rate unchanged: {usd_per_gbp or 1.27:.4f})"
                    )
                elif fresh_fx is not None:
                    usd_per_gbp = fresh_fx
                    settings["usd_per_gbp"] = fresh_fx
                    print(
                        f"  Fresh GBP→USD: {fresh_fx:.4f} (electricity rate unchanged)"
                    )

            local_row_builder = lambda workload: [
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "existing",
                    power_watts=settings["power_watts_extra"],
                    electricity_rate_per_kwh=electricity_rate,
                    name="Local (already-on PC)",
                ),
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "always_on",
                    idle_watts=settings["power_watts_total"]
                    - settings["power_watts_extra"],
                    extra_watts=settings["power_watts_extra"],
                    electricity_rate_per_kwh=electricity_rate,
                    name="Local (kept on 24/7)",
                ),
            ]
        else:  # rent
            local_row_builder = lambda workload: [
                build_local_row(
                    workload,
                    tokens_per_sec,
                    "rent",
                    hourly_rate=settings["hourly_rate"],
                )
            ]
            display_currency = "USD"
            usd_per_gbp = None

        selected_models = settings.get("selected_models")
        selected = set(selected_models) if selected_models is not None else None

        print(f"  Mode: {mode}, {tokens_per_sec:.1f} tok/s")
        if selected is None:
            print("  Providers: all")
        elif selected:
            print(f"  Providers: {', '.join(sorted(selected))}")
        else:
            print("  Providers: none (local only)")
        print()
    else:
        scenarios = interactive_workload()
        local_row_builder, display_currency, usd_per_gbp, tokens_per_sec, settings = (
            interactive_local_setup()
        )
        selected = interactive_provider_selection(pricing)
        settings["selected_models"] = sorted(selected) if selected is not None else None
        if len(scenarios) == len(WORKLOAD_PRESETS) and [k for k, _, _ in scenarios] == [
            p.key for p in WORKLOAD_PRESETS
        ]:
            settings["workload_presets"] = [k for k, _, _ in scenarios]
        else:
            key, _label, workload = scenarios[0]
            if key == "custom":
                settings["workload"] = {
                    "requests_per_day": workload.requests_per_day,
                    "avg_input_tokens": workload.avg_input_tokens,
                    "avg_output_tokens": workload.avg_output_tokens,
                }
            else:
                settings["workload_preset"] = key

    multiple = len(scenarios) > 1
    scenario_labels_rows = []
    scaled_scenarios = []
    for _key, label, workload in scenarios:
        effective_workload, feasible, coverage_pct = scale_workload_to_local_capacity(
            workload, tokens_per_sec
        )
        if not feasible:
            scaled_scenarios.append((label, coverage_pct))
        rows = list(local_row_builder(effective_workload))
        rows.extend(build_hosted_rows(effective_workload, pricing, selected))
        if display_currency == "GBP":
            rows = convert_rows_currency(rows, usd_per_gbp)
        scenario_labels_rows.append((label, rows))

    if scaled_scenarios:
        print(
            "Note: this local setup cannot produce the full requested traffic "
            "running 24/7 for every scenario below, so affected figures are "
            "scaled to what it can actually generate in a real month:"
        )
        for label, coverage_pct in scaled_scenarios:
            print(f"  - {label}: scaled to {coverage_pct:.0f}% of its original traffic")

    if multiple:
        print("\n== Results (all scenarios) ==")
        print(render_combined_table(scenario_labels_rows, currency=display_currency))
    else:
        label, rows = scenario_labels_rows[0]
        print(f"\n== {label} ==")
        print(render_table(rows, currency=display_currency))

    if prompt_yes_no("\nExport results to a file?", default=False):
        fmt = prompt_choice("Format", ["csv", "json"], default="csv")
        default_name = f"cost_comparison.{fmt}"
        out_path = Path(
            input(f"Output path [{default_name}]: ").strip() or default_name
        )
        if multiple:
            if fmt == "csv":
                export_combined_csv(
                    scenario_labels_rows, out_path, currency=display_currency
                )
            else:
                export_combined_json(
                    scenario_labels_rows, out_path, currency=display_currency
                )
        else:
            _label, rows = scenario_labels_rows[0]
            if fmt == "csv":
                export_csv(rows, out_path, currency=display_currency)
            else:
                export_json(rows, out_path, currency=display_currency)
        print(f"Wrote {out_path}")

    if defaults is None and prompt_yes_no(
        "\nSave these settings as defaults for the next run?", default=False
    ):
        save_last_run(settings)
        print(f"Saved {DEFAULT_LAST_RUN_PATH}")

    return 0


# --------------------------------------------------------------------------
# Non-interactive entry point (for scripting/CI/testing)
# --------------------------------------------------------------------------


def _require_keys(section: dict, required: list, context: str) -> None:
    missing = [k for k in required if k not in section]
    if missing:
        raise ConfigError(
            f"{context} config is missing required field(s): {', '.join(missing)}"
        )


def _require_numeric_fields(
    section: dict, fields: list, context: str, *, allow_zero: bool = True
) -> None:
    """Validate each field is a number, raising ``ConfigError`` if not.

    Without this, a stray string (e.g. a quoted ``"450"`` instead of ``450``)
    in ``hardware_cost``/``power_watts``/``hourly_rate`` etc. would fail with
    a raw ``TypeError`` deep inside the cost math instead of a clear,
    field-named error at config-load time.
    """
    bound = ">= 0" if allow_zero else "> 0"
    for field_name in fields:
        value = section[field_name]
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            ok = value >= 0 if allow_zero else value > 0
        if not ok:
            raise ConfigError(
                f"{context}.{field_name} must be a number {bound}, got {value!r}"
            )


def _resolve_workload_scenarios(config: dict) -> list:
    """Resolve the config's workload section to a list of scenarios.

    Exactly one of three shapes is accepted:
      * ``"workload"``: an explicit ``{requests_per_day, avg_input_tokens,
        avg_output_tokens}`` dict — the original, fully-custom shape.
      * ``"workload_preset"``: a single preset key (see ``WORKLOAD_PRESETS``).
      * ``"workload_presets"``: a list of preset keys, compared side by side.

    Returns a list of ``(key, label, Workload)`` tuples, mirroring
    ``interactive_workload``'s return shape so both paths share the same
    downstream printing/export logic.
    """
    provided = [
        k for k in ("workload", "workload_preset", "workload_presets") if k in config
    ]
    if not provided:
        raise ConfigError(
            "top-level config must include one of: workload, workload_preset, "
            "workload_presets"
        )
    if len(provided) > 1:
        raise ConfigError(
            f"top-level config must include only one of: workload, workload_preset, "
            f"workload_presets — got {', '.join(provided)}"
        )

    if "workload" in config:
        _require_keys(
            config["workload"],
            ["requests_per_day", "avg_input_tokens", "avg_output_tokens"],
            "workload",
        )
        for field_name in ("requests_per_day", "avg_input_tokens", "avg_output_tokens"):
            value = config["workload"][field_name]
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(
                    f"workload.{field_name} must be a non-negative number, got {value!r}"
                )
        try:
            workload = Workload(**config["workload"])
        except TypeError as exc:
            raise ConfigError(
                f"workload config has an unexpected field: {exc}"
            ) from exc
        if workload.monthly_total_tokens <= 0:
            raise ConfigError(
                "workload produces zero total tokens/month — set requests_per_day and "
                "at least one of avg_input_tokens/avg_output_tokens above zero"
            )
        return [("custom", "Custom", workload)]

    keys = (
        [config["workload_preset"]]
        if "workload_preset" in config
        else config["workload_presets"]
    )
    return [(p.key, p.label, p.to_workload()) for p in (get_preset(k) for k in keys)]


def run_non_interactive(
    config_path: Path, export_fmt: Optional[str], export_path: Optional[Path]
) -> int:
    """Run the comparison from a JSON config instead of interactive prompts.

    Config shape (most common case — hardware you already own):
        {
          "workload": {"requests_per_day": 1000, "avg_input_tokens": 500, "avg_output_tokens": 300},
          "local": {"mode": "existing", "power_watts": 450,
                     "electricity_rate_per_kwh": 0.15, "tokens_per_sec": 40},
          "pricing_file": "pricing.json",
          "selected_models": ["claude/opus-5", "deepseek/deepseek-v3"]
        }

    ``local.mode`` is one of:
      * ``"existing"`` — you already own the PC/GPU; only electricity counts
        (``power_watts``, ``electricity_rate_per_kwh``). Use the GPU/CPU's
        *extra* draw above idle if the machine is already on for other
        reasons, or the *whole system's* draw if it's only powered on to run
        this.
      * ``"own"`` — you're deciding whether to buy hardware specifically for
        this; adds ``hardware_cost`` and ``lifetime_years`` to amortize the
        purchase alongside electricity.
      * ``"rent"`` — a rented cloud GPU billed hourly (``hourly_rate``).

    In place of ``"workload"``, use ``"workload_preset": "<key>"`` for a
    single named scenario, or ``"workload_presets": ["<key>", ...]`` to
    compare several at once — see ``WORKLOAD_PRESETS`` for valid keys.
    Exactly one of ``workload`` / ``workload_preset`` / ``workload_presets``
    must be given.

    ``pricing_file``, if relative, is resolved against ``config_path``'s
    directory (not the process's working directory) so the example config
    works regardless of where the script is invoked from.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"config file {config_path} is not valid JSON: {exc}"
        ) from exc

    _require_keys(config, ["local"], "top-level")
    scenarios = _resolve_workload_scenarios(config)

    pricing_path = Path(config.get("pricing_file", DEFAULT_PRICING_PATH))
    if not pricing_path.is_absolute():
        pricing_path = config_path.parent / pricing_path
    # load_pricing raises ConfigError (wrapping FileNotFoundError for a
    # missing file, or JSONDecodeError for invalid JSON) with a clear
    # message. Let it propagate rather than swallowing it here: main()
    # already catches ConfigError, prints "Config error: <message>" to
    # stderr (no raw traceback), and exits non-zero, so callers that
    # invoke run_non_interactive directly (e.g. tests, other tooling)
    # still get the exception to handle as they see fit.
    pricing = load_pricing(pricing_path)
    selected = set(config["selected_models"]) if "selected_models" in config else None

    local_cfg = config["local"]
    _require_keys(local_cfg, ["mode", "tokens_per_sec"], "local")
    tokens_per_sec = local_cfg["tokens_per_sec"]
    # Mirror the interactive prompt's `minimum=0.001` guard: a zero or
    # negative throughput would silently propagate into the cost math
    # (hours_needed_for_workload raises ValueError deep inside, or worse,
    # produces nonsensical figures), so reject it here with a clear,
    # field-named config error instead.
    if (
        not isinstance(tokens_per_sec, (int, float))
        or isinstance(tokens_per_sec, bool)
        or tokens_per_sec <= 0
    ):
        raise ConfigError(
            f"local.tokens_per_sec must be a positive number, got {tokens_per_sec!r}"
        )
    mode = local_cfg["mode"]
    if mode == "own":
        _require_keys(
            local_cfg,
            [
                "hardware_cost",
                "lifetime_years",
                "power_watts",
                "electricity_rate_per_kwh",
            ],
            "local (mode=own)",
        )
        _require_numeric_fields(
            local_cfg,
            ["hardware_cost", "power_watts", "electricity_rate_per_kwh"],
            "local",
        )
        _require_numeric_fields(
            local_cfg, ["lifetime_years"], "local", allow_zero=False
        )

        def build_local(workload: Workload) -> ComparisonRow:
            return build_local_row(
                workload,
                local_cfg["tokens_per_sec"],
                "own",
                hardware_cost=local_cfg["hardware_cost"],
                lifetime_years=local_cfg["lifetime_years"],
                power_watts=local_cfg["power_watts"],
                electricity_rate_per_kwh=local_cfg["electricity_rate_per_kwh"],
            )

    elif mode == "existing":
        _require_keys(
            local_cfg,
            ["power_watts", "electricity_rate_per_kwh"],
            "local (mode=existing)",
        )
        _require_numeric_fields(
            local_cfg, ["power_watts", "electricity_rate_per_kwh"], "local"
        )

        def build_local(workload: Workload) -> ComparisonRow:
            return build_local_row(
                workload,
                local_cfg["tokens_per_sec"],
                "existing",
                power_watts=local_cfg["power_watts"],
                electricity_rate_per_kwh=local_cfg["electricity_rate_per_kwh"],
            )

    elif mode == "rent":
        _require_keys(local_cfg, ["hourly_rate"], "local (mode=rent)")
        _require_numeric_fields(local_cfg, ["hourly_rate"], "local")

        def build_local(workload: Workload) -> ComparisonRow:
            return build_local_row(
                workload,
                local_cfg["tokens_per_sec"],
                "rent",
                hourly_rate=local_cfg["hourly_rate"],
            )

    else:
        raise ConfigError(
            f"local.mode must be 'own', 'existing', or 'rent', got {mode!r}"
        )

    multiple = len(scenarios) > 1
    scenario_labels_rows = []
    scaled_scenarios = []
    for key, label, workload in scenarios:
        effective_workload, feasible, coverage_pct = scale_workload_to_local_capacity(
            workload, tokens_per_sec
        )
        if not feasible:
            scaled_scenarios.append((label, coverage_pct))
        rows = [build_local(effective_workload)] + build_hosted_rows(
            effective_workload, pricing, selected
        )
        scenario_labels_rows.append((label, rows))

    if scaled_scenarios:
        print(
            "Note: this local setup can't produce the full requested traffic "
            "running 24/7 for every scenario below — figures for these are "
            "scaled to what it can actually generate in a real month, so local "
            "and hosted costs stay directly comparable:"
        )
        for label, coverage_pct in scaled_scenarios:
            print(f"  - {label}: scaled to {coverage_pct:.0f}% of its original traffic")

    if export_fmt and export_path:
        if multiple:
            if export_fmt == "csv":
                export_combined_csv(scenario_labels_rows, export_path)
            else:
                export_combined_json(scenario_labels_rows, export_path)
        else:
            _label, rows = scenario_labels_rows[0]
            if export_fmt == "csv":
                export_csv(rows, export_path)
            else:
                export_json(rows, export_path)
        print(f"Wrote {export_path}")

    if multiple:
        print("\n== Results (all scenarios) ==")
        print(render_combined_table(scenario_labels_rows))
    else:
        label, rows = scenario_labels_rows[0]
        print(f"\n== {label} ==")
        print(render_table(rows))
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Read all inputs from --config instead of prompting.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file for --non-interactive mode (see run_non_interactive docstring).",
    )
    parser.add_argument(
        "--export", choices=["csv", "json"], help="Export results in this format."
    )
    parser.add_argument("--export-path", type=Path, help="Path to write the export to.")
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Skip prompts and reuse settings saved by a previous interactive run.",
    )
    parser.add_argument(
        "--update-pricing",
        action="store_true",
        help="Fetch latest DeepSeek pricing from the official API docs and exit.",
    )
    args = parser.parse_args(argv)

    if args.update_pricing:
        ok_ds = fetch_deepseek_pricing()
        ok_bd = fetch_bedrock_pricing()
        if ok_ds or ok_bd:
            print(f"Updated {DEFAULT_PRICING_PATH}")
            return 0
        print(
            "Failed to fetch pricing — check your network or try again later.",
            file=sys.stderr,
        )
        return 1

    if args.non_interactive:
        if not args.config:
            parser.error("--non-interactive requires --config")
        # --export without --export-path would otherwise silently export
        # nothing (run_non_interactive requires both to be truthy) — default
        # a path rather than let the flag be a no-op.
        if args.export and not args.export_path:
            args.export_path = Path(f"cost_comparison.{args.export}")
        try:
            return run_non_interactive(args.config, args.export, args.export_path)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 1

    if args.export or args.export_path:
        print(
            "Note: --export/--export-path only apply to --non-interactive mode; "
            "interactive mode asks about exporting at the end.",
            file=sys.stderr,
        )

    try:
        return run_interactive(use_defaults=args.use_defaults)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 1
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
