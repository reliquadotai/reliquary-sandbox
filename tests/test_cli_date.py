"""Synthetic reminder policies test validation; the exact native hash is CPU-probed."""

import copy
import hashlib
from dataclasses import replace

import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox.tool_routing import RoutingError, ToolRouter
from reliquary_sandbox_verifiers import cli

pytest_plugins = ("test_cli_bridge",)


@pytest.fixture
def date_policy(monkeypatch, capability):
    template = "<system-reminder>\n# currentDate\nToday's date is <native-episode-date>.\n</system-reminder>"
    monkeypatch.setattr(cli, "CLI_DATE_REMINDER_TEMPLATE_SHA256",
                        hashlib.sha256(template.encode()).hexdigest())

    def messages(day, prompt="trusted task"):
        return [{"role": "system", "content": cli.CLI_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": template.replace(cli.CLI_DATE_MARKER, day)},
                    {"type": "text", "text": prompt}]}]

    def frozen(day):
        return replace(capability, date_recipe=cli.CLI_DATE_RECIPE,
                       initial_template_sha256=cli.initial_template_sha256(
                           messages(day), "trusted task", context_date=day,
                           date_recipe=cli.CLI_DATE_RECIPE))

    return messages, frozen


def test_exact_native_date_template_is_portable_across_controlled_dates(date_policy):
    messages, frozen = date_policy
    first, tomorrow = frozen("2026-10-22"), frozen("2026-10-23")
    assert first.initial_template_sha256 == tomorrow.initial_template_sha256
    assert cli.initial_template_sha256(messages("2026-10-23", "task dated 2020-01-01"),
        "task dated 2020-01-01", context_date="2026-10-23",
        date_recipe=cli.CLI_DATE_RECIPE) == tomorrow.initial_template_sha256


def test_native_date_recipe_rejects_stale_altered_duplicate_or_extra_context(date_policy):
    messages, _ = date_policy
    original = messages("2026-10-22")
    stale = messages("2026-10-21")
    altered = copy.deepcopy(original)
    altered[1]["content"][0]["text"] += " altered reminder"
    duplicate = copy.deepcopy(original)
    duplicate[1]["content"][0]["text"] += " 2026-10-22"
    injected = copy.deepcopy(original)
    injected[1]["content"][1]["text"] += " injected 2026-10-22"
    extra = copy.deepcopy(original)
    extra[1]["content"].append({"type": "text", "text": "extra local context"})
    for changed in (stale, altered, duplicate, injected, extra):
        with pytest.raises(RoutingError):
            cli.initial_template_sha256(changed, "trusted task", context_date="2026-10-22",
                                        date_recipe=cli.CLI_DATE_RECIPE)


def test_actual_date_is_sealed_in_history_after_initial_template_validation(date_policy):
    messages, frozen = date_policy
    bridge = cli.SignedCLIToolBridge(ToolRouter(object()), frozen("2026-10-22"),
                                     prompt="trusted task", context_date="2026-10-22")
    bridge.validate_messages(messages("2026-10-22"))
    with pytest.raises(RoutingError):
        bridge.validate_messages(messages("2026-10-23"))


@pytest.mark.parametrize("invalid", [None, "20261022", "2026-1-1", "2026-02-30"])
def test_context_date_requires_canonical_real_iso_calendar_date(invalid):
    with pytest.raises(ValueError):
        cli.require_context_date(invalid)
