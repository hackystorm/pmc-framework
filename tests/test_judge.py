# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the external judge.

The judge is the gateway from a self-reported improvement (E1) to an
independently verified one (E2+). These tests cover the part that decides
whether a verdict can be trusted: the budget is spent, a malformed verdict is
refused, the evidence level is derived and not asserted, and the verdict is
marked on the taint ledger as the external source it is.

They exist so that a module which fails to import fails the suite, instead of
sitting in the tree as code nothing runs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from pmc.runtime.client import ChatError
from pmc.runtime.judge import (
    TAINT_SOURCE_EXTERNAL,
    ExternalJudgementError,
    JudgeRequest,
    _build_judge_prompt,
    _determine_evidence_level,
    default_judge,
    external_judge,
    get_judge,
    register_judge,
)
from pmc.runtime.providers import Provider, ResourceBudget


def _build_prompt(request: JudgeRequest) -> List[dict[str, str]]:
    return _build_judge_prompt(request)


class FakeClient:
    """A judge model that returns a fixed reply, or raises."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: List[List[dict[str, str]]] = []
        self.temperature: float | None = "unset"

    def complete(
        self, messages: List[dict[str, str]], *, temperature: float | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(messages)
        self.temperature = temperature

        class _Reply:
            text = self.reply

        return _Reply()


class ExplodingClient:
    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise ChatError("judge model unavailable")


class RecordingLedger:
    """A taint ledger that records what it was marked with."""

    def __init__(self) -> None:
        self.marks: List[tuple[str, str]] = []

    def mark(self, digest: str, source: str) -> None:
        self.marks.append((digest, source))


def _request(**overrides: Any) -> JudgeRequest:
    payload: Dict[str, Any] = dict(
        baseline_config={"layers": 2},
        candidate_config={"layers": 2},
        preference_pairs=[(0.4, 0.1)],
        eval_token_digest="a" * 64,
        evaluation_type="preference_pair",
        judge_model="judge-model",
    )
    payload.update(overrides)
    return JudgeRequest(**payload)


VERDICT = json.dumps(
    {
        "improved": True,
        "confidence": 0.93,
        "reason": "held-out gain is real",
        "measured_metrics": {"baseline_perplexity": 12.0, "candidate_perplexity": 9.0},
    }
)


def test_the_prompt_asks_for_a_strict_verdict():
    request = _request()
    messages = _build_prompt(request)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "JSON" in messages[0]["content"]
    # The judge is told the invariant it must not break, not left to infer it.
    assert "pure_generative_ratio" in messages[1]["content"]
    # And it is told not to accept a self-generated sample.
    assert "synthetic/self-generated" in messages[1]["content"]


def test_a_verdict_is_parsed_into_an_evidence_level():
    budget = ResourceBudget(network_requests_per_day=5)
    response = external_judge(FakeClient(VERDICT), _request(), budget)

    assert response.result.improved is True
    assert response.result.evidence_level == "E3"
    assert response.result.metrics["candidate_perplexity"] == 9.0
    assert response.raw_output == VERDICT


def test_the_budget_is_spent_for_the_call():
    budget = ResourceBudget(network_requests_per_day=5)
    external_judge(FakeClient(VERDICT), _request(), budget)
    assert budget.spent == 1


def test_an_exhausted_budget_refuses_the_judgement():
    budget = ResourceBudget(network_requests_per_day=0)
    with pytest.raises(ExternalJudgementError):
        external_judge(FakeClient(VERDICT), _request(), budget)


def test_invalid_json_is_refused():
    budget = ResourceBudget(network_requests_per_day=5)
    with pytest.raises(ExternalJudgementError):
        external_judge(FakeClient("not json"), _request(), budget)


def test_a_verdict_missing_keys_is_refused():
    budget = ResourceBudget(network_requests_per_day=5)
    incomplete = json.dumps({"improved": True})
    with pytest.raises(ExternalJudgementError):
        external_judge(FakeClient(incomplete), _request(), budget)


def test_a_failed_judge_call_is_refused():
    budget = ResourceBudget(network_requests_per_day=5)
    with pytest.raises(ExternalJudgementError):
        external_judge(ExplodingClient(), _request(), budget)


def test_the_verdict_is_marked_on_the_taint_ledger():
    ledger = RecordingLedger()
    budget = ResourceBudget(network_requests_per_day=5)
    external_judge(FakeClient(VERDICT), _request(), budget, taint_ledger=ledger)

    assert len(ledger.marks) == 1
    digest, source = ledger.marks[0]
    assert source == TAINT_SOURCE_EXTERNAL
    # The digest is of the verdict it actually received, so the mark can be
    # checked against the recorded output rather than trusted.
    assert digest == _digest_of(VERDICT)


def _digest_of(payload: str) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(json.loads(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def test_default_judge_spends_and_marks():
    ledger = RecordingLedger()
    provider = Provider(name="p", endpoint="http://127.0.0.1:9/v1", model="judge")
    budget = ResourceBudget(network_requests_per_day=5)

    response = default_judge(
        provider,
        budget,
        _request(),
        taint_ledger=ledger,
        client_factory=lambda: FakeClient(VERDICT),
    )

    assert response.result.evidence_level == "E3"
    assert budget.spent == 1
    assert ledger.marks and ledger.marks[0][1] == TAINT_SOURCE_EXTERNAL


def test_default_judge_marks_the_verdict_it_actually_received():
    ledger = RecordingLedger()
    provider = Provider(name="p", endpoint="http://127.0.0.1:9/v1", model="judge")
    budget = ResourceBudget(network_requests_per_day=5)

    default_judge(
        provider,
        budget,
        _request(),
        taint_ledger=ledger,
        client_factory=lambda: FakeClient(VERDICT),
    )

    digest, _ = ledger.marks[0]
    assert digest == _digest_of(VERDICT)


def test_a_judge_can_be_registered_and_retrieved():
    register_judge("test-judge", VERDICT)
    assert get_judge("test-judge") == VERDICT
    assert get_judge("never-registered") is None


@pytest.mark.parametrize(
    ("improved", "confidence", "expected"),
    [
        (False, 0.99, "E1"),
        (True, 0.79, "E1"),
        (True, 0.85, "E2"),
        (True, 0.92, "E3"),
        (True, 0.97, "E4"),
    ],
)
def test_the_evidence_level_follows_the_confidence(improved, confidence, expected):
    assert _determine_evidence_level(improved, confidence) == expected


def test_a_negative_verdict_never_rises_above_self_reported():
    # A judge that says the candidate is not an improvement grants nothing.
    verdict = json.dumps(
        {
            "improved": False,
            "confidence": 0.99,
            "reason": "no gain",
            "measured_metrics": {},
        }
    )
    budget = ResourceBudget(network_requests_per_day=5)
    response = external_judge(FakeClient(verdict), _request(), budget)
    assert response.result.evidence_level == "E1"
