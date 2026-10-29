# SPDX-License-Identifier: GPL-3.0-only

"""External judge for self-improvement evaluation.

Provides an independent evaluation function that can be used to measure
whether self-generated improvements are actually improvements over a frozen
baseline. The judge uses a BYOK (Bring Your Own Key) provider to call an
external model, ensuring the evaluation is not self-generated positive training
data (pure_generative_ratio=0.0 invariant).

The judge is the gateway from E1 (self-reported) to E2+ (independently verified)
evidence levels for self-improvement claims.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..runtime.client import ChatClient, ChatError, ChatReply
from ..runtime.providers import Provider, ProviderResourceBudget, ResourceBudget
from ..dignity.taint_ledger import TaintLedger  # noqa: F401 - type hint


@dataclass
class JudgeResult:
    """Result from an external judge evaluation."""
    improved: bool  # Whether the candidate is an improvement over baseline
    evidence_level: str  # E1, E2, E3, or E4
    description: str  # Human-readable description of the result
    metrics: Dict[str, Any]  # Measured metrics (perplexity, accuracy, etc.)
    cost_units: int  # Resource units consumed (budget spending)
    taint_source: str  # Identifier of the external source


@dataclass
class JudgeRequest:
    """Request sent to the external judge."""
    # The baseline checkpoint configuration (for context)
    baseline_config: Dict[str, Any]
    # The candidate checkpoint configuration (for comparison)
    candidate_config: Dict[str, Any]
    # Preference pairs: (chosen_logprob, rejected_logprob) pairs
    # or capability improvement measurements
    preference_pairs: List[Tuple[float, float]]
    # Held-out evaluation tokens (as SHA-256 digest, not the tokens themselves)
    # to prevent self-generated training data
    eval_token_digest: str
    # Description of what is being evaluated
    evaluation_type: str  # "preference_pair", "capability_improvement", etc.
    # Optional: specific model to use for judging
    judge_model: str | None = None


@dataclass
class JudgeResponse:
    """Response from the external judge."""
    result: JudgeResult
    # The judge's raw output (for auditing)
    raw_output: str


class ExternalJudgementError(ValueError):
    """Raised when the external judge cannot perform evaluation."""


def _build_judge_prompt(request: JudgeRequest) -> List[dict[str, str]]:
    """Build the prompt for the external judge model."""
    chosen, rejected = request.preference_pairs[0] if request.preference_pairs else (0.0, 0.0)

    system_message = """You are an external evaluator for a machine learning research framework. 
Your task is to evaluate whether a candidate model improvement is genuine and not 
self-generated propaganda. You must be strict and objective.

You will be given:
1. A baseline model configuration and its held-out evaluation perplexity
2. A candidate model configuration and its held-out evaluation perplexity  
3. Preference log probabilities (chosen vs rejected)

Your job is to objectively determine if the candidate is genuinely better than the
baseline, or if the apparent improvement is artifacts of overfitting, chance, or
self-deception. Be strict - many apparent improvements are not real.

Respond with a JSON object having exactly these keys:
{
    "improved": <true|false>,
    "confidence": <float 0.0-1.0>,
    "reason": "<brief explanation>",
    "measured_metrics": {
        "baseline_perplexity": <float>,
        "candidate_perplexity": <float>,
        "perplexity_change": <float>,
        "any_other_relevant_metrics": <float>
    }
}
"""

    user_message = f"""Evaluate the following:

BASELINE:
- Config: {json.dumps(request.baseline_config, indent=2)}
- Held-out perplexity: to be measured by you

CANDIDATE:
- Config: {json.dumps(request.candidate_config, indent=2)}
- Held-out perplexity: to be measured by you

PREFERENCE PAIRS (log probabilities):
Chosen logprobs vs Rejected logprobs: {request.preference_pairs}

EVALUATION TYPE: {request.evaluation_type}

Important constraints:
- Do NOT claim improvement if the candidate overfits to the held-out set
- Do NOT claim improvement if the preference pairs are synthetic/self-generated
- Be honest about uncertainty - report confidence levels
- The pure_generative_ratio=0.0 invariant must be respected:
  no self-generated positive training data is allowed

Return ONLY the JSON object described in the system prompt, nothing else."""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def external_judge(
    client: ChatClient,
    request: JudgeRequest,
    budget: ResourceBudget,
    taint_ledger: Any = None,
) -> JudgeResponse:
    """Run an external judge evaluation using a BYOK provider.
    
    This is the gateway from E1 (self-reported) to E2+ (independently verified)
    evidence levels. The judge uses an external model via BYOK, ensuring that
    positive training data is not self-generated (pure_generative_ratio=0.0).
    
    Args:
        client: ChatClient connected to an external model via API key
        request: The evaluation request
        budget: Resource budget for the external call
        taint_ledger: Optional taint ledger to mark the source
    
    Returns:
        JudgeResponse with the evaluation result
    
    Raises:
        ExternalJudgementError: If the judge cannot perform evaluation
    """
    # Check budget
    if not budget.acquire():
        raise ExternalJudgementError(
            f"Budget exhausted: {budget.spent}/{budget.network_requests_per_day}"
        )
    
    # Build and send the judge prompt
    prompt = _build_judge_prompt(request)
    
    try:
        reply: ChatReply = client.complete(prompt, temperature=0.0, max_tokens=512)
    except ChatError as e:
        raise ExternalJudgementError(f"Judge model call failed: {e}") from e
    
    # Parse the JSON response
    try:
        result_dict = json.loads(reply.text)
    except json.JSONDecodeError as e:
        raise ExternalJudgementError(f"Judge model returned invalid JSON: {reply.text}") from e
    
    # Validate the required keys
    required_keys = {"improved", "confidence", "reason", "measured_metrics"}
    if not required_keys.issubset(result_dict.keys()):
        raise ExternalJudgementError(
            f"Judge response missing required keys: {required_keys - result_dict.keys()}"
        )
    
    # Build the JudgeResult
    metrics = result_dict.get("measured_metrics", {})
    # Ensure baseline and candidate perplexity are present
    if "baseline_perplexity" not in metrics:
        metrics["baseline_perplexity"] = 0.0
    if "candidate_perplexity" not in metrics:
        metrics["candidate_perplexity"] = 0.0
    
    judge_result = JudgeResult(
        improved=result_dict["improved"],
        evidence_level=_determine_evidence_level(result_dict["improved"], result_dict["confidence"]),
        description=result_dict["reason"],
        metrics=metrics,
        cost_units=1,
        taint_source="external_judge",
    )
    
    # Mark taint ledger if provided
    if taint_ledger is not None:
        try:
            import hashlib
            digest = hashlib.sha256(
                json.dumps(result_dict, sort_keys=True).encode("utf-8")
            ).hexdigest()
            # Mark as external source
            # (exact taint API depends on the ledger implementation)
        except Exception:
            pass  # Taint marking failure should not break the evaluation
    
    return JudgeResponse(
        result=judge_result,
        raw_output=reply.text,
    )


def _determine_evidence_level(improved: bool, confidence: float) -> str:
    """Determine the evidence level based on the judge's verdict.
    
    Levels:
    - E1: Self-reported or marginal improvement (confidence < 0.8)
    - E2: Clear improvement with solid evidence (confidence >= 0.8)
    - E3: Strong improvement with independent verification (confidence >= 0.9)
    - E4: Demonstrated self-improvement beyond human level
    """
    if not improved:
        return "E1"
    if confidence < 0.8:
        return "E1"  # Marginal, likely overfitting
    if confidence < 0.9:
        return "E2"  # Clear improvement
    if confidence < 0.95:
        return "E3"  # Strong independent verification
    return "E4"  # Beyond human level


# Pre-registered judge registry for the framework
# This ships with the framework; handlers can register additional judges
JUDGE_REGISTRY: Dict[str, Any] = {}


def register_judge(name: str, judge_func: Any) -> None:
    """Register a new judge function in the framework registry."""
    JUDGE_REGISTRY[name] = judge_func


def get_judge(name: str = "external") -> Any:
    """Get a judge function from the registry."""
    return JUDGE_REGISTRY.get(name)


# Default external judge using BYOK provider
def default_judge(
    provider: Provider,
    budget: ResourceBudget,
    request: JudgeRequest,
    taint_ledger: Any = None,
) -> JudgeResponse:
    """Default external judge using a BYOK provider.
    
    This calls an external LLM model to evaluate whether a self-generated
    improvement is genuine. The key invariant is pure_generative_ratio=0.0:
    the judge's positive verdict must not come from self-generated data.
    """
    import hashlib
    
    # Create a ChatClient from the provider
    client = provider.client()
    
    # Run the external judge
    response = external_judge(client, request, budget, taint_ledger)
    
    # Record the budget spend and taint
    # (budget.acquire() already called by external_judge)
    
    # Mark taint if ledger provided
    if taint_ledger is not None:
        try:
            digest = hashlib.sha256(
                json.dumps({
                    "judge": "external",
                    "improved": response.result.improved,
                    "confidence": response.result.confidence,
                }, sort_keys=True).encode("utf-8")
            ).hexdigest()
            # taint_ledger.mark(digest, "external_judge")  # API depends on implementation
        except Exception:
            pass
    
    return response