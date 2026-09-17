"""
token_economics.py

Cost/latency estimator used by the Model Gateway (Phase 3) to decide
which model tier to route a request to, and by the API layer to reject
or warn on requests that would blow a per-query cost budget.

Mental model for a .NET dev: this is analogous to a request-costing
middleware you might put in front of an expensive downstream service —
except the "cost" here is a token count, and different providers/models
are literally different price tiers of the same operation.
"""

from dataclasses import dataclass
from enum import Enum
import math


class ModelTier(str, Enum):
    EXTRACTION = "extraction"      # cheap/fast: table extraction, classification
    STANDARD = "standard"          # mid-tier: RAG Q&A, summarization
    FRONTIER = "frontier"          # expensive: multi-doc reasoning, complex analysis


# $ per 1M tokens (input, output) — update from provider pricing pages.
# Kept as data, not hardcoded logic, so it can be hot-reloaded from a
# config service / feature flag store in production.
PRICING_PER_MILLION = {
    ModelTier.EXTRACTION: {"input": 0.15, "output": 0.60},
    ModelTier.STANDARD:   {"input": 3.00, "output": 15.00},
    ModelTier.FRONTIER:   {"input": 15.00, "output": 75.00},
}

# Rough chars-per-token for English financial text (tables/numbers skew
# this lower than prose — treat as an estimate, not ground truth; the
# gateway should reconcile against actual usage returned by the API).
CHARS_PER_TOKEN = 3.8


@dataclass
class CostEstimate:
    tier: ModelTier
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    est_latency_ms: int


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_cost(
    prompt_text: str,
    tier: ModelTier,
    expected_output_tokens: int = 500,
) -> CostEstimate:
    input_tokens = estimate_tokens(prompt_text)
    pricing = PRICING_PER_MILLION[tier]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]

    # Rough latency model: base network/queue overhead + per-token
    # generation time, which varies significantly by tier. These
    # constants should be recalibrated from real p50/p95 traces
    # (Phase 7 observability feeds this back).
    base_latency = {ModelTier.EXTRACTION: 150, ModelTier.STANDARD: 300, ModelTier.FRONTIER: 500}
    ms_per_output_token = {ModelTier.EXTRACTION: 8, ModelTier.STANDARD: 18, ModelTier.FRONTIER: 35}

    est_latency = base_latency[tier] + (expected_output_tokens * ms_per_output_token[tier])

    return CostEstimate(
        tier=tier,
        input_tokens=input_tokens,
        estimated_output_tokens=expected_output_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
        est_latency_ms=est_latency,
    )


def choose_tier_for_intent(intent: str) -> ModelTier:
    """
    Static routing table for Phase 1 scoping purposes only.
    Phase 3 replaces this with a learned/classifier-based router plus
    a semantic cache lookup before any model call happens at all.
    """
    extraction_intents = {"table_extraction", "entity_extraction", "field_lookup"}
    frontier_intents = {"multi_doc_comparison", "risk_synthesis", "investment_thesis"}

    if intent in extraction_intents:
        return ModelTier.EXTRACTION
    if intent in frontier_intents:
        return ModelTier.FRONTIER
    return ModelTier.STANDARD


if __name__ == "__main__":
    sample_10k_excerpt = "Item 7. Management's Discussion and Analysis " * 200
    tier = choose_tier_for_intent("risk_synthesis")
    est = estimate_cost(sample_10k_excerpt, tier, expected_output_tokens=800)
    print(est)
