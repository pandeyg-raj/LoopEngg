"""Cache-aware cost accounting for agentic loops.

The central object is a Trajectory: an append-only sequence of turns, where each
turn sends the entire prior transcript plus a new observation. Prior routing work
prices this as `input_tokens * list_price`, which ignores that almost every input
token after turn 1 is served from a prefix cache at roughly a tenth of list price.

This module computes both accountings so the gap can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

MTOK = 1_000_000


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices for one model.

    cache_read and cache_write default to the common provider pattern: reads are
    heavily discounted, writes carry a premium over base input.
    """

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def __post_init__(self) -> None:
        if self.cache_read_per_mtok is None:
            object.__setattr__(self, "cache_read_per_mtok", 0.10 * self.input_per_mtok)
        if self.cache_write_per_mtok is None:
            object.__setattr__(self, "cache_write_per_mtok", 1.25 * self.input_per_mtok)

    @property
    def cache_read_ratio(self) -> float:
        """cache_read / input. The term prior work implicitly sets to 1.0."""
        return self.cache_read_per_mtok / self.input_per_mtok


@dataclass
class Turn:
    """One iteration of the loop.

    context_tokens is the full prompt sent this turn. observation_tokens is what
    the tool returned *after* this turn, and therefore what gets appended to the
    transcript before the next one.
    """

    context_tokens: int
    output_tokens: int
    observation_tokens: int = 0
    turn_type: str = "unknown"
    delegatable: bool = False


@dataclass
class Trajectory:
    turns: list[Turn] = field(default_factory=list)
    task_id: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @classmethod
    def from_observations(
        cls,
        system_tokens: int,
        observations: Sequence[int],
        output_tokens: int | Sequence[int] = 400,
        turn_types: Sequence[str] | None = None,
        delegatable: Sequence[bool] | None = None,
        task_id: str = "",
    ) -> "Trajectory":
        """Build an append-only trajectory from a list of observation sizes.

        Context grows as C_{t+1} = C_t + o_t + r_t, which is the defining
        structure of a ReAct-style loop and the reason cost is superlinear in
        trajectory length.
        """
        n = len(observations)
        outs = [output_tokens] * n if isinstance(output_tokens, int) else list(output_tokens)
        types = list(turn_types) if turn_types else ["unknown"] * n
        dels = list(delegatable) if delegatable else [False] * n

        turns: list[Turn] = []
        context = system_tokens
        for i in range(n):
            turns.append(
                Turn(
                    context_tokens=context,
                    output_tokens=outs[i],
                    observation_tokens=observations[i],
                    turn_type=types[i],
                    delegatable=dels[i],
                )
            )
            context += outs[i] + observations[i]
        return cls(turns=turns, task_id=task_id)


@dataclass
class CostBreakdown:
    cached_read_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    weak_input_tokens: int = 0
    weak_output_tokens: int = 0
    dollars: float = 0.0

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            cached_read_tokens=self.cached_read_tokens + other.cached_read_tokens,
            uncached_input_tokens=self.uncached_input_tokens + other.uncached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            weak_input_tokens=self.weak_input_tokens + other.weak_input_tokens,
            weak_output_tokens=self.weak_output_tokens + other.weak_output_tokens,
            dollars=self.dollars + other.dollars,
        )


def score_flat(traj: Trajectory, strong: Pricing) -> CostBreakdown:
    """Price a trajectory the way prior routing work does: every input token at
    list price, no cache discount. This is the inflated baseline."""
    ctx = sum(t.context_tokens for t in traj.turns)
    out = sum(t.output_tokens for t in traj.turns)
    return CostBreakdown(
        uncached_input_tokens=ctx,
        output_tokens=out,
        dollars=(ctx * strong.input_per_mtok + out * strong.output_per_mtok) / MTOK,
    )


def score_cache_aware(
    traj: Trajectory,
    strong: Pricing,
    cache_efficiency: float = 1.0,
    cold_system_prompt: bool = True,
) -> CostBreakdown:
    """Price a trajectory with prefix caching.

    In an append-only transcript the only genuinely new input at turn t is the
    observation appended after turn t-1; everything before it is a cache hit.

    cache_efficiency in [0,1] scales the fraction of the theoretically-cacheable
    prefix that actually hits, covering block-boundary quantisation, TTL expiry
    and providers that require explicit breakpoints. Sweep it for sensitivity.
    """
    if not 0.0 <= cache_efficiency <= 1.0:
        raise ValueError("cache_efficiency must be in [0, 1]")

    total = CostBreakdown()
    for i, turn in enumerate(traj.turns):
        if i == 0:
            # Nothing cached yet on a fresh task, unless the system prompt is
            # shared across tasks and already warm.
            cacheable = 0
        else:
            prev = traj.turns[i - 1]
            # Everything except the newly appended observation was in the last
            # request or response, so it is already in the cache.
            cacheable = turn.context_tokens - prev.observation_tokens

        hit = int(cacheable * cache_efficiency)
        miss = turn.context_tokens - hit

        dollars = (
            hit * strong.cache_read_per_mtok
            + miss * strong.cache_write_per_mtok
            + turn.output_tokens * strong.output_per_mtok
        ) / MTOK

        total = total + CostBreakdown(
            cached_read_tokens=hit,
            uncached_input_tokens=miss,
            cache_write_tokens=miss,
            output_tokens=turn.output_tokens,
            dollars=dollars,
        )

    if cold_system_prompt is False and traj.turns:
        pass  # reserved: warm shared prefix across tasks
    return total


# ---------------------------------------------------------------------------
# Routing designs
# ---------------------------------------------------------------------------


def apply_design_b(
    traj: Trajectory,
    sigma: float = 0.05,
    weak_prompt_overhead: int = 200,
) -> tuple[Trajectory, int, int]:
    """Out-of-band delegation with context slicing.

    Delegatable observations are compressed to sigma of their size before they
    enter the main transcript. The weak model sees only the observation, not the
    running transcript, so the strong model's prefix cache is untouched and the
    saving compounds across every remaining turn.

    Returns the rewritten trajectory plus weak-model input/output token totals.
    """
    if not 0.0 < sigma < 1.0:
        raise ValueError("sigma must be in (0, 1)")

    weak_in = 0
    weak_out = 0
    new_obs: list[int] = []
    for turn in traj.turns:
        if turn.delegatable and turn.observation_tokens > 0:
            compressed = max(1, int(turn.observation_tokens * sigma))
            weak_in += turn.observation_tokens + weak_prompt_overhead
            weak_out += compressed
            new_obs.append(compressed)
        else:
            new_obs.append(turn.observation_tokens)

    rebuilt = Trajectory.from_observations(
        system_tokens=traj.turns[0].context_tokens if traj.turns else 0,
        observations=new_obs,
        output_tokens=[t.output_tokens for t in traj.turns],
        turn_types=[t.turn_type for t in traj.turns],
        delegatable=[t.delegatable for t in traj.turns],
        task_id=traj.task_id,
    )
    return rebuilt, weak_in, weak_out


def score_design_a(
    traj: Trajectory,
    strong: Pricing,
    weak: Pricing,
    cache_efficiency: float = 1.0,
) -> CostBreakdown:
    """In-band model swap: delegatable turns run on the weak model, but with the
    full running transcript and therefore a cold prefill.

    This is the implementation the literature does not distinguish from Design B,
    and the one the break-even analysis predicts will usually lose: it trades a
    cached read on the strong model for an uncached read on the weak one.
    """
    total = CostBreakdown()
    for i, turn in enumerate(traj.turns):
        if i == 0:
            cacheable = 0
        else:
            cacheable = turn.context_tokens - traj.turns[i - 1].observation_tokens
        hit = int(cacheable * cache_efficiency)
        miss = turn.context_tokens - hit

        if turn.delegatable:
            # Weak model has none of this prefix cached.
            dollars = (
                turn.context_tokens * weak.input_per_mtok
                + turn.output_tokens * weak.output_per_mtok
            ) / MTOK
            total = total + CostBreakdown(
                weak_input_tokens=turn.context_tokens,
                weak_output_tokens=turn.output_tokens,
                dollars=dollars,
            )
        else:
            dollars = (
                hit * strong.cache_read_per_mtok
                + miss * strong.cache_write_per_mtok
                + turn.output_tokens * strong.output_per_mtok
            ) / MTOK
            total = total + CostBreakdown(
                cached_read_tokens=hit,
                uncached_input_tokens=miss,
                cache_write_tokens=miss,
                output_tokens=turn.output_tokens,
                dollars=dollars,
            )
    return total


def score_design_b(
    traj: Trajectory,
    strong: Pricing,
    weak: Pricing,
    sigma: float = 0.05,
    cache_efficiency: float = 1.0,
    compressor: str = "weak",
) -> CostBreakdown:
    """Out-of-band delegation.

    compressor="weak" is condition R1 (edge tier does the compressing).
    compressor="strong" is condition B2, which isolates how much of the benefit
    is compression rather than cheap tokens -- the decomposition the paper turns on.
    compressor="script" is condition B3: deterministic truncation, no model call.
    """
    rebuilt, weak_in, weak_out = apply_design_b(traj, sigma=sigma)
    base = score_cache_aware(rebuilt, strong, cache_efficiency=cache_efficiency)

    if compressor == "script":
        return base
    if compressor == "strong":
        # Compression calls billed at strong-model rates, cold (a separate
        # request with only the observation in context).
        extra = (weak_in * strong.input_per_mtok + weak_out * strong.output_per_mtok) / MTOK
        return base + CostBreakdown(
            uncached_input_tokens=weak_in, output_tokens=weak_out, dollars=extra
        )
    if compressor == "weak":
        extra = (weak_in * weak.input_per_mtok + weak_out * weak.output_per_mtok) / MTOK
        return base + CostBreakdown(
            weak_input_tokens=weak_in, weak_output_tokens=weak_out, dollars=extra
        )
    raise ValueError(f"unknown compressor: {compressor!r}")


# ---------------------------------------------------------------------------
# Break-even analysis (Eq. 4 in the paper)
# ---------------------------------------------------------------------------


def breakeven_remaining_turns(
    strong: Pricing,
    weak: Pricing,
    sigma: float = 0.05,
    overhead_ratio: float = 0.0,
) -> float:
    """Minimum remaining turns for out-of-band delegation to pay for itself.

    Exact form before taking sigma -> 0:
        (T - j) > [p_in_W (1+e) + p_out_W * sigma] / [(1 - sigma) * p_cache_S]
    """
    num = weak.input_per_mtok * (1 + overhead_ratio) + weak.output_per_mtok * sigma
    den = (1 - sigma) * strong.cache_read_per_mtok
    return num / den


def design_a_is_profitable(strong: Pricing, weak: Pricing) -> bool:
    """In-band swapping helps only when the weak model's input price undercuts
    the strong model's *cached-read* price -- not its list price.

    This is the comparison the literature gets wrong, and it is much tighter than
    it looks: cached reads are typically ~10% of list, and small models are
    rarely that cheap.
    """
    return weak.input_per_mtok < strong.cache_read_per_mtok
