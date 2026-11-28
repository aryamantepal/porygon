"""The agent's own past trajectories within one run: pair selection for insights, and retrieval as examples.

ExpeL (Zhao et al., AAAI 2024, arXiv:2308.10144) keeps the trajectories of its training
trials. It compares each successful trajectory with each failed one of the same task to
extract insights, and at test time puts the most similar successful trajectories in the
prompt as examples. Here a "task" is the run's single goal and the trajectories are the
earlier trials of this run, so:

- `pick_pair` pairs the trial that just ended with the most recent stored trial of the
  opposite outcome (the reference pairs every success with every failure, which is
  quadratic in trials). With none, a success is shown with up to `SUCCESSES_SHOWN` of
  the most recent successes, and a failure has nothing to be compared with.
- `rank` orders stored successes for retrieval by bag-of-words cosine between the query
  (goal text plus first observation) and each stored trial's own goal text plus first
  observation, the same similarity Voyager's skill library uses. The reference embeds
  with a sentence model and searches a vector store. Ties go to the shorter trial, then
  the more recent. Within one run on one saved start the first observation is the same
  every trial, so the ranking is often decided by the tie-break; with visual observation
  the history keeps no text of the screen, so it rests on the goal alone.

Success and failure are the environment's verdict (`TrialOutcome`), never the model's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agents.examples import ExampleTrajectory, ExampleTurn
from agents.react.player import Turn
from agents.retrieval import cosine, words

SUCCESSES_SHOWN = 3  # successes shown together when a success has no failure to be compared with


@dataclass(frozen=True)
class Trajectory:
    """One finished trial: the agent's own turns (observation text, thought, action) and how the environment judged it."""

    trial: int
    goal: str
    turns: tuple[Turn, ...]
    success: bool
    reason: str
    steps: int
    first_observation: str  # kept apart because a capped trial no longer starts at its first turn

    def outcome_line(self) -> str:
        return f"STATUS: {'SUCCESS' if self.success else 'FAIL'} ({self.reason} after {self.steps} steps)"

    def event(self, similarity: float | None = None) -> dict:
        row = {"trial": self.trial, "success": self.success, "steps": self.steps}
        return {**row, "similarity": round(similarity, 4)} if similarity is not None else row


class TrajectoryStore:
    """The trajectories of finished trials, the last `capacity` kept, each capped at `max_turns` (its last turns)."""

    def __init__(self, capacity: int = 10, max_turns: int = 100) -> None:
        self.capacity = capacity
        self.max_turns = max_turns
        self.items: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        turns = trajectory.turns[max(len(trajectory.turns) - self.max_turns, 0) :]
        self.items.append(replace(trajectory, turns=turns))
        del self.items[: max(len(self.items) - self.capacity, 0)]

    def successes(self) -> list[Trajectory]:
        return [t for t in self.items if t.success]

    def pick_pair(self, current: Trajectory) -> tuple[str, list[Trajectory]]:
        """How to compare `current` with what is stored: ("compare", [success, failure]), ("successes", [...]), or ("none", [])."""
        opposite = [t for t in self.items if t.success != current.success]
        if opposite:
            other = opposite[-1]
            return "compare", [current, other] if current.success else [other, current]
        if current.success:
            return "successes", [*self.successes()[-(SUCCESSES_SHOWN - 1) :], current]
        return "none", []

    def rank(self, goal: str, first_observation: str, k: int) -> list[tuple[Trajectory, float]]:
        """The top `k` stored successes by similarity to the query, best first."""
        query = words(f"{goal} {first_observation}")
        scored = [(t, cosine(query, words(f"{t.goal} {t.first_observation}"))) for t in self.successes()]
        scored.sort(key=lambda pair: (-pair[1], pair[0].steps, -pair[0].trial))
        return scored[: max(k, 0)]


def as_example(trajectory: Trajectory, max_turns: int, tools: bool) -> ExampleTrajectory:
    """A stored trial as an `ExampleTrajectory`, so `agents/examples.py` renders it: its last `max_turns` turns, then the outcome.

    Under the tools interface a Result line follows the last turn, as in a live prompt, since no later Observation shows how its tool ended.
    """
    shown = trajectory.turns[max(len(trajectory.turns) - max_turns, 0) :]
    omitted = len(trajectory.turns) - len(shown)
    lines = [f"(the first {omitted} turns of this trial are not shown; the turns above are its last ones)"] if omitted else []
    if tools and shown and not shown[-1].action.startswith("(invalid"):
        lines.append(f"Result {len(shown)}: {shown[-1].result or '(the trial ended while this tool was still running)'}")
    lines.append(f"Outcome: reached the goal in {trajectory.steps} steps.")
    return ExampleTrajectory(trajectory.goal, tuple(ExampleTurn(t.observation, t.thought, t.action) for t in shown), "\n".join(lines))

