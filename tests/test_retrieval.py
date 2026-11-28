"""Utility-ranked retrieval (`utility_retrieval`) and memory decay (`memory_decay`).

Toy environments and a ScriptedModel: no ROM, no model server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agents.continual import ContinualHarness
from agents.continual.harness import HarnessState
from agents.harness import CrystalHarness
from agents.harness.memory import Journal, KnowledgeGraph
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from agents.retrieval import MemoryParams, Policy, Stats
from agents.voyager import Voyager
from agents.voyager.skills import SkillLibrary, parse
from environments.gridworld import GridWorld
from environments.lineworld import LineWorld
from experiments.run import make_player
from tests.test_harness import BACKBONE, planner_model, to_milestone
from tests.test_skill_parameters import CRITIC_NO, CRITIC_OK, _drive, kinds, play, scripted_voyager

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["LEFT", "RIGHT", "A"]
WALK = "Explain: none\nPlan:\n1) walk\nCode:\n```skill\nskill walk_right\nuntil x == 9 (max 12):\n  RIGHT\n```"
LIGHT = "Plan:\n1) reuse walk\nCode:\n```skill\nskill light_lamp\ncall walk_right\nA\n```"
BAD_CALL = "Plan:\n1) reuse walk\nCode:\n```skill\nskill bad_call\ncall walk_right\nuntil x == 99 (max 2):\n  LEFT\n```"
TASKS = ["Walk right to the end", "Light the lamp"]
SKILL = "skill walk_right\nuntil x == 9 (max 12):\n  RIGHT"


def policy(utility=True, decay=False, **params):
    return Policy(utility=utility, decay=decay, params=MemoryParams(**params))


def stats_with(policy_, q=None, last=0, strength=None):
    stats = policy_.fresh(last)
    if q is not None:
        stats.q = q
    if strength is not None:
        stats.strength = strength
    return stats


# The math


def test_q_moves_toward_the_reward_by_alpha():
    p = policy(alpha=0.1, q_init=0.5)
    stats = p.fresh(0)
    assert p.update(stats, 1.0) == (0.5, pytest.approx(0.55))
    assert p.update(stats, 0.0) == (pytest.approx(0.55), pytest.approx(0.495))
    assert stats.visits == 2
    for _ in range(200):
        p.update(stats, 1.0)
    assert stats.q == pytest.approx(1.0, abs=1e-6)


def test_q_init_and_alpha_are_configurable():
    p = policy(alpha=0.5, q_init=0.0)
    stats = p.fresh(0)
    assert stats.q == 0.0 and p.update(stats, 1.0)[1] == 0.5


def test_retention_is_the_ebbinghaus_curve_and_a_threshold_hides():
    p = policy(utility=False, decay=True, strength=10.0, threshold=0.3)
    stats = p.fresh(5)
    assert p.retention(stats, 5) == 1.0
    assert p.retention(stats, 15) == pytest.approx(math.exp(-1))
    assert not p.forgotten(stats, 15)
    assert p.forgotten(stats, 5 + 13)  # exp(-1.3) = 0.27 < 0.3


def test_recall_multiplies_strength_and_restarts_time_once_per_step():
    p = policy(utility=False, decay=True, strength=10.0, growth=2.0)
    stats = p.fresh(0)
    p.recall(stats, 8)
    assert (stats.strength, stats.last_access, stats.recalls) == (20.0, 8, 1)
    assert p.retention(stats, 28) == pytest.approx(math.exp(-1))  # t counts from the recall, S doubled
    p.recall(stats, 8)  # shown twice in one step
    assert stats.strength == 20.0 and stats.recalls == 1


def test_a_full_store_evicts_the_least_retained_forgotten_entry_and_none_otherwise():
    p = policy(utility=False, decay=True, strength=10.0, threshold=0.3)
    entries = {"old": p.fresh(0), "older": p.fresh(-5), "fresh": p.fresh(40)}
    assert p.victim(entries, 41) == "older"
    assert p.victim({"fresh": p.fresh(40)}, 41) is None
    assert policy(utility=True, decay=False).victim(entries, 1000) is None


def test_phase_one_filters_by_relevance_before_utility_ranks():
    p = policy(candidates=2, weight=0.5)
    entries = [("a", 0.9, stats_with(p, 0.2)), ("b", 0.8, stats_with(p, 0.9)), ("c", 0.7, stats_with(p, 1.0)), ("d", 0.1, stats_with(p, 1.0))]
    rows = p.rank(entries, 1, 0)
    assert [r.key for r in rows[:2]] == ["b", "a"]  # utility reorders the two relevant candidates
    assert [r.shown for r in rows[:2]] == [True, False]
    assert {r.key: r.reason for r in rows[2:]} == {"c": "not in phase 1", "d": "not in phase 1"}  # a high Q cannot buy a way in


def test_utility_can_overturn_a_small_similarity_lead_and_weight_zero_cannot():
    entries = lambda p: [("near", 0.60, stats_with(p, 0.1)), ("far", 0.50, stats_with(p, 0.9))]
    assert [r.key for r in policy(weight=0.7).rank(entries(policy()), 2, 0)][0] == "far"
    assert [r.key for r in policy(weight=0.0).rank(entries(policy()), 2, 0)][0] == "near"


def test_with_equal_utility_the_ranking_is_the_relevance_ranking_in_input_order_for_ties():
    p = policy()
    entries = [("x", 0.5, p.fresh(0)), ("y", 0.9, p.fresh(0)), ("z", 0.5, p.fresh(0)), ("w", 0.0, p.fresh(0))]
    assert [r.key for r in p.rank(entries, 3, 0)] == ["y", "x", "z", "w"]


def test_a_forgotten_entry_is_never_shown_and_retention_multiplies_relevance():
    p = policy(utility=False, decay=True, strength=10.0, threshold=0.3)
    entries = [("stale", 1.0, stats_with(p, last=0)), ("mid", 0.9, stats_with(p, last=12)), ("new", 0.5, stats_with(p, last=20))]
    rows = {r.key: r for r in p.rank(entries, 3, 20)}
    assert rows["stale"].shown is False and rows["stale"].reason == "forgotten"
    assert rows["mid"].score == pytest.approx(0.9 * math.exp(-0.8)) and rows["new"].score == 0.5
    assert [r.key for r in p.rank(entries, 3, 20)] == ["new", "mid", "stale"] and rows["new"].shown


def test_rows_report_similarity_utility_retention_and_score():
    p = policy(decay=True)
    row = p.rank([("a", 0.5, p.fresh(0))], 1, 0)[0]
    assert row.event() == {"id": "a", "similarity": 0.5, "utility": 0.5, "retention": 1.0, "score": 0.0, "shown": True}
    assert policy(utility=False, decay=True).rank([("a", 0.5, Stats(0.5, 50.0))], 1, 0)[0].event()["utility"] is None


def test_bad_parameters_are_rejected():
    for bad in ({"alpha": 0}, {"alpha": 1.5}, {"weight": 2}, {"candidates": 0}, {"strength": 0}, {"growth": 0.5}, {"threshold": 1}):
        with pytest.raises(ValueError):
            MemoryParams(**bad)


# Voyager: reward wiring


def trial(programs, critic, steps=80, **kwargs):
    model = scripted_voyager(TASKS, programs, critic=critic)
    player = Voyager(model, utility_retrieval=True, **kwargs)
    return player, play(player, LineWorld(), steps=steps, tasks=len(programs))


def test_a_called_skill_gets_reward_one_when_the_critic_accepts_the_program():
    player, events = trial([WALK, LIGHT], lambda n: CRITIC_OK)
    (update,) = kinds(events, "utility_update")
    assert update | {} == {"type": "utility_update", "store": "skills", "id": "walk_right", "reward": 1.0, "old": 0.5, "new": 0.55}
    assert player.library.stats["walk_right"].q == pytest.approx(0.55)
    assert player.library.stats["light_lamp"].q == 0.5  # the program itself is new, not a use


def test_a_called_skill_gets_reward_zero_when_the_critic_rejects_the_program():
    player, events = trial([WALK, LIGHT], lambda n: CRITIC_OK if n == 0 else CRITIC_NO, max_attempts=1)
    (update,) = kinds(events, "utility_update")
    assert update["reward"] == 0.0 and update["new"] == pytest.approx(0.45)


def test_an_execution_error_is_reward_zero_even_with_self_verification_off():
    player, events = trial([WALK, BAD_CALL], lambda n: CRITIC_OK, self_verification=False, max_attempts=1)
    assert [e["reward"] for e in kinds(events, "utility_update")] == [0.0]
    player, events = trial([WALK, LIGHT], lambda n: CRITIC_NO, self_verification=False)
    assert [e["reward"] for e in kinds(events, "utility_update")] == [1.0]  # no error counts as success when the critic is off


def test_a_skill_in_the_prompt_but_not_called_is_not_scored():
    player, events = trial([WALK, WALK.replace("walk_right", "walk_again")], lambda n: CRITIC_OK)
    assert kinds(events, "utility_update") == []
    assert player.library.stats["walk_right"].visits == 0


def test_the_retrieval_event_lists_every_candidate_with_its_numbers():
    player, events = trial([WALK, LIGHT], lambda n: CRITIC_OK, memory_decay=True)
    first, second = kinds(events, "retrieval")[:2]  # later ones are the filler tasks `play` runs on
    assert first["store"] == "skills" and first["candidates"] == []
    (row,) = second["candidates"]
    assert set(row) == {"id", "similarity", "utility", "retention", "score", "shown"} and row["id"] == "walk_right" and row["shown"] is True
    assert row["utility"] == 0.5 and row["retention"] == 1.0 and 0 < row["similarity"] <= 1


def test_learned_utility_changes_which_skill_the_prompt_shows():
    def prompt_programs(q_by_name):
        library = SkillLibrary(policy=policy(candidates=3))
        for name in ("walk_a", "walk_b", "walk_c"):
            library.add(parse(f"skill {name}\nRIGHT", KEYS), "walks right", "walk right", )
            library.stats[name].q = q_by_name[name]
        skills, rows = library.retrieve_ranked("walk right", 1)
        return [s.program.name for s in skills], rows

    assert prompt_programs({"walk_a": 0.5, "walk_b": 0.5, "walk_c": 0.5})[0] == ["walk_a"]  # no feedback yet: the similarity order
    shown, rows = prompt_programs({"walk_a": 0.1, "walk_b": 0.5, "walk_c": 0.95})
    assert shown == ["walk_c"] and [r.key for r in rows] == ["walk_c", "walk_b", "walk_a"]


def test_a_new_version_of_a_skill_starts_its_utility_over():
    library = SkillLibrary(policy=policy())
    library.add(parse("skill s\nRIGHT", KEYS), "d", "t")
    library.credit("s", 1.0)
    assert library.stats["s"].q > 0.5
    library.add(parse("skill s\nLEFT", KEYS), "d", "t")
    assert library.stats["s"].q == 0.5 and library.stats["s"].visits == 0


def test_a_decayed_skill_is_not_retrieved_and_a_recalled_one_lasts():
    library = SkillLibrary(policy=policy(utility=False, decay=True, strength=10.0, threshold=0.3))
    for name in ("walk_old", "walk_kept"):
        library.add(parse(f"skill {name}\nRIGHT", KEYS), "walks right", "walk right")
    library.clock = 8
    skills, _ = library.retrieve_ranked("walk right", 1)  # shows walk_kept (name order) and recalls it
    kept = skills[0].program.name
    other = "walk_old" if kept == "walk_kept" else "walk_kept"
    assert kept == "walk_kept"
    library.clock = 14
    skills, rows = library.retrieve_ranked("walk right", 5)
    assert [s.program.name for s in skills] == ["walk_kept"]
    assert {r.key: r.reason for r in rows}[other] == "forgotten"
    assert (library.stats["walk_kept"].strength, library.stats["walk_kept"].last_access) == (40.0, 14)  # recalled again: doubled again


def test_voyager_persists_utility_and_strength_only_with_the_switch_on(tmp_path):
    path = tmp_path / "library.json"
    player, _ = trial([WALK, LIGHT], lambda n: CRITIC_OK, library_path=path, memory_decay=True)
    saved = json.loads(path.read_text())
    assert set(saved["walk_right"]["memory"]) == {"q", "visits", "last_access", "strength", "recalls"}
    assert saved["walk_right"]["memory"]["q"] == pytest.approx(0.55)
    again = Voyager(scripted_voyager(TASKS, [WALK]), utility_retrieval=True, memory_decay=True, library_path=path)
    assert again.library.stats["walk_right"].q == pytest.approx(0.55)
    assert again.library.clock == max(s["memory"]["last_access"] for s in saved.values()) and again._clock == again.library.clock
    off = tmp_path / "off.json"
    play(Voyager(scripted_voyager(TASKS, [WALK, LIGHT]), library_path=off), LineWorld(), steps=60, tasks=2)
    assert all("memory" not in skill for skill in json.loads(off.read_text()).values())
    only = tmp_path / "only.json"
    trial([WALK, LIGHT], lambda n: CRITIC_OK, library_path=only)
    assert set(json.loads(only.read_text())["walk_right"]["memory"]) == {"q", "visits"}


def test_the_end_of_trial_trace_carries_the_stats_only_when_on():
    on, _ = trial([WALK, LIGHT], lambda n: CRITIC_OK)
    assert on.end_trial(None).trace["memory"]["walk_right"]["q"] == pytest.approx(0.55)
    off = Voyager(scripted_voyager(TASKS, [WALK]))
    assert "memory" not in off.end_trial(None).trace


def test_voyager_switches_need_the_library_and_are_reported():
    model = scripted_voyager(TASKS, [WALK])
    with pytest.raises(ValueError, match="need skill_library"):
        Voyager(model, utility_retrieval=True, skill_library=False)
    on = Voyager(model, utility_retrieval=True, memory_decay=True).mechanisms()
    assert on["utility_retrieval"] is True and on["memory_decay"] is True
    off = Voyager(model).mechanisms()
    assert off["utility_retrieval"] is False and off["memory_decay"] is False


# Continual Harness


def call(tool, **args):
    return json.dumps({"why": "b", "tool": tool, "args": args})


def continual(agent, **kwargs):
    def reply(messages):
        if messages[0].role == "system":
            return agent(messages)
        return '{"analysis": "none"}'

    return ContinualHarness(ScriptedModel(reply), **kwargs)


def decisions(player, env, presses):
    """Decide until `presses` presses were made; returns every event of the trace."""
    player.start_trial("Walk to the far right.")
    observation, events, made = env.reset(), [], 0
    while made < presses:
        decision = player.decide(observation, env.legal_actions())
        events += decision.trace["events"]
        if decision.action is None:
            break
        made += 1
        observation = env.step(decision.action).observation
    events += player.decide(observation, env.legal_actions()).trace["events"]  # lets the last tool finish and be scored
    return events


def test_continual_memory_reward_is_one_when_the_state_moved_and_zero_when_it_did_not():
    player = continual(lambda m: call("press", key="RIGHT"), utility_retrieval=True, carry_across_trials=True)
    entry, _ = player.harness.add_memory("fact", "the end is to the right", 3)
    events = decisions(player, LineWorld(), 2)
    updates = [e for e in events if e["type"] == "utility_update"]
    assert updates and all(u["store"] == "memory" and u["id"] == entry.id for u in updates)
    assert updates[0]["reward"] == 1.0 and updates[0]["old"] == 0.5 and updates[0]["new"] == pytest.approx(0.55)


def test_a_press_into_the_wall_scores_every_memory_in_the_prompt_zero():
    player = continual(lambda m: call("press", key="LEFT"), utility_retrieval=True, carry_across_trials=True)
    entry, _ = player.harness.add_memory("fact", "the end is to the right", 3)
    events = decisions(player, LineWorld(), 1)
    (update,) = [e for e in events if e["type"] == "utility_update"]
    assert update["reward"] == 0.0 and update["new"] == pytest.approx(0.45)


def test_a_skill_run_that_moves_the_state_scores_the_skill_and_the_memory():
    def agent(messages):
        if "- walk_right (effectiveness" in messages[1].content:
            return call("run_skill", name="walk_right")
        return call("press", key="RIGHT")

    player = continual(agent, utility_retrieval=True, carry_across_trials=True)
    player.harness.bind(KEYS)
    player.harness.add_skill(parse(SKILL, KEYS), "Walks right.", "high", 4, "refiner")
    entry, _ = player.harness.add_memory("fact", "the end is to the right", 3)
    events = decisions(player, LineWorld(), 12)
    scored = {}
    for u in events:
        if u["type"] == "utility_update":
            scored.setdefault((u["store"], u["id"]), u["reward"])  # the first run: the skill walks right from x=0
    assert scored[("skills", "walk_right")] == 1.0 and scored[("memory", entry.id)] == 1.0


def test_a_failed_skill_run_is_reward_zero():
    broken = parse("skill broken\nuntil x == 99 (max 2):\n  RIGHT", KEYS)
    player = continual(lambda m: call("run_skill", name="broken"), utility_retrieval=True, carry_across_trials=True)
    player.harness.add_skill(broken, "Fails.", "high", 4, "refiner")
    events = decisions(player, LineWorld(), 2)
    scores = [u for u in events if u["type"] == "utility_update" and u["store"] == "skills"]
    assert scores and scores[0]["reward"] == 0.0 and scores[0]["new"] == pytest.approx(0.45)


def test_continual_phase_two_picks_a_high_utility_memory_over_a_more_important_one():
    state = HarnessState(policy=policy(candidates=4))
    state.add_memory("low q", "x", 5)
    state.add_memory("high q", "y", 4)
    state.add_memory("other", "z", 2)
    state.stats["memory:mem_1"].q = 0.0
    state.stats["memory:mem_2"].q = 1.0
    text = state.memory_text(1, rank=True)
    assert "high q" in text and "low q" not in text and "(2 other entries not shown)" in text
    assert [r.key for r in state.retrieved["memory"]][0] == "mem_2"
    assert "low q" in state.memory_text(3)  # the Refiner's view is unranked and complete


def test_continual_with_equal_utility_shows_memory_as_it_always_did():
    ranked = HarnessState(policy=policy())
    plain = HarnessState()
    for step, (title, importance) in enumerate([("a", 2), ("b", 5), ("c", 5), ("d", 3)]):
        ranked.add_memory(title, "t", importance, step=step)
        plain.add_memory(title, "t", importance, step=step)
    assert ranked.memory_text(3, rank=True).splitlines()[:3] == plain.memory_text(3).splitlines()[:3]


def test_continual_skills_are_capped_by_utility_only_with_the_switch_on():
    def build(**kwargs):
        state = HarnessState(**kwargs)
        for name in ("s_a", "s_b", "s_c"):
            state.add_skill(parse(f"skill {name}\nRIGHT", KEYS), "d", "medium", 3, "refiner")
        return state

    state = build(policy=policy(candidates=3))
    state.stats["skill:s_c"].q = 1.0
    shown = state.skill_text(rank=1)
    assert "- s_c" in shown and "- s_a" not in shown and "2 other skills not shown" in shown
    assert state.skill_text().count("\n- ") == 2 and "- s_a" in state.skill_text()  # unranked: everything
    assert build().skill_text(rank=1) == build().skill_text()  # switch off: no cap


def test_continual_forgotten_memory_is_hidden_and_the_first_evicted():
    state = HarnessState(max_memory=2, policy=policy(utility=False, decay=True, strength=10.0, threshold=0.3))
    state.add_memory("old", "x", 5)
    state.clock = 5
    state.add_memory("newer", "y", 1)
    state.clock = 14
    assert "old" not in state.memory_text(5, rank=True) and "newer" in state.memory_text(5, rank=True)
    state.clock = 40
    entry, evicted = state.add_memory("third", "z", 1)  # both are forgotten, so importance 5 does not protect "old"
    assert entry is not None and evicted == ["mem_1"]  # the stalest goes
    plain = HarnessState(max_memory=1)
    plain.add_memory("old", "x", 5)
    assert plain.add_memory("weak", "y", 1) == (None, [])  # without decay the importance rule still refuses


def test_continual_recall_strengthens_what_the_prompt_showed():
    state = HarnessState(policy=policy(utility=False, decay=True, strength=10.0))
    state.add_memory("a", "x", 3)
    state.clock = 4
    state.memory_text(5, rank=True)
    assert state.stats["memory:mem_1"].strength == 20.0 and state.stats["memory:mem_1"].last_access == 4


def test_continual_rewriting_a_memory_starts_over_but_changing_importance_does_not():
    state = HarnessState(policy=policy())
    entry, _ = state.add_memory("a", "x", 3)
    state.credit("memory:mem_1", 1.0)
    state.revise_memory(entry, "x", 5)
    assert state.stats["memory:mem_1"].visits == 1
    state.revise_memory(entry, "new text", 5)
    assert state.stats["memory:mem_1"].visits == 0


def test_continual_stats_round_trip_and_are_absent_with_the_switches_off(tmp_path):
    state = HarnessState(policy=policy(decay=True))
    state.add_memory("a", "x", 3)
    state.add_skill(parse("skill s\nRIGHT", KEYS), "d", "high", 4, "refiner")
    state.clock = 7
    state.credit("memory:mem_1", 1.0)
    state.credit("skill:s", 0.0)
    data = json.loads(json.dumps(state.to_dict()))
    assert data["retrieval"]["clock"] == 7 and set(data["retrieval"]["stats"]) == {"memory:mem_1", "skill:s"}
    back = HarnessState.from_dict(data, policy=policy(decay=True))
    assert back.clock == 7 and back.stats["memory:mem_1"].q == pytest.approx(0.55) and back.stats["skill:s"].q == pytest.approx(0.45)
    assert "retrieval" not in HarnessState().to_dict()
    assert HarnessState.from_dict(HarnessState(policy=policy()).to_dict()).to_dict().get("retrieval") is None  # loading with the switches off drops it
    path = tmp_path / "harness.json"
    player = continual(lambda m: call("press", key="RIGHT"), utility_retrieval=True, harness_path=path, carry_across_trials=True)
    player.harness.add_memory("a", "x", 3)
    decisions(player, LineWorld(), 3)
    player.end_trial(None)
    reloaded = continual(lambda m: call("press", key="RIGHT"), utility_retrieval=True, harness_path=path, carry_across_trials=True)
    assert reloaded.harness.stats["memory:mem_1"].visits > 0


def test_continual_logs_a_retrieval_event_per_store_shown():
    player = continual(lambda m: call("press", key="RIGHT"), utility_retrieval=True, memory_decay=True, carry_across_trials=True)
    player.harness.add_memory("a", "x", 3)
    events = decisions(player, LineWorld(), 1)
    retrieval = next(e for e in events if e["type"] == "retrieval" and e["store"] == "memory")
    assert retrieval["candidates"][0]["id"] == "mem_1" and retrieval["candidates"][0]["retention"] == 1.0


# Crystal harness


def crystal_events(harness, plan=to_milestone, steps=50, **kwargs):
    model = planner_model(plan)
    player = CrystalHarness(model, backbone=BACKBONE, **kwargs) if harness is None else harness(model)
    player.start_trial("Reach the milestone in town.")
    env = GridWorld()
    observation, events = env.reset(), []
    for _ in range(steps):
        decision = player.decide(observation, env.legal_actions())
        events += decision.trace["events"]
        if decision.action is not None:
            observation = env.step(decision.action).observation
    return player, events


def test_notes_shown_at_a_planner_decision_are_scored_by_whether_the_tool_finished():
    def build(model):
        player = CrystalHarness(model, backbone=BACKBONE, utility_retrieval=True)
        player.knowledge.note("town", "the milestone is in the southeast of town")
        return player

    player, events = crystal_events(build)
    updates = [e for e in events if e["type"] == "utility_update"]
    ends = [e for e in events if e["type"] == "tool_end" and e["handler"] == "planner"]
    assert updates and updates[0]["store"] == "knowledge" and updates[0]["id"] == "town/notes/the milestone is in the southeast of town"
    assert ends[0]["outcome"] == "done" and updates[0]["reward"] == 1.0 and updates[0]["new"] == pytest.approx(0.55)


def test_a_tool_that_fails_scores_the_shown_notes_zero():
    def plan(state):
        return {"why": "w", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "reach the milestone"}

    def build(model):
        player = CrystalHarness(model, backbone=BACKBONE, utility_retrieval=True)
        player.knowledge.note("town", "the milestone is in the southeast of town")
        return player

    _, events = crystal_events(build, plan=plan, steps=6)
    updates = [e for e in events if e["type"] == "utility_update"]
    assert updates and all(u["reward"] == 0.0 for u in updates) and updates[0]["new"] == pytest.approx(0.45)


def test_the_knowledge_retrieval_event_explains_the_choice():
    def build(model):
        player = CrystalHarness(model, backbone=BACKBONE, utility_retrieval=True, memory_decay=True, notes_shown=1)
        player.knowledge.note("town", "the milestone is in the southeast of town")
        player.knowledge.note("town", "grass in town hides creatures")
        return player

    _, events = crystal_events(build, steps=3)
    retrieval = next(e for e in events if e["type"] == "retrieval" and e["store"] == "knowledge")
    rows = retrieval["candidates"]
    assert len(rows) == 2 and sum(r["shown"] for r in rows) == 1
    assert all({"similarity", "utility", "retention", "score"} <= set(r) for r in rows)
    assert rows[0]["similarity"] >= rows[1]["similarity"] or rows[0]["score"] >= rows[1]["score"]


def test_knowledge_utility_reorders_notes_within_the_candidates():
    graph = KnowledgeGraph(BACKBONE, policy=policy(candidates=2))
    graph.note("town", "grass hides creatures in town")
    graph.note("town", "milestone town southeast")
    graph.stats["town/notes/grass hides creatures in town"].q = 1.0
    graph.stats["town/notes/milestone town southeast"].q = 0.0
    shown = graph.relevant("milestone in town", notes_shown=1)["nodes"]["town"]["notes"]
    assert shown == ["grass hides creatures in town"] and graph.last_rows[0].shown


def test_a_decayed_note_is_hidden_and_a_full_note_list_drops_a_forgotten_note_first():
    graph = KnowledgeGraph(BACKBONE, policy=policy(utility=False, decay=True, strength=10.0, threshold=0.3))
    graph.note("town", "first town note")
    graph.clock = 3
    for index in range(9):
        graph.note("town", f"town note {index}")
    graph.clock = 20  # every note is forgotten (the first is the stalest); refresh one
    graph.stats["town/notes/town note 0"].last_access = 20
    assert graph.relevant("town", notes_shown=20)["nodes"]["town"]["notes"] == ["town note 0"]
    graph.note("town", "eleventh town note")  # 11 entries: drop a forgotten one, not the oldest by position if a staler exists
    notes = graph.data["nodes"]["town"]["notes"]
    assert len(notes) == 10 and "town note 0" in notes and "first town note" not in notes
    assert "town/notes/first town note" not in graph.stats
    plain = KnowledgeGraph(BACKBONE, policy=policy())
    for index in range(11):
        plain.note("town", f"n{index}")
    assert plain.data["nodes"]["town"]["notes"][0] == "n1"  # no decay: the oldest goes, as it always did


def test_journal_leads_decay_and_are_strengthened_by_being_shown():
    journal = Journal(policy=policy(utility=False, decay=True, strength=10.0, threshold=0.3))
    journal.add(0, "lead", "stale lead")
    fresh = journal.add(10, "lead", "fresh lead")
    shown = journal.recent(6, ("lead",), now=13)
    assert shown == [fresh]
    journal.recall(shown, 13)
    rows = journal.retrieval(("lead",), shown, 13)
    assert [(r.shown, r.reason) for r in rows] == [(False, "forgotten"), (True, "")]
    assert journal.recent(6, ("lead",)) == journal.entries  # no `now`: everything, as before
    assert Journal().recent(6, ("lead",), now=999) == []


def test_crystal_stats_round_trip_through_knowledge_path_only_with_the_switches_on(tmp_path):
    path = tmp_path / "knowledge.json"

    def build(model):
        player = CrystalHarness(model, backbone=BACKBONE, knowledge_path=path, utility_retrieval=True, memory_decay=True)
        player.knowledge.note("town", "the milestone is in the southeast of town")
        return player

    player, _ = crystal_events(build, steps=20)
    player.end_trial(TrialOutcome(False, "max_steps", 20))
    saved = json.loads(path.read_text())
    stats = saved["retrieval"]["stats"]["town/notes/the milestone is in the southeast of town"]
    assert set(stats) == {"q", "visits", "last_access", "strength", "recalls"} and stats["visits"] > 0 and saved["retrieval"]["clock"] == 20
    again = KnowledgeGraph(BACKBONE, path, policy(decay=True))
    assert again.clock == 20 and again.stats["town/notes/the milestone is in the southeast of town"].visits == stats["visits"]
    off_path = tmp_path / "off.json"
    graph = KnowledgeGraph(BACKBONE, off_path)
    graph.note("town", "x")
    assert "retrieval" not in json.loads(off_path.read_text())


def test_crystal_reports_its_switches():
    model = planner_model(to_milestone)
    assert CrystalHarness(model).mechanisms()["utility_retrieval"] is False
    on = CrystalHarness(model, utility_retrieval=True, memory_decay=True).mechanisms()
    assert on["utility_retrieval"] is True and on["memory_decay"] is True


# The switches off leave prompts and decisions as they were

# Taken from the commit before these switches existed (every request, decision, and end-of-trial trace).
# The buttons and tools Voyager and the Continual Harness digests are in tests/test_action_interface.py
# and tests/test_skill_parameters.py.
OFF_DIGESTS = {"crystal": "9181d945c8fd88c7", "crystal_notes": "a08dbc1fe7f5305c", "continual_memory": "65dd75322e777fd0"}


def test_crystal_prompts_and_decisions_are_unchanged_with_the_switches_off():
    model = planner_model(to_milestone)
    assert _drive(CrystalHarness(model, backbone=BACKBONE), GridWorld(), 50, model) == OFF_DIGESTS["crystal"]
    count = {"i": 0}

    def plan(state):
        count["i"] += 1
        if count["i"] % 3 == 0:
            return {"why": "w", "tool": "note", "args": {"node": "town", "text": f"note {count['i']} about milestone town"}, "intention": "reach the milestone"}
        return to_milestone(state)

    model = planner_model(plan)
    assert _drive(CrystalHarness(model, backbone=BACKBONE), GridWorld(), 50, model) == OFF_DIGESTS["crystal_notes"]


def test_continual_memory_prompts_are_unchanged_with_the_switches_off():
    count = {"i": 0}
    walk = "skill walk_right\nuntil x == 9 (max 12):\n  RIGHT"

    def reply(messages):
        if messages[0].role == "system":
            count["i"] += 1
            if count["i"] % 4 == 0:
                return json.dumps({"why": "b", "tool": "write_memory", "args": {"title": f"fact {count['i']}", "content": "walking right goes to the end of the line", "importance": 1 + count["i"] % 5}})
            tool = {"tool": "run_skill", "args": {"name": "walk_right"}} if "- walk_right (effectiveness" in messages[1].content else {"tool": "press", "args": {"key": "LEFT" if count["i"] % 2 else "RIGHT"}}
            return json.dumps({"why": "b", **tool})
        text = messages[0].content
        if text.startswith("You are a prompt optimization expert"):
            return "Walk right until the end."
        if text.startswith("You are a harness evolution system"):
            return json.dumps({"analysis": "x", "add": [{"name": "walk_right", "description": "Walks right.", "code": walk, "effectiveness": "high", "importance": 4}]})
        return '{"analysis": "none"}'

    model = ScriptedModel(reply)
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2, max_memory=4)
    assert _drive(player, LineWorld(), 60, model) == OFF_DIGESTS["continual_memory"]
    assert player.harness.memory and "retrieval" not in player.harness.to_dict()  # the digest covered memory writes and eviction


def test_the_switches_on_do_change_the_library_prompt_once_feedback_exists():
    # not a digest: with the switch on and feedback in, the prompt is allowed to differ, and it is logged why
    on, events = trial([WALK, LIGHT], lambda n: CRITIC_OK)
    assert kinds(events, "retrieval") and kinds(events, "utility_update")
    off = Voyager(scripted_voyager(TASKS, [WALK, LIGHT]))
    assert "retrieval" not in {e["type"] for e in play(off, LineWorld(), steps=60, tasks=2)}


# make_player and the configs


def test_make_player_wires_the_switches_and_tuning_for_all_three_agents():
    model = ScriptedModel(lambda m: "")
    tuning = {"utility_alpha": 0.3, "utility_q_init": 0.2, "utility_weight": 0.7, "utility_candidates": 4, "decay_strength": 20, "decay_growth": 3, "decay_threshold": 0.4}
    for name in ("voyager", "continual", "crystal_harness"):
        off = make_player({"name": name}, model)
        assert off.mechanisms()["utility_retrieval"] is False and off.mechanisms()["memory_decay"] is False and not off.policy.active
        on = make_player({"name": name, "utility_retrieval": True, "memory_decay": True, **tuning}, model)
        assert on.mechanisms()["utility_retrieval"] is True and on.mechanisms()["memory_decay"] is True
        assert on.policy.params == MemoryParams(alpha=0.3, q_init=0.2, weight=0.7, candidates=4, strength=20, growth=3, threshold=0.4)
        only = make_player({"name": name, "utility_retrieval": True}, model)
        assert only.mechanisms()["utility_retrieval"] is True and only.mechanisms()["memory_decay"] is False
    assert make_player({"name": "continual", "skills_shown": 2}, model).skills_shown == 2
    assert make_player({"name": "crystal_harness", "notes_shown": 3}, model).notes_shown == 3


def test_the_utility_configs_turn_on_exactly_the_switches_they_name():
    expected = {
        "gridworld_7b_voyager_utility": (True, True),
        "gridworld_7b_voyager_utility_only": (True, False),
        "gridworld_7b_continual_utility": (True, True),
        "gridworld_7b_crystal_harness_utility": (True, True),
    }
    model = ScriptedModel(lambda m: "")
    for name, (utility, decay) in expected.items():
        spec = json.loads((ROOT / f"experiments/{name}.json").read_text())
        player = make_player(spec["agent"], model)
        assert (player.mechanisms()["utility_retrieval"], player.mechanisms()["memory_decay"]) == (utility, decay), name
        base = json.loads((ROOT / f"experiments/{name.removesuffix('_utility_only').removesuffix('_utility')}.json").read_text())
        changed = {k for k in spec["agent"] if spec["agent"][k] != base["agent"].get(k)}
        assert changed == ({"utility_retrieval", "memory_decay"} if decay else {"utility_retrieval"}), name
        assert {k: v for k, v in spec.items() if k != "agent"} == {k: v for k, v in base.items() if k != "agent"}


def test_the_runner_logs_retrieval_and_utility_events_as_json(tmp_path):
    from evaluation.goals import Goal
    from evaluation.runner import Budgets, run

    def logged(player, env, goal, name):
        run(run_dir=tmp_path / name, env=env, player=player, goal=goal, budgets=Budgets(max_steps=40))
        lines = (tmp_path / name / "trials/000/steps.jsonl").read_text().splitlines()
        return {e["type"] for r in map(json.loads, lines) if r["kind"] == "decision" for e in (r["decision"]["trace"] or {}).get("events", [])}

    both = {"utility_retrieval": True, "memory_decay": True}
    lamp = Goal("never", "Light the lamp at the far right end.", {"x": 99})  # unreachable, so the trial outlives both programs
    assert {"retrieval", "utility_update"} <= logged(Voyager(scripted_voyager(TASKS, [WALK, LIGHT]), **both), LineWorld(), lamp, "voyager")
    player = continual(lambda m: call("press", key="RIGHT"), carry_across_trials=True, **both)
    player.harness.add_memory("fact", "the end is to the right", 3)
    assert {"retrieval", "utility_update"} <= logged(player, LineWorld(), lamp, "continual")
    milestone = Goal("milestone", "Reach the milestone in town.", {"milestone": True})
    player = CrystalHarness(planner_model(to_milestone), backbone=BACKBONE, **both)
    player.knowledge.note("town", "the milestone is in the southeast of town")
    assert {"retrieval", "utility_update"} <= logged(player, GridWorld(), milestone, "crystal")
