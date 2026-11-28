"""Parameterized skills (`skill_parameters`) and testing them before they stay (`test_skills`).

LineWorld and GridWorld with a ScriptedModel: no ROM, no model server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agents.continual import ContinualHarness
from agents.continual.harness import HarnessState
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from agents.voyager import Voyager, prompts
from agents.voyager.skills import Run, SkillError, SkillLibrary, execute, parse, parse_all
from environments.gridworld import GridWorld, GridWorldConfig
from environments.lineworld import LineWorld
from environments.protocol import ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run
from experiments.run import make_player

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["LEFT", "RIGHT", "A"]
GRID_KEYS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
LAMP = Goal("lamp", "Light the lamp at the far right end.", {"lamp": True})
WALK_TO = "skill walk_to target\nuntil x == $target (max 12):\n  RIGHT"


def go(program, env, library=None, args=(), keys=None):
    """Run a program to its end on `env`; returns the Run, or raises its SkillError."""
    counted = Run()
    steps = execute(program, library, env.reset(), counted, keys, args)
    try:
        action = next(steps)
        while True:
            action = steps.send(env.step(action).observation)
    except StopIteration:
        return counted


def grid():
    return GridWorld(GridWorldConfig(mode=ObservationMode.STRUCTURED))


# The language


def test_a_skill_without_parameters_parses_the_same_with_the_switch_on():
    source = "skill demo\nRIGHT x2\nrepeat 2:\n  LEFT\ncall other"
    assert parse(source, KEYS, parameters=True) == parse(source, KEYS) and parse(source, KEYS).params == ()


def test_a_parameter_stands_where_a_json_value_goes():
    program = parse(WALK_TO, KEYS, parameters=True)
    assert program.params == ("target",)
    env = LineWorld()
    assert go(program, env, args=(3,)).presses == 3 and env.get_progress()["x"] == 3


def test_arguments_bind_by_position():
    program = parse("skill between low high\nuntil x >= $low (max 12):\n  RIGHT\nuntil x >= $high (max 12):\n  RIGHT", KEYS, parameters=True)
    env = LineWorld()
    assert go(program, env, args=(2, 5)).presses == 5 and env.get_progress()["x"] == 5


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "`walk_to` takes 1 argument \\(target\\) but got 0"),
        ((1, 2), "`walk_to` takes 1 argument \\(target\\) but got 2"),
    ],
)
def test_missing_and_extra_arguments_are_execution_errors(args, message):
    with pytest.raises(SkillError, match=message):
        go(parse(WALK_TO, KEYS, parameters=True), LineWorld(), args=args)


def test_arguments_for_a_skill_without_parameters_are_an_error():
    with pytest.raises(SkillError, match="takes no arguments but got 1"):
        go(parse("skill s\nRIGHT", KEYS, parameters=True), LineWorld(), args=(1,))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("skill s\nuntil x == $nope (max 3):\n  RIGHT", r"`\$nope` is not a parameter of this skill \(this skill declares no parameters\)"),
        ("skill s a\nuntil x == $b (max 3):\n  RIGHT", r"\(this skill declares a\)"),
        ("skill s a a\nRIGHT", "at most 4 parameters, with different names"),
        ("skill s a b c d e\nRIGHT", "at most 4 parameters"),
        ("skill s a\ncall other $b", "`\\$b` is not a parameter"),
        ("skill s\ncall other [4, 0", "unbalanced"),
        ("skill s\ncall other a@b", "not a place id, an int, or a JSON value"),
    ],
)
def test_parameter_parse_errors_are_readable(source, message):
    with pytest.raises(SkillError, match=message):
        parse(source, KEYS, parameters=True)


def test_call_passes_arguments_and_the_run_records_them():
    walk = parse(WALK_TO, KEYS, parameters=True)
    main = parse("skill main\ncall walk_to 4", KEYS, parameters=True)
    env = LineWorld()
    counted = go(main, env, {"walk_to": walk})
    assert env.get_progress()["x"] == 4 and counted.calls == [("walk_to", (4,))]


def test_call_arguments_are_json_values_place_ids_or_the_callers_own_parameters():
    main = parse('skill outer t\ncall inner $t door_0 [4, 0] "a b" 2.5 true', KEYS, parameters=True)
    assert main.body[0].args[1:] == ("door_0", [4, 0], "a b", 2.5, True) and main.body[0].args[0].name == "t"
    inner = parse("skill inner a b c d\nRIGHT", KEYS, parameters=True)
    outer = parse("skill outer t\ncall inner $t door_0 [4, 0] 7", KEYS, parameters=True)
    counted = Run()
    steps = execute(parse("skill top\ncall outer 3", KEYS, parameters=True), {"outer": outer, "inner": inner}, LineWorld().reset(), counted)
    next(steps)
    assert counted.calls == [("outer", (3,)), ("inner", (3, "door_0", [4, 0], 7))]


def test_a_call_with_the_wrong_number_of_arguments_fails_when_it_runs():
    walk = parse(WALK_TO, KEYS, parameters=True)
    for source, got in (("skill main\ncall walk_to", 0), ("skill main\ncall walk_to 1 2", 2)):
        with pytest.raises(SkillError, match=f"`walk_to` takes 1 argument \\(target\\) but got {got}"):
            go(parse(source, KEYS, parameters=True), LineWorld(), {"walk_to": walk})


def test_nested_calls_pass_arguments_down_and_keep_the_depth_limit():
    library = {"s5": parse("skill s5 t\nuntil x == $t (max 12):\n  RIGHT", KEYS, parameters=True)}
    for level in range(4, -1, -1):
        library[f"s{level}"] = parse(f"skill s{level} t\ncall s{level + 1} $t", KEYS, parameters=True)
    env = LineWorld()
    go(parse("skill top\ncall s1 2", KEYS, parameters=True), env, library)  # s1 ... s5 is five nested calls
    assert env.get_progress()["x"] == 2
    with pytest.raises(SkillError, match="calls nested deeper than 5"):
        go(parse("skill top\ncall s0 2", KEYS, parameters=True), LineWorld(), library)


def test_a_parameterized_skill_still_cannot_call_itself():
    with pytest.raises(SkillError, match="calls itself"):
        parse("skill walk t\nRIGHT\ncall walk $t", KEYS, parameters=True)


def test_a_place_parameter_drives_go_to_in_the_tools_interface():
    program = parse("skill leave door\ngo_to $door", GRID_KEYS, tools=True, parameters=True)
    env = grid()
    counted = go(program, env, args=("door_to_town",), keys=GRID_KEYS)
    assert env.get_progress()["map"] == "town" and counted.notes == ["go_to door_to_town: went through to town"]
    assert counted.presses == 5  # every press the navigator made is counted


def test_a_position_parameter_and_a_bad_value_for_go_to():
    program = parse("skill step_to spot\ngo_to $spot", GRID_KEYS, tools=True, parameters=True)
    env = grid()
    go(program, env, args=([2, 2],), keys=GRID_KEYS)
    assert env.get_progress()["position"] == [2, 2]
    with pytest.raises(SkillError, match=r"`go_to \$spot` needs \$spot to be a place id or \[x, y\], not 7"):
        go(program, grid(), args=(7,), keys=GRID_KEYS)


def test_a_parameter_in_until_compares_strings_and_lists():
    program = parse("skill s door home\ngo_to $door\nuntil map != $home (max 2):\n  A", GRID_KEYS, tools=True, parameters=True)
    env = grid()
    go(program, env, args=("door_to_town", "house"), keys=GRID_KEYS)
    assert env.get_progress()["map"] == "town"


# Blocks that define several skills


def test_the_last_skill_runs_and_the_earlier_ones_are_definitions():
    programs = parse_all(f"{WALK_TO}\n\nskill walk_five\ncall walk_to 5", KEYS, parameters=True)
    assert [p.name for p in programs] == ["walk_to", "walk_five"] and programs[0].params == ("target",)


def test_a_block_whose_last_skill_takes_parameters_says_how_to_fix_it():
    with pytest.raises(SkillError, match=r"the last skill in the block runs with no arguments, so `walk_to` cannot declare parameters \(target\); end the block"):
        parse_all(WALK_TO, KEYS, parameters=True)


def test_block_errors_name_the_line_in_the_whole_block_and_duplicates_are_refused():
    with pytest.raises(SkillError, match="line 7: `JUMP` is not a legal action"):
        parse_all(f"{WALK_TO}\n\nskill main\ncall walk_to 5\nJUMP", KEYS, parameters=True)
    with pytest.raises(SkillError, match="defines `walk_to` twice"):
        parse_all(f"{WALK_TO}\n{WALK_TO}\nskill m\ncall walk_to 1", KEYS, parameters=True)
    with pytest.raises(SkillError, match="must start with `skill <name>`"):
        parse_all("RIGHT\nskill m\nA", KEYS, parameters=True)


def test_with_the_switch_off_a_block_is_one_program_as_before():
    with pytest.raises(SkillError, match="cannot parse `skill b`"):
        parse_all("skill a\nA\nskill b\nA", KEYS)
    assert [p.name for p in parse_all("skill a\nA", KEYS)] == ["a"]


# The switch off


@pytest.mark.parametrize(
    "source",
    [
        "skill walk_to target\nRIGHT",
        "skill s\nuntil x == $target (max 3):\n  RIGHT",
        "skill s\ncall walk_to 5",
    ],
)
def test_parameter_syntax_is_rejected_with_the_switch_off(source):
    with pytest.raises(SkillError, match=r"line \d: parameters are not enabled for this player \(.*\); write the steps for one fixed target"):
        parse(source, KEYS)


def test_go_to_with_a_parameter_is_rejected_off_and_in_buttons_mode():
    source = "skill leave door\ngo_to $door"
    with pytest.raises(SkillError, match="parameters are not enabled"):
        parse(source, GRID_KEYS, tools=True)
    with pytest.raises(SkillError, match="`go_to` is only available with the tools interface"):
        parse(source, GRID_KEYS, parameters=True)


def test_a_dollar_sign_inside_a_json_string_still_parses_with_the_switch_off():
    assert parse('skill s\nuntil name == "a$b" (max 2):\n  A', KEYS).body[0].value == "a$b"


# Voyager


WALK_THEN_USE = (
    "Explain: define and use.\nPlan:\n1) walk\nCode:\n```skill\n"
    "skill walk_to target\nuntil x == $target (max 12):\n  RIGHT\n\n"
    "skill {main}\ncall walk_to {arg}\n{tail}```"
)
CRITIC_OK = '{"reasoning": "done", "success": true, "critique": ""}'
CRITIC_NO = '{"reasoning": "no", "success": false, "critique": "Not there."}'


def action_prompts(model):
    return [c.messages[1]["content"] for c in model.calls if c.messages[0]["content"].startswith("You are a helpful assistant that writes skill")]


def system_prompts(model):
    return [c.messages[0]["content"] for c in model.calls if c.messages[0]["content"].startswith("You are a helpful assistant that writes skill")]


def scripted_voyager(tasks, programs, critic=lambda n: CRITIC_OK):
    """Task `n` gets `tasks[n]`, an action reply from `programs[n]`, and the verdict `critic(n)`."""
    state = {"task": -1}

    def reply(messages):
        system = messages[0].content
        if system.startswith(prompts.CURRICULUM[:40]):
            state["task"] += 1
            return f"Reasoning: r.\nTask: {tasks[min(state['task'], len(tasks) - 1)]}"
        if system == prompts.CRITIC:
            return critic(state["task"])
        if system == prompts.SKILL_DESCRIPTION:
            return "Walks right."
        # past the scripted tasks the program is invalid, so extra tasks end without touching the library
        return programs[state["task"]] if state["task"] < len(programs) else "Code:\n```skill\nskill filler\nJUMP\n```"

    return ScriptedModel(reply)


def play(player, env, steps=80, tasks=1):
    """Decide and step until `tasks` tasks have ended (completed or given up) or `steps` presses were made."""
    player.start_trial("goal text")
    obs = env.reset()
    events = []
    for _ in range(steps):
        decision = player.decide(obs, env.legal_actions())
        events += (decision.trace or {}).get("events", [])
        if decision.action is None or len([e for e in events if e["type"] in ("task_completed", "task_failed")]) >= tasks:
            break
        obs = env.step(decision.action).observation
    return events


def kinds(events, kind):
    return [e for e in events if e["type"] == kind]


def three_uses(first, second, third):
    return [
        WALK_THEN_USE.format(main="walk_five", arg=first, tail=""),
        f"Code:\n```skill\nskill walk_again\ncall walk_to {second}\n```",
        f"Code:\n```skill\nskill walk_third\ncall walk_to {third}\n```",
    ]


def test_a_new_skill_is_defined_and_used_in_one_reply_and_enters_as_a_candidate():
    model = scripted_voyager(["Walk to 5"], [WALK_THEN_USE.format(main="walk_five", arg=5, tail="")])
    player = Voyager(model, skill_parameters=True, test_skills=True)
    events = play(player, LineWorld(), steps=8)
    added = {e["name"]: e for e in kinds(events, "skill_added")}
    assert list(added) == ["walk_to", "walk_five"]
    assert added["walk_to"]["params"] == ["target"] and added["walk_to"]["status"] == "candidate"
    assert added["walk_five"]["status"] == "verified"  # no parameters, nothing to test
    skill = player.library.skills["walk_to"]
    assert (skill.status, skill.uses, skill.successes, skill.proven_with) == ("candidate", 1, 1, ("[5]",))
    assert kinds(events, "program")[0]["definitions"].keys() == {"walk_to"}
    assert len([c for c in model.calls if c.messages[0]["content"] == prompts.SKILL_DESCRIPTION]) == 2


def test_a_candidate_is_promoted_by_a_success_with_different_arguments():
    model = scripted_voyager(["Walk to 5", "Walk to 7", "Walk to 7 again"], three_uses(5, 7, 7))
    player = Voyager(model, skill_parameters=True, test_skills=True)
    events = play(player, LineWorld(), steps=60, tasks=3)
    used = [(e["name"], e["args"], e["status"], e["uses"], e["successes"]) for e in kinds(events, "skill_used")]
    assert used == [("walk_to", [7], "verified", 2, 2), ("walk_to", [7], "verified", 3, 3)]
    assert [(e["name"], e["args"]) for e in kinds(events, "skill_promoted")] == [("walk_to", [7])]
    assert player.library.skills["walk_to"].proven_with == ("[5]", "[7]")
    status = player.end_trial(TrialOutcome(False, "max_steps", 60)).trace["skill_status"]
    assert status["walk_to"] == {"status": "verified", "uses": 3, "successes": 3, "proven_with": ["[5]", "[7]"]}


def test_a_success_with_the_same_arguments_does_not_promote():
    model = scripted_voyager(["Walk to 5", "Walk to 5 again"], three_uses(5, 5, 5)[:2])
    player = Voyager(model, skill_parameters=True, test_skills=True)
    events = play(player, LineWorld(), steps=40, tasks=2)
    assert not kinds(events, "skill_promoted")
    skill = player.library.skills["walk_to"]
    assert (skill.status, skill.uses, skill.successes) == ("candidate", 2, 2)


def test_a_failed_use_counts_but_proves_nothing():
    model = scripted_voyager(["Walk to 5", "Walk to 7"], three_uses(5, 7, 7)[:2], critic=lambda n: CRITIC_OK if n == 0 else CRITIC_NO)
    player = Voyager(model, skill_parameters=True, test_skills=True, max_attempts=1)
    events = play(player, LineWorld(), steps=40, tasks=2)
    skill = player.library.skills["walk_to"]
    assert (skill.status, skill.uses, skill.successes, skill.proven_with) == ("candidate", 2, 1, ("[5]",))
    assert [e["success"] for e in kinds(events, "skill_used")] == [False]


def test_repeating_a_stored_definition_is_a_use_not_a_new_version():
    same = WALK_THEN_USE.format(main="walk_seven", arg=7, tail="")
    model = scripted_voyager(["Walk to 5", "Walk to 7"], [WALK_THEN_USE.format(main="walk_five", arg=5, tail=""), same])
    player = Voyager(model, skill_parameters=True, test_skills=True)
    events = play(player, LineWorld(), steps=40, tasks=2)
    skill = player.library.skills["walk_to"]
    assert (skill.version, skill.status, skill.uses, skill.proven_with) == (1, "verified", 2, ("[5]", "[7]"))
    assert [e["name"] for e in kinds(events, "skill_added")] == ["walk_to", "walk_five", "walk_seven"]
    assert [e["name"] for e in kinds(events, "skill_promoted")] == ["walk_to"]


def test_a_changed_definition_is_a_new_untested_version():
    changed = WALK_THEN_USE.format(main="walk_seven", arg=7, tail="").replace("until x == $target (max 12)", "until x >= $target (max 12)")
    model = scripted_voyager(["Walk to 5", "Walk to 7"], [WALK_THEN_USE.format(main="walk_five", arg=5, tail=""), changed])
    player = Voyager(model, skill_parameters=True, test_skills=True)
    play(player, LineWorld(), steps=40, tasks=2)
    skill = player.library.skills["walk_to"]
    assert (skill.version, skill.status, skill.uses, skill.proven_with) == (2, "candidate", 1, ("[7]",))


def test_a_program_that_errors_is_no_proof_even_if_the_critic_accepts_it():
    bad = "Code:\n```skill\nskill walk_wrong\ncall walk_to 7 8\n```"
    model = scripted_voyager(["Walk to 5", "Walk to 7"], [three_uses(5, 7, 7)[0], bad])
    player = Voyager(model, skill_parameters=True, test_skills=True, max_attempts=1)
    play(player, LineWorld(), steps=40, tasks=2)
    skill = player.library.skills["walk_to"]
    assert (skill.status, skill.uses) == ("candidate", 1)  # the call never bound, so it never ran


def test_retrieval_labels_candidates_and_the_label_goes_when_verified():
    model = scripted_voyager(["Walk to 5", "Walk to 7", "Walk to 8"], three_uses(5, 7, 8))
    player = Voyager(model, skill_parameters=True, test_skills=True)
    play(player, LineWorld(), steps=60, tasks=3)
    prompts_seen = system_prompts(model)
    assert "# candidate: succeeded 1 of 1 uses, only with arguments [5]; not yet verified\nskill walk_to target" in prompts_seen[1]
    assert "# candidate:" not in prompts_seen[2] and "skill walk_to target" in prompts_seen[2]  # verified by the second task


def test_retrieval_prefers_a_verified_skill_but_a_much_better_candidate_still_shows():
    library = SkillLibrary()
    walk_to = parse("skill walk_to t\nuntil x == $t (max 3):\n  RIGHT", KEYS, parameters=True)
    library.add(parse("skill walk_right\nRIGHT", KEYS), "Walks right along the corridor.", "walk right")
    library.add(walk_to, "Walks right along the corridor.", "walk right", status="candidate", proven_with=[[1]])
    assert [s.program.name for s in library.retrieve("walk right corridor", k=2)] == ["walk_right", "walk_to"]  # same score: verified first
    library.add(parse("skill press_a\nA", KEYS), "Presses A once.", "interact")
    library.add(parse("skill walk_left\nLEFT", KEYS), "Walks left.", "walk left")
    library.skills["walk_to"] = replace(library.skills["walk_to"], description="Walks to any tile of the corridor.")
    assert library.retrieve("walk to any tile of the corridor", k=1)[0].program.name == "walk_to"  # relevance still wins


def test_the_library_records_uses_and_persists_status(tmp_path):
    path = tmp_path / "skills.json"
    library = SkillLibrary(path)
    library.add(parse("skill plain\nA", KEYS), "Presses A.", "press")
    assert set(json.loads(path.read_text())["plain"]) == {"source", "description", "task", "version"}  # the file format is unchanged
    library.add(parse(WALK_TO, KEYS, parameters=True), "Walks right.", "walk", status="candidate", proven_with=[[5], [5]])
    skill, promoted = library.record_use("walk_to", [5], True)
    assert not promoted and (skill.status, skill.uses, skill.successes) == ("candidate", 2, 2)
    skill, promoted = library.record_use("walk_to", [9], True)
    assert promoted and skill.status == "verified" and skill.proven_with == ("[5]", "[9]")
    reloaded = SkillLibrary(path)
    reloaded.bind(KEYS, parameters=True)
    loaded = reloaded.skills["walk_to"]
    assert (loaded.status, loaded.uses, loaded.successes, loaded.program.params) == ("verified", 3, 3, ("target",))
    plain = SkillLibrary(path)
    plain.bind(KEYS)  # a player without the switch cannot parse a parameterized skill, so it drops it
    assert set(plain.skills) == {"plain"}


def test_an_unused_parameterized_definition_is_not_stored():
    unused = "Code:\n```skill\nskill walk_to target\nuntil x == $target (max 12):\n  RIGHT\n\nskill just_press\nA\n```"
    model = scripted_voyager(["Press"], [unused])
    player = Voyager(model, skill_parameters=True, test_skills=True)
    events = play(player, LineWorld(), steps=4)
    assert [e["name"] for e in kinds(events, "skill_untested")] == ["walk_to"]
    assert sorted(player.library.skills) == ["just_press"]


def test_parameters_without_testing_store_the_skill_as_verified():
    model = scripted_voyager(["Walk to 5"], [WALK_THEN_USE.format(main="walk_five", arg=5, tail="")])
    player = Voyager(model, skill_parameters=True)
    events = play(player, LineWorld(), steps=8)
    assert {e["name"]: e["status"] for e in kinds(events, "skill_added")} == {"walk_to": "verified", "walk_five": "verified"}
    assert player.library.skills["walk_to"].uses == 0 and "skill_status" not in player.end_trial(TrialOutcome(False, "x", 1)).trace


def test_a_failed_block_is_fed_back_whole_with_its_definitions():
    block = WALK_THEN_USE.format(main="walk_five", arg="5 6", tail="")
    verdicts = iter([CRITIC_NO, CRITIC_OK])
    model = scripted_voyager(["Walk to 5"], [block, WALK_THEN_USE.format(main="walk_five", arg=5, tail="")], critic=lambda n: next(verdicts))
    player = Voyager(model, skill_parameters=True, test_skills=True)
    play(player, LineWorld(), steps=20)
    second = action_prompts(model)[1]
    assert "Execution error: `walk_to` takes 1 argument (target) but got 2" in second
    assert "skill walk_to target\nuntil x == $target (max 12):\n  RIGHT\n\nskill walk_five\ncall walk_to 5 6" in second


def test_a_parameter_block_is_rejected_with_the_switch_off_and_the_reason_is_fed_back():
    model = scripted_voyager(["Walk to 5"], [WALK_THEN_USE.format(main="walk_five", arg=5, tail="")])
    player = Voyager(model)
    events = play(player, LineWorld(), steps=8)
    assert "parameters are not enabled for this player" in kinds(events, "program")[0]["error"]
    assert not player.library.skills


def test_the_prompt_teaches_parameters_only_with_the_switch():
    off = Voyager(scripted_voyager(["t"], ["x"]))
    on = Voyager(scripted_voyager(["t"], ["x"]), skill_parameters=True)
    testing = Voyager(scripted_voyager(["t"], ["x"]), skill_parameters=True, test_skills=True)
    for player in (off, on, testing):
        play(player, LineWorld(), steps=2)
    first = [system_prompts(p.model)[0] for p in (off, on, testing)]
    assert "Skills can take parameters" not in first[0] and "candidate" not in first[0]
    assert "Skills can take parameters" in first[1] and "candidate" not in first[1] and "One code block may hold several skills" in first[1]
    assert "Skills can take parameters" in first[2] and "marked `# candidate`" in first[2]
    assert first[1] == prompts.ACTION_TEMPLATE.format(language=prompts.skill_language(tools=False, parameters=True).replace("<ACTION>", "LEFT|RIGHT|A"), programs="(none yet)", response_format=prompts.ACTION_RESPONSE_FORMAT)


def test_the_tools_language_adds_go_to_with_a_parameter():
    assert "go_to $<param>" in prompts.skill_language(tools=True, parameters=True)
    assert "go_to $<param>" not in prompts.skill_language(tools=False, parameters=True)
    assert prompts.skill_language(tools=True) == prompts.SKILL_LANGUAGE_TOOLS and prompts.skill_language(tools=False) == prompts.SKILL_LANGUAGE


def test_a_parameterized_skill_walks_a_door_in_the_tools_interface():
    code = "Code:\n```skill\nskill leave door\ngo_to $door\n\nskill leave_house\ncall leave door_to_town\n```"
    model = scripted_voyager(["Leave the house"], [code])
    player = Voyager(model, action_interface="tools", skill_parameters=True, test_skills=True)
    events = play(player, grid(), steps=20)
    assert player.library.skills["leave"].status == "candidate" and player.library.skills["leave"].proven_with == ('["door_to_town"]',)
    assert any(e["type"] == "task_completed" for e in events)


def test_test_skills_needs_parameters_and_the_library():
    model = scripted_voyager(["t"], ["x"])
    with pytest.raises(ValueError, match="test_skills only works with skill_parameters"):
        Voyager(model, test_skills=True)
    with pytest.raises(ValueError, match="test_skills only works with skill_parameters"):
        Voyager(model, skill_parameters=True, skill_library=False, test_skills=True)
    mechanisms = Voyager(model, skill_parameters=True, test_skills=True).mechanisms()
    assert mechanisms["skill_parameters"] is True and mechanisms["test_skills"] is True
    off = Voyager(model).mechanisms()
    assert off["skill_parameters"] is False and off["test_skills"] is False and all(isinstance(v, bool) for v in off.values())


# The switches off leave prompts and decisions as they were


def _digest(model, actions, review):
    payload = json.dumps({"actions": actions, "calls": [[c.messages, c.output] for c in model.calls], "review": review}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _drive(player, env, steps, model):
    player.start_trial("goal text")
    obs = env.reset()
    actions = []
    for _ in range(steps):
        d = player.decide(obs, env.legal_actions())
        actions.append(d.action.key if d.action else None)
        if d.action:
            obs = env.step(d.action).observation
    review = player.end_trial(TrialOutcome(False, "max_steps", steps))
    return _digest(model, actions, review.trace)


# Taken from the commit before these switches existed. The buttons-mode Voyager digest is in
# tests/test_action_interface.py; these are the tools-mode Voyager and the Continual Harness.
PARAMETER_OFF_DIGESTS = {"voyager_tools_grid": "31bd91733e645c98", "continual_line": "862bb390a3e5213a"}


def test_prompts_and_decisions_are_unchanged_with_the_switches_off():

    def reply(messages):
        system = messages[0].content
        if system.startswith(prompts.CURRICULUM[:40]):
            return "Reasoning: r.\nTask: Reach the milestone"
        if system == prompts.CRITIC:
            return CRITIC_OK
        if system == prompts.SKILL_DESCRIPTION:
            return "Goes to the milestone."
        return "Plan:\n1) go\nCode:\n```skill\nskill to_milestone\ngo_to door_to_town\ngo_to milestone\n```"

    model = ScriptedModel(reply)
    assert _drive(Voyager(model, action_interface="tools", specialists=True), grid(), 40, model) == PARAMETER_OFF_DIGESTS["voyager_tools_grid"]

    walk = "skill walk_right\nuntil x == 9 (max 12):\n  RIGHT"

    def reply(messages):
        if messages[0].role == "system":
            tool = {"tool": "run_skill", "args": {"name": "walk_right"}} if "- walk_right (effectiveness" in messages[1].content else {"tool": "press", "args": {"key": "LEFT"}}
            return json.dumps({"why": "b", **tool})
        text = messages[0].content
        if text.startswith("You are a prompt optimization expert"):
            return "Walk right until the end."
        if text.startswith("You are a harness evolution system"):
            return json.dumps({"analysis": "x", "add": [{"name": "walk_right", "description": "Walks right.", "code": walk, "effectiveness": "high", "importance": 4}]})
        return '{"analysis": "none"}'

    model = ScriptedModel(reply)
    assert _drive(ContinualHarness(model, refine_warmup=2, refine_early_every=2), LineWorld(), 30, model) == PARAMETER_OFF_DIGESTS["continual_line"]


# Continual Harness


def call(tool, **args):
    return json.dumps({"why": "because", "tool": tool, "args": args})


def continual(agent, *, skills='{"analysis": "none"}', **kwargs):
    def reply(messages):
        if messages[0].role == "system":
            return agent(messages)
        text = messages[0].content
        if text.startswith("You are a harness evolution system"):
            return skills
        return "Walk right." if text.startswith("You are a prompt optimization expert") else '{"analysis": "none"}'

    return ContinualHarness(ScriptedModel(reply), **kwargs)


def add_params_skill(player, code=WALK_TO):
    player.harness.bind(KEYS, parameters=True)
    player.harness.add_skill(parse(code, KEYS, parameters=True), "Walks right to a place.", "high", 4, "refiner")


def drive_continual(player, env, presses):
    player.start_trial("Walk to the far right.")
    obs = env.reset()
    decisions = []
    while sum(d.action is not None for d in decisions) < presses:
        decision = player.decide(obs, env.legal_actions())
        decisions.append(decision)
        if decision.action is not None:
            obs = env.step(decision.action).observation
    return decisions


def test_run_skill_passes_arguments_and_counts_every_press():
    player = continual(lambda m: call("run_skill", name="walk_to", args=[4]), skill_parameters=True, carry_across_trials=True)
    add_params_skill(player)
    env = LineWorld()
    decisions = drive_continual(player, env, 4)
    assert [d.action.key for d in decisions] == ["RIGHT"] * 4 and env.get_progress()["x"] == 4
    assert decisions[0].trace["skill"] == "walk_to" and player.harness.meta["walk_to"].uses == 1
    plan = [e for e in decisions[0].trace["events"] if e["type"] == "plan"][0]
    assert plan["args"] == {"name": "walk_to", "args": [4]}


def test_run_skill_with_the_wrong_argument_count_is_rejected_with_the_reason():
    replies = iter([call("run_skill", name="walk_to"), call("run_skill", name="walk_to", args=[1, 2]), call("press", key="A")])
    player = continual(lambda m: next(replies), skill_parameters=True, carry_across_trials=True)
    add_params_skill(player)
    drive_continual(player, LineWorld(), 1)
    asked = [c.messages[1]["content"] for c in player.model.calls if c.messages[0]["role"] == "system"]
    assert "run_skill: skill walk_to takes 1 argument (target) but got 0" in asked[1]
    assert "run_skill: skill walk_to takes 1 argument (target) but got 2" in asked[2]


def test_run_skill_args_must_be_a_list_and_a_plain_skill_takes_none():
    player = continual(lambda m: "", skill_parameters=True, carry_across_trials=True)
    add_params_skill(player, "skill plain\nRIGHT")
    player._keys = KEYS
    from agents.continual.validate import Invalid

    with pytest.raises(Invalid, match='"args" must be a list'):
        player._check_run_skill({"name": "plain", "args": 3})
    with pytest.raises(Invalid, match="takes no arguments but got 1"):
        player._check_run_skill({"name": "plain", "args": [1]})
    assert player._check_run_skill({"name": "plain"}) == {"name": "plain", "args": []}


def test_run_skill_takes_no_args_with_the_switch_off():
    player = continual(lambda m: "", carry_across_trials=True)
    player.harness.bind(KEYS)
    player.harness.add_skill(parse("skill plain\nRIGHT", KEYS), "d", "high", 4, "refiner")
    from agents.continual.validate import Invalid

    with pytest.raises(Invalid, match=r"unknown fields \['args'\]"):
        player._check_run_skill({"name": "plain", "args": []})


def test_the_refiner_writes_a_parameterized_skill_only_with_the_switch():
    reply = json.dumps({"add": [{"name": "walk_to", "description": "Walks right to x.", "code": WALK_TO}]})
    for on in (True, False):
        player = continual(lambda m: "", skills=reply, skill_parameters=on)
        harness = HarnessState()
        skills = player.refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1).passes["skills"]
        if on:
            assert not skills.rejected and harness.library.skills["walk_to"].program.params == ("target",)
            assert "walk_to(target)" in harness.skill_text()
        else:
            assert not harness.library.skills and "parameters are not enabled" in skills.rejected[0]["reason"]


def test_the_refiner_prompt_and_the_agent_prompt_teach_parameters_only_with_the_switch():
    texts = {}
    for on in (False, True):
        player = continual(lambda m: call("press", key="LEFT"), skill_parameters=on, refine_warmup=2, refine_early_every=2)
        drive_continual(player, LineWorld(), 3)
        calls = player.model.calls
        texts[on] = (
            calls[0].messages[0]["content"],
            next(c.messages[0]["content"] for c in calls if c.messages[0]["content"].startswith("You are a harness evolution system")),
        )
    assert '"args": [<value>, ...]' not in texts[False][0] and "Skills can take parameters" not in texts[False][1]
    assert '**run_skill** args: {"name": "<skill name>", "args": [<value>, ...]}' in texts[True][0]
    assert "Skills can take parameters" in texts[True][1] and "Parameters are on." in texts[True][1]


def test_a_refined_parameterized_skill_is_used_with_arguments(tmp_path):
    reply = json.dumps({"add": [{"name": "walk_to", "description": "Walks right to x.", "code": WALK_TO}]})

    def agent(messages):
        if "- walk_to(target) (effectiveness" in messages[1].content:
            return call("run_skill", name="walk_to", args=[9])
        return call("press", key="LEFT")

    player = continual(agent, skills=reply, skill_parameters=True, refine_warmup=2, refine_early_every=2)
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=player, goal=Goal("reach", "Walk to the far right.", {"x": 9}), budgets=Budgets(max_steps=30))
    assert results[0].success
    summary = json.loads((tmp_path / "run/trials/000/steps.jsonl").read_text().splitlines()[-1])["review"]["trace"]["harness"]["skills"]["walk_to"]
    assert summary["uses"] == 1 and "$target" in summary["source"]


# Config


def test_make_player_wires_both_switches():
    model = ScriptedModel(lambda m: "")
    voyager = make_player({"name": "voyager", "skill_parameters": True, "test_skills": True}, model)
    assert (voyager.skill_parameters, voyager.test_skills) == (True, True)
    voyager = make_player({"name": "voyager", "skill_parameters": True}, model)
    assert (voyager.skill_parameters, voyager.test_skills) == (True, False)
    plain = make_player({"name": "voyager"}, model)
    assert (plain.skill_parameters, plain.test_skills) == (False, False)
    assert make_player({"name": "continual", "skill_parameters": True}, model).mechanisms()["skill_parameters"] is True
    assert make_player({"name": "continual"}, model).mechanisms()["skill_parameters"] is False
    assert make_player({"name": "continual", "skill_parameters": True}, model).refiner.skill_parameters is True


def test_the_parameter_configs_differ_from_their_base_only_by_the_switches():
    for env in ("gridworld", "firered"):
        spec = json.loads((ROOT / f"experiments/{env}_7b_voyager_tools_params.json").read_text())
        base = json.loads((ROOT / f"experiments/{env}_7b_voyager_tools.json").read_text())
        assert spec["agent"] == {**base["agent"], "skill_parameters": True, "test_skills": True}
        assert {k: v for k, v in spec.items() if k != "agent"} == {k: v for k, v in base.items() if k != "agent"}
        assert make_player(spec["agent"], ScriptedModel(lambda m: "")).mechanisms()["test_skills"] is True
    spec = json.loads((ROOT / "experiments/gridworld_7b_continual_params.json").read_text())
    base = json.loads((ROOT / "experiments/gridworld_7b_continual.json").read_text())
    assert spec["agent"] == {**base["agent"], "skill_parameters": True}
    assert {k: v for k, v in spec.items() if k != "agent"} == {k: v for k, v in base.items() if k != "agent"}
