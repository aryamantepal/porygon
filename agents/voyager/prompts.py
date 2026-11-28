"""Voyager's prompts, ported from github.com/MineDojo/Voyager (MIT), voyager/prompts/.

Each keeps the original's structure and wording where it is not about
Minecraft. Changes:
- "Minecraft" becomes "a video game"; Mineflayer JavaScript becomes the skill
  language in `skills.py`.
- Minecraft-specific fields (biome, inventory, chests, hunger) become the
  generic snapshot, and Minecraft few-shot examples are removed.
- The curriculum gets the trial's goal. Voyager's curriculum is open-ended
  ("discover as many diverse things as possible"); here it proposes the next
  sub-task toward a goal the runner set, because every architecture is
  compared on the same goal.
"""

SKILL_LANGUAGE = """\
A skill program is a list of statements, one per line, with blocks indented under a header:
    skill <name>                      first line: a meaningful snake_case name
    <ACTION> x<n>                     press a legal action n times (x1 if omitted)
    repeat <n>:                       repeat the indented block n times
    until <field> <op> <json> (max <n>):   repeat the indented block until the condition holds, at most n times
    call <skill_name>                 run a skill from the library
Conditions read fields of the State (dotted paths such as party.0.hp) and compare with ==, !=, <, <=, >, >= against JSON values such as 10 or [4, 0]."""

# With the tools interface the language gains three statements, listed only then.
_CALL_LINE = "    call <skill_name>                 run a skill from the library\n"
_TOOL_LINES = """\
    go_to <place_id>                  walk to a place listed in the State's `places` (walking onto a door goes through it)
    go_to [x, y]                      walk to a tile of the map
    talk_to <place_id>                walk next to a person, face them, and talk
go_to and talk_to expand into button presses, and each press counts toward the program's limit of 200. They fail when the place does not exist or there is no path, and stop early if the game leaves the overworld (a battle, a menu, a conversation).
"""
assert _CALL_LINE in SKILL_LANGUAGE
SKILL_LANGUAGE_TOOLS = SKILL_LANGUAGE.replace(_CALL_LINE, _CALL_LINE + _TOOL_LINES)

# With the skill_parameters switch the language gains parameters, listed only then.
SKILL_PARAMETERS = """\
Skills can take parameters, so one skill serves any target:
    skill <name> <param> ...          optional parameter names after the skill name, e.g. `skill walk_to target`
    until <field> <op> $<param> (max <n>):   a parameter can stand where a JSON value goes
    call <skill_name> <arg> ...       pass arguments by position; a skill that takes parameters needs exactly that many
An argument is a place id (door_0), a number, a JSON value (such as [4, 0]), or one of your own `$` parameters. There are no expressions."""
_PARAMETER_TOOL_LINE = "    go_to $<param>                    a parameter can stand where a place id goes (also talk_to)\n"
SKILL_PROGRAMS = """\
One code block may hold several skills, each starting with its own `skill` line. The last one runs, with no arguments, so it must not take parameters; it calls the earlier ones (and library skills) with arguments. Skills defined earlier in the block are saved to the library when the task succeeds."""
SKILL_TESTING = """\
A library skill marked `# candidate` has succeeded only with the arguments listed. Call it with different arguments when the task allows: once it succeeds with new arguments it becomes verified."""


def skill_parameters(*, tools: bool = False) -> str:
    """The parameter lines of the language; `tools` adds `go_to $<param>`."""
    marker = "    call <skill_name> <arg>"
    return SKILL_PARAMETERS.replace(marker, (_PARAMETER_TOOL_LINE if tools else "") + marker)


def skill_language(*, tools: bool, parameters: bool = False, testing: bool = False) -> str:
    """The language for the action prompt. With `parameters` and `testing` off it is `SKILL_LANGUAGE` or `SKILL_LANGUAGE_TOOLS` exactly."""
    text = SKILL_LANGUAGE_TOOLS if tools else SKILL_LANGUAGE
    if parameters:
        text += "\n" + skill_parameters(tools=tools) + "\n" + SKILL_PROGRAMS + ("\n" + SKILL_TESTING if testing else "")
    return text

ACTION_RESPONSE_FORMAT = """\
Explain: ...
Plan:
1) ...
2) ...
3) ...
...
Code:
```skill
skill your_skill_name
...
```"""

ACTION_TEMPLATE = """\
You are a helpful assistant that writes skill programs to complete any task specified by me in a video game.

{language}

Here are some useful programs from the skill library.

{programs}

At each round of conversation, I will give you
Code from the last round: ...
Execution error: ...
Environment feedback: ...
State: ...
Task: ...
Context: ...
Critique: ...

You should then respond to me with
Explain (if applicable): Are there any steps missing in your plan? Why does the code not complete the task? What do the execution error and environment feedback imply?
Plan: How to complete the task step by step. The task completeness check is based on the final state.
Code:
    1) Write one skill program in the language above.
    2) Reuse the above useful programs as much as possible with `call`.
    3) Your program will be reused for building more complex programs. Therefore, you should make it generic and reusable.
    4) Programs in the "Code from the last round" section are not saved. Do not call them.
    5) Do not write loops without a bound; every `until` needs a (max n).
    6) Name your skill in a meaningful way (can infer the task from the name).

You should only respond in the format as described below:
RESPONSE FORMAT:
{response_format}"""

EXAMPLE_PROGRAMS = """\
Here are example programs for a different task. They show the language only. They are not in the skill library, so do not `call` them.

{programs}"""

CRITIC = """\
You are an assistant that assesses my progress of playing a video game and provides useful guidance.

You are required to evaluate if I have met the task requirements. Exceeding the task requirements is also considered a success while failing to meet them requires you to provide critique to help me improve.

I will give you the following information:

State before: The game state before the task execution.
State after: The game state after the task execution.
Screen: The screen after the task execution, if attached.
Task: The objective I need to accomplish.
Context: The context of the task.

You should only respond in JSON format as described below:
{
    "reasoning": "reasoning",
    "success": boolean,
    "critique": "critique"
}
Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc."""

CURRICULUM = """\
You are a helpful assistant that tells me the next immediate task to do in a video game. My ultimate goal is: {goal}

I will give you the following information:
State: ...
Completed tasks so far: ...
Failed tasks that are too hard: ...

You must follow the following criteria:
1) You should act as a mentor and guide me to the next task based on my current learning progress.
2) Please be very specific about what I need to do and where.
3) The next task should follow a concise format and be a single phrase. Do not propose multiple tasks at the same time. Do not mention anything else.
4) The next task should not be too hard since I may not have learned enough skills to complete it yet.
5) The next task should move me toward my ultimate goal. I should not be doing the same thing over and over again.
6) I may sometimes need to repeat some tasks. Only repeat tasks if necessary.
7) Tasks that cannot be verified from the state and screen I report should be avoided.

You should only respond in the format as described below:
RESPONSE FORMAT:
Reasoning: Based on the information I listed above, do reasoning about what the next task should be.
Task: The next task."""

SKILL_DESCRIPTION = """\
You are a helpful assistant that writes a description of the given skill program.

1) Do not mention the skill name.
2) Try to summarize the skill in no more than 6 sentences.
3) Your response should be a single line of text."""
