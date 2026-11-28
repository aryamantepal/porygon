"""Continual Harness prompts, ported from github.com/sethkarten/continual-harness (MIT).

Sources: the seed strategic guidance and the system prompt under
agents/prompts/ (agent side), and the prompts inside agents/utils/harness_evolver.py
and agents/utils/prompt_optimizer.py (Refiner side). Where a passage is kept it
keeps the original's wording. Changes:
- The game is "a video game". Game-specific tools, button names, map fields, and
  the Python skill API are removed. Skills are programs in the skill language of
  `agents/voyager/skills.py`.
- The agent's tools are the four in `TOOLS`. The original's objective, sub-agent,
  trajectory-history, and code tools are not built.
- The prompt pass rewrites a bounded "learned guidance" section, not the whole
  base prompt, so the fixed instructions cannot be edited away.
"""

# Agent side

_BASE_HEAD = """\
# Strategic Guidance (Seed)

You are playing a video game with no walkthrough or wiki. Learn game mechanics through observation{store}.

## Decision Framework

**Every step, assess your context and act accordingly:**

- **Read the state carefully.** Understand what mode you are in (exploration, combat, dialogue, menu) before acting.
- **In dialogue**: Advance with the confirm button.{dialogue}
- **In exploration**: Move toward your goal.{explore}
- **In combat/battle**: Observe the interface and learn the mechanics.
- **In menus**: Navigate carefully with the directional, confirm, and cancel buttons.
- **Stuck for 3+ steps**: Do something different from what you have been doing{stuck}."""

_BASE_TOOLKIT = """\
## Building Your Toolkit

You may start with no skills. Build them:

- **If you are repeatedly pressing the same buttons**: that sequence should become a skill. Call `refine` so the Refiner can write it from your recent steps.
- **If a skill fails**: its error is shown in the next step's results. `refine` repairs it."""

_BASE_MEMORY = "- **Memory**: Store game mechanics, places, and lessons with `write_memory`. Give each entry an importance from 1 to 5."
_BASE_SKILLS = "- **Skills**: Saved programs that run press by press with `run_skill`."
_BASE_LEARNED = """\
---

*The learned guidance below is updated over time by the Refiner.*"""


def base(names, *, learned: bool) -> str:
    """The seed guidance, mentioning only the tools this run has."""
    memory, skills, refine = "write_memory" in names, "run_skill" in names, "refine" in names
    text = _BASE_HEAD.format(
        store=" and store them in memory" if memory else "",
        dialogue=" Store important information from dialogue in memory." if memory else "",
        explore=" If you are repeating the same movement, build a skill for it (see \"Building Your Toolkit\")." if refine else "",
        stuck=", or call `refine` to have your recent steps reviewed" if refine else "",
    )
    parts = [text]
    if refine:
        parts.append(_BASE_TOOLKIT)
    knowledge = [line for line, on in ((_BASE_MEMORY, memory), (_BASE_SKILLS, skills)) if on]
    if knowledge:
        parts.append("## Knowledge Management\n\n" + "\n".join(knowledge))
    if learned:
        parts.append(_BASE_LEARNED)
    return "\n\n".join(parts)


_SYSTEM = """\
# Game Agent: Continual Harness Scaffold

You are playing a video game. You receive the current state (a screenshot, text, or both, as this run allows), your goal, and a short history. You start with NO walkthrough and NO wiki. You navigate with `press` and build your own capabilities through play.

**You must discover how the game works through observation and experimentation.**{store}

Each turn, reply with exactly one tool call as one JSON object and nothing after it:
{{"why": "<one sentence>", "tool": "<tool name>", "args": {{...}}}}

## Callable tools (parameters)

{tools}

## Seeing results

On the **next** step, the harness shows RESULTS FROM PREVIOUS STEP with what a tool returned, including errors. Read that block before deciding.

## Bootstrap strategy

Build your skills and memory as you play:

{strategy}

## Constraints

{constraints}"""


def system(tools_text: str, names) -> str:
    memory, skills, refine = "write_memory" in names, "run_skill" in names, "refine" in names
    strategy = ["**Observe first**: look at the state to understand what you see and what the controls do."]
    if memory:
        strategy.append("**Store what you learn**: every new mechanic, place, or strategy you discover should go into memory with `write_memory`.")
    if refine:
        strategy.append("**Automate recurring actions**: when you find yourself repeating the same multi-step sequence, ask for a skill with `refine`.")
    if memory:
        strategy.append("**Record successful strategies**: after solving a challenge, make sure it is in memory.")
    ends = " or ".join(f"`{name}`" for name in ("press", "run_skill") if name in names)
    constraints = [
        "- **Valid keys only**: `press` takes only the keys listed in the tool description.",
        f"- **Every step must end** with {ends}.",
    ]
    free = [f"`{name}`" for name in ("write_memory", "refine") if name in names]
    if free:
        constraints.append(f"- {' and '.join(free)} take{'s' if len(free) == 1 else ''} no game time, and you choose again afterwards.")
    constraints.append("- **One JSON object per reply.** A reply that is not a valid tool call is rejected and you are asked again.")
    return _SYSTEM.format(
        store=" Store what you learn in memory." if memory else "",
        tools=tools_text,
        strategy="\n".join(f"{i}. {line}" for i, line in enumerate(strategy, start=1)),
        constraints="\n".join(constraints),
    )


TOOLS = {
    "press": '**press** args: {{"key": "<one of {keys}>", "times": n}}  ("times" is optional, 1 to {max_times}). Press a key.',
    "run_skill": '**run_skill** args: {{"name": "<skill name>"}}. Run a skill from the library; each press it makes is one step. Use it for any sequence you would otherwise press by hand.',
    "write_memory": '**write_memory** args: {{"title": "<short title>", "content": "<what you learned>", "importance": 1-5}}. Save a note. Writing an existing title replaces that entry.',
    "refine": '**refine** args: {{"reason": "<what is not working>", "num_steps": n}}  (both optional). Ask the Refiner to review your recent steps now and edit your guidance, skills, and memory. Takes no game time. Rate-limited; use it when a skill underperforms or you are stuck.',
}

# With the skill_parameters switch, run_skill takes arguments for a skill that declares parameters.
RUN_SKILL_PARAMETERS = '**run_skill** args: {{"name": "<skill name>", "args": [<value>, ...]}}. Run a skill from the library; each press it makes is one step. Use it for any sequence you would otherwise press by hand. A skill listed as `name(param, ...)` takes one argument per parameter, in order, each a JSON value such as 10, [4, 0], or "door_0"; "args" is omitted for a skill with no parameters.'

EXAMPLE_REPLIES = """\
Here are example replies, from a different goal than yours. They show the form, not the answer to your task.

{replies}"""

BAD_REPLY = (
    "Your last reply could not be used: {error}\n"
    "Your reply was: {reply}\n"
    'Reply with exactly one JSON object with "tool" and "args" inside it, and nothing after it.'
)

REPEAT_NOTE = (
    "You have made the same call ({call}) {count} times in a row and the state did not change. "
    "Do something different from what has not worked, or call `refine`."
)

REFINED = "HARNESS REFINEMENT completed. Changes to your toolkit:"

# Added to the skills pass with skill_parameters: parameters take values here (`$name` in an `until`, or in a `call`).
REFINER_PARAMETERS = """\
Parameters are on. A skill you write may take parameters and `run_skill` passes the agent's arguments to it. Here a parameter can stand only where a JSON value goes (`until x == $target (max 20):`) or in a `call`. List a skill's parameters in its header and write a parameterized skill when the same steps recur with a different target.
"""

# Refiner side

PROMPT_PASS = """\
You are a prompt optimization expert. Your task is to improve an AI agent's learned guidance based on its recent performance in a video game.

## Main Agent's Fixed Instructions (FIXED - Cannot Be Changed):
The agent receives these every step. You must **not** duplicate them into the guidance. They list the tools and the hard constraints. Strategy, goals, and playstyle lessons belong in the learned guidance you are editing.

{fixed}

---

## Goal
{goal}

## Current Learned Guidance:
This is the optimizable section. It is combined with runtime context (recent actions, state, memory, skills) and sent to the main agent. You can modify it to improve the agent's decision-making.

{guidance}

## Recent Agent Trajectories (Last {count} steps):
{window}

## Your Task:
Analyze the agent's recent performance and write an IMPROVED learned guidance section that:

1. **Addresses observed failures** - If the agent made mistakes, add specific guidance to prevent them
2. **Reinforces successful patterns** - If certain strategies worked well, emphasize them
3. **Adds learned lessons** - Include insights derived from trajectory analysis

## Analysis Guidelines:

**Look for these patterns:**
- Repeated failures (stuck in loops, wrong tool usage, blocked movement)
- Successful strategies (good memory usage, efficient movement, smart choices)
- Progress toward the goal
- Tool usage patterns (are they using the tools at their disposal effectively?)
- Memory management (are they storing and retrieving information in memory appropriately?)
- Not adapting when stuck -> emphasize flexibility and trying new approaches

## Output Format:
Provide the complete improved guidance as plain text, at most {cap} characters. Make targeted improvements, keeping the elements that are working while adding additional guidance if necessary. Do not write JSON or tool calls.

IMPROVED GUIDANCE:
"""

SKILLS_PASS = """\
You are a harness evolution system analyzing an AI agent's recent gameplay in a video game.

Your job: identify reusable behavioral patterns (skills) from successful actions and evaluate existing skills.

## Goal
{goal}

## Current Skill Library
{skills}

## Recent Trajectories (last {count} steps)
{window}

{failures}
{rejected}
## Skill Language (MUST follow this exactly)

{language}

Legal keys: {keys}. A skill runs one press per step. Conditions need the State, so they fail when the run shows no State.
{parameters}{examples}
## Analysis Tasks

1. **Identify underperforming skills from reasoning traces**: Review the agent's reasoning to find skills that failed, errored, or produced undesirable outcomes. If the agent complained about a skill or abandoned it, recommend a rewrite with full replacement `code`.

2. **Fix broken skills**: If skills error or use keys that are not legal, fix the code.

3. **Record new skills**: Look for successful action sequences. Include `code` for skills that automate multi-step actions. Make skills reusable, not tied to one moment.

4. **Update effectiveness ratings** based on trajectory outcomes.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary",
  "add": [
    {{
      "name": "snake_case_name",
      "description": "What the skill does",
      "code": "skill snake_case_name\\n<KEY> x2",
      "effectiveness": "low|medium|high",
      "importance": 3
    }}
  ],
  "update": [
    {{
      "id": "<skill name>",
      "effectiveness": "low|medium|high",
      "description": "optional updated description",
      "code": "optional: FULL replacement program, starting with `skill <same name>`"
    }}
  ]
}}

Make at most {max_edits} changes. Only include sections with actual recommendations. Empty arrays are fine.
"""

MEMORY_PASS = """\
You are a memory curator for an AI agent playing a video game.

Your job: review the agent's memory store and recent trajectories, then recommend targeted improvements. Be conservative — the agent itself writes memory during gameplay. You fill gaps and clean up.

## Goal
{goal}

## Current Memory Overview
{memory}

## Recent Trajectories (last {count} steps)
{window}

## Curation Tasks

1. **Fill knowledge gaps**: If the agent encountered key events or learned how something works but has no memory entry for it, recommend adding one.
2. **Update stale entries**: If memory content contradicts what trajectories show, recommend updates.
3. **Rebalance importance**: Entries about things the agent has moved past can have importance lowered (1-2). Knowledge about what the agent is doing now stays at 3-5.

Keep recommendations minimal (max 3-5 changes). Do NOT duplicate what the agent already stored.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary of memory state",
  "add": [
    {{
      "path": "category/subcategory",
      "title": "string",
      "content": "string",
      "importance": 3
    }}
  ],
  "update": [
    {{
      "id": "mem_N",
      "content": "optional updated content",
      "importance": 2
    }}
  ]
}}

Make at most {max_edits} changes. "importance" is a whole number from 1 to 5.
"""
