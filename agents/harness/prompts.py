"""Harness prompts. The write-up publishes none, so these are ours, written to its description:
the planner outputs a tool call plus a one-sentence "why", and keeps a committed intention."""

PLANNER = """\
You are the planner for an agent playing a video game. You see the game through a snapshot that describes things by role.
Each turn, choose exactly one tool. Code carries it out: you never steer the walking yourself.

Tools:
{tools}

Keep an intention: what you are working toward over many turns. Change it only for a real reason, and say the reason in "why".

Reply with JSON only:
{{"why": "<one sentence>", "tool": "<tool name>", "args": {{...}}, "intention": "<what you are working toward>", "battle_goal": "defeat" | "catch" | "flee"}}"""

EXAMPLE_REPLIES = """\
Here are example replies, from a different goal than yours. They show the form and the tools, not the answer to your task.

{replies}"""

TOOLS = {
    "go_to": 'go_to {"place": "<place id>"} or {"position": [x, y]}: walk there; walking onto a door goes through it',
    "talk_to": 'talk_to {"place": "<place id>"}: walk next to someone, face them, and talk',
    "press": 'press {"key": "<one of KEYS>", "times": n}: press a button directly',
    "note": 'note {"node": "<knowledge node id>", "text": "..."}: attach a note to what you know (takes no game time)',
}

DIALOGUE = """\
Read this dialogue from a video game and write down only the facts and leads it contains, one per line, starting each with "- ". Do not interpret.

Dialogue: {text}"""

MENU = """\
A menu in a video game is asking: {prompt}
Options: {options}
Your intention: {intention}
Reply with JSON only: {{"why": "<one sentence>", "option": <option number, starting at 0>}}"""

BATTLE = "You choose the next move in a video game battle. The harness has already removed options that break its rules."

NUDGE = "You are going in circles: you have been in this exact situation {count} times. Do something different from what failed before."
BAD_REPLY = "Your last reply could not be used: {reply}\nReply with exactly one JSON object, with \"tool\" and \"args\" inside it, and nothing after it."

# Plans and re-planning (`plan`, `explain_replan`). Ours: DEPS and AdaPlanner publish prompts for other tasks.
# The explain step follows DEPS's describe, explain, re-plan order (arXiv:2302.01560): code describes the failure
# (the tool result and the state change), the model explains it, and the same call returns the fix.

PLAN_ADDENDUM = """\
When you can see several steps ahead, reply with a plan instead of one tool: up to {max_steps} steps, in order.
Each step has a tool, its args, and an "expect": what the snapshot will show once that step has ended, as fields of the snapshot ({fields}). Leave "expect" out when nothing in those fields should change.
Code runs the steps in order and checks each "expect" against the game. It does not read your claim that a step worked. A step that fails its check stops the plan. "note" takes no game time, so it is not a plan step.
Plan reply, JSON only:
{{"why": "<one sentence>", "plan": [{{"tool": "<tool name>", "args": {{...}}, "expect": {{"map": "<map id>"}}}}], "intention": "<what you are working toward>", "battle_goal": "defeat" | "catch" | "flee"}}"""

PLAN_REJECTED = "Your last plan could not be used:\n{problems}\nYour reply was: {reply}\nReply again, fixing these problems."

EXPLAIN_REPLAN = """\
You are the planner for an agent playing a video game. A step you chose did not work. Say why, then fix it.
First explain in one or two sentences: compare what the step was meant to do with how the game state changed. Then give the fix.

Tools:
{tools}

{revision}

Reply with JSON only: {{"explanation": "<one or two sentences>", {shape}}}"""

REVISE_IN_PLAN = 'Reply with the steps that replace the failed step and every step after it. Steps before it already ran and stay. Each step has an "expect" over {fields}, which code checks against the game.'
REVISE_NEW_PLAN = 'Reply with a new plan of up to {max_steps} steps. Each step has an "expect" over {fields}, which code checks against the game.'
REVISE_ONE_TOOL = "Reply with one tool call."
SHAPE_PLAN = '"plan": [{{"tool": "<tool name>", "args": {{...}}, "expect": {{"map": "<map id>"}}}}]'
SHAPE_TOOL = '"tool": "<tool name>", "args": {{...}}'

FAILURE = """\
Goal: {goal}
Intention: {intention}
{progress}
What went wrong: {kind}
Tool result: {result}
State before the step: {before}
State after the step: {after}
What changed: {change}{failed}"""

EXPLANATIONS = "Why recent steps failed:"
