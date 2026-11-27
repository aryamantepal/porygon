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
