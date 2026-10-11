# AGENTS.md

Instructions for coding agents and human contributors. This file is public. It describes boundaries that stay true for the life of the project.

The working sequence is `LOCAL.md` at the repo root. That file is gitignored. When it is present, follow it. Do not copy its schedule, status, or "now / later" notes into `README.md`, this file, or commit messages.

## Boundaries

| Package | Owns | May depend on |
| --- | --- | --- |
| `environments` | The adapter: emulator, saves, observations, actions, progress | nothing in `agents` or `evaluation` |
| `agents` | Direct policy, ReAct, Reflexion, and Voyager (curriculum, skill library, iterative prompting) | the environment interface only |
| `evaluation` | Goals, trials, budgets, logs, metrics, ablations | environment and agent interfaces |

The adapter is the only code that knows the game is Pokémon. Agents see a snapshot and a goal. Evaluation decides scores from adapter state, budgets, and logs.

An agent imports no emulator, reads no game memory, and parses no save. It does not branch on the title of the game. A button enum or memory address from FireRed stays inside the adapter.

The adapter calls no language model, stores no reflections, and does not keep the skill library.

## Environment

- `reset()` starts or restores a known state. Two trials with the same config start from the same world.
- `observe()` returns the current snapshot.
- `step(action)` applies one environment action and returns the next snapshot.
- `get_progress()` returns progression read from the game.

Observation mode is part of the run config:

- **visual** — the screen only
- **structured** — symbolic state from emulator memory
- **both** — visual and structured together, reported as its own condition

A field a model inferred from pixels belongs to the agent. It is not part of the adapter's structured snapshot.

The adapter defines the legal actions (buttons, menus, movement, NPC interaction, battle choices). A higher-level action is allowed only when its expansion into environment steps is explicit, counted, and logged. Save and restore are adapter operations. Saved states, trajectories, and raw logs stay out of git (`runs/`, `saves/`).

## Agents

The runner calls one interface: observe, decide, step, until the goal is met or a budget runs out. The model id is part of the run config. An open-weights model served locally is a valid model. The player never imports the server, the weights path, or the emulator.

- **Direct policy** — the current snapshot and the legal actions go in; one action comes out. The request includes no earlier turns, reflections, skills, or curriculum. Disable thinking and any other hidden trace so the output is the decision. A typed-decision model uses this same player: one pass, a choice among the actions. Ablation against ReAct is the comparison.
- **ReAct** — reason, act, observe, repeat.
- **Reflexion** — ReAct, plus verbal reflections on failures that later decisions can read. Ablation: reflections off.
- **Voyager** — automatic curriculum, skill library, and iterative prompting. The curriculum proposes the next goal from progress and state. The library stores, retrieves, and composes skills. Iterative prompting revises a skill using environment feedback, execution errors, and self-verification. Each of those three can be turned off.

Voyager's Minecraft skills are code. FireRed skills have to be something this adapter can execute. Do not invent a Minecraft action space inside the Pokémon adapter.

The goal is data. Completing it is `get_progress()` and related adapter state. A sentence from the model saying the goal is done does not end the trial.

## Evaluation

Every run config records:

- task id, goal, and initial saved state
- agent name and which mechanisms are enabled
- model id
- observation mode
- action space
- budgets: environment steps, model calls, tokens, wall clock as applicable

Record task success, game progression, environment steps, token usage and cost, recovery, skill reuse, and reliability across repeated trials. Persist prompts, model calls, trajectories, and checkpoints with the run. Ablations are configs for the same runner.

## Layout

```text
environments/          # protocol, plus one folder per game
agents/                # protocol, plus one folder per architecture
evaluation/            # goals, runner, metrics, log schema
experiments/           # configs
```

Python. Contract tests for `reset` / `observe` / `step` / `get_progress` come before agent tests. Agent tests use a fake environment that implements the interface.

## What stays out of git

`LOCAL.md`, `.env`, `runs/`, `saves/`, emulator save states, API keys, and raw trajectories. Commit source, configs, and docs that still match the interfaces. Commit messages describe the change, not the private schedule.
