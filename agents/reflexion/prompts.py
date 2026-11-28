REFLECTION_PROMPT = (
    "You will be given the history of a past experience in which you were placed in an "
    "environment and given a task to complete. You were unsuccessful in completing the task. "
    "Do not summarize your environment, but rather think about the strategy and path you took "
    "to attempt to complete the task. Devise a concise, new plan of action that accounts for "
    "your mistake with reference to specific actions that you should have taken. For example, "
    "if you tried A and B but forgot C, then devise a plan to achieve C with "
    "environment-specific actions. You will need this later when you are solving the same "
    'task. Give your plan after "Plan".'
)

EXAMPLES = (
    "Here are examples of a failed attempt and the plan written after it, from a different "
    "goal than yours. They show the form, not the answer to your task."
)

# Cross-trial insights. Ported from ExpeL (Zhao et al., AAAI 2024, arXiv:2308.10144;
# github.com/LeapLabTHU/ExpeL, Apache-2.0), changing only what names a task family or
# an environment:
#   prompts/templates/human.py   FORMAT_RULES_OPERATION_TEMPLATE, CRITIQUE_SUMMARY_SUFFIX,
#                                the compare and all-success critique templates, RULE_TEMPLATE
#   prompts/alfworld.py          SYSTEM_CRITIQUE_*_EXISTING_RULES_INSTRUCTION
#   prompts/templates/system.py  "You are {ai_name}. {instruction}"
# ExpeL's "rules" are called insights in the paper and here. The operation words are the
# code's (ADD, EDIT, AGREE, REMOVE); `agents/reflexion/insights.py` also reads the
# paper's UPVOTE and DOWNVOTE as AGREE and REMOVE.

INSIGHT_AI_NAME = (
    "an advanced reasoning agent that can add, edit or remove rules from your existing rule set, "
    "based on forming new critiques of past task trajectories"
)

INSIGHT_SYSTEM = "You are {ai_name}. {instruction}"

INSIGHT_INSTRUCTION_COMPARE = (
    "You will be given two previous task trials in which you were placed in a game environment and a "
    "task to complete: one successful and one unsuccessful trial. You failed the trial because the "
    "environment ended it with {reason} before the task was complete."
)

INSIGHT_INSTRUCTION_SUCCESS = (
    "You will be given successful tasks trials in which you were placed in a game environment and tasks to complete."
)

INSIGHT_FORMAT = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most {max_ops} operations and each existing rule can only get a maximum of 1 operation. """

INSIGHT_COMPARE = """Here are the two previous trials to compare and critique:
TRIAL TASK:
{task}

SUCCESSFUL TRIAL:
{success}

FAILED TRIAL:
{failure}

Here are the EXISTING RULES:
{rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

"""

INSIGHT_SUCCESS = """Here are the trials:
{success}

Here are the EXISTING RULES:
{rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

"""

INSIGHT_SUFFIX_FULL = (
    "Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different "
    "from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:\n"
)
INSIGHT_SUFFIX_NOT_FULL = "Below are the operations you do to the above list of EXISTING RULES:\n"

INSIGHT_NONE_YET = "(none yet)"

INSIGHTS_HEADER = (
    "The following are some experience you gathered on earlier trials of this same task, by playing the "
    "game, in decreasing order of importance. Use these as references to help you perform this task:"
)

RETRIEVED_HEADER = (
    "Here are successful trials of this same task from your earlier trials, the most similar first. "
    "They show what worked; the game is the same."
)
