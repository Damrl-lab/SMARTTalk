# Prompt Files

This folder stores the prompt files used by the SMARTTalk inference and
evaluation scripts.

- `raw_system_prompt.txt`: Raw-LLM baseline prompt from the supplementary material Figure 1.
- `raw_user_prompt_template.txt`: runtime wrapper for raw SMART windows.
- `heuristic_system_prompt.txt`: Heuristic-LLM baseline prompt from supplementary Figure 2.
- `heuristic_user_prompt_template.txt`: runtime wrapper for heuristic trend summaries.
- `smarttalk_system_prompt.txt`: SMARTTalk classifier prompt from supplementary Figure 3.
- `smarttalk_user_prompt_template.txt`: runtime wrapper for prototype summaries.
- `explanation_oracle_system_prompt.txt`: explanation/recommendation generation prompt for failed windows.
- `explanation_oracle_user_prompt_template.txt`: runtime wrapper used before judge and perturbation scoring.
- `judge_prompt_template.txt`: LLM-as-a-judge prompt from supplementary Figure 4.

The runnable scripts also keep in-code fallbacks for robustness, but these files
are included so the prompts can be inspected directly and reused by external
runners if needed.
