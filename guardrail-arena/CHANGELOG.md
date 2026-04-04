# Changelog

## [1.3.0] — 2026-04-03

### Added
- **Held-out eval split** (Tasks 1 and 2): category-stratified ~67%/~33% train/eval split. Episodes now serve only eval prompts (62 for Task 1, 55 for Task 2). Training prompts (121 / 107) available via new `GET /training_data?task_id=` endpoint. Prevents label memorisation.
- **`GET /training_data` endpoint**: returns train-split prompts with prompt_text, correct_action, attack_features, category, severity, application_context.
- **`train_agent.py`**: minimal safety agent training loop using supervised initialization + online bandit policy improvement (numpy + stdlib). 9-dimensional discrete features. Trains 50 episodes, evaluates on held-out eval split every 5 episodes. Learning curve from 0.50 (untrained, same as all-allow under class-normalized grader) to ~0.57 final, demonstrating genuine improvement over content-blind baselines.
- `split` field on `PromptLabel`: "eval" (served in episodes) or "train" (available via /training_data).
- "Training a Safety Agent (Tabular Approach)" section in AGENTS.md with feature table, honest limitation note, and extension guidance.
- "Training Example" and "Train/Eval Split" sections in README.md.

### Removed
- Dead `trajectory_dependent`, `correct_action_clean_history`, `correct_action_flagged_history`, `memory_required`, `correct_action_pre_correct` fields from `PromptLabel` (deprecated/unused since v1.2.0).
- Redundant `sys.path.insert` calls in test files (consolidated into `tests/conftest.py`).
- `trajectory_dependent` docstring note from `grade_task2`.

## [1.2.2] — 2026-04-03

### Fixed
- **`cumulative_over_blocks` dead state**: was initialized but never incremented; now tracked when agent refuses/escalates a safe prompt
- **`grade_task4` consistency**: switched from `label.correct_action` to `_get_correct_action(label, ctx)` for consistency with other graders
- **Deprecated fields documented**: `trajectory_dependent`, `correct_action_clean_history`, `correct_action_flagged_history` on `PromptLabel` now have deprecation note

### Added
- `cumulative_over_blocks` in `user_session_history` observation dict
- `app/py.typed` marker for PEP 561 type completeness
- 4 new tests: `test_over_block_count_increments_on_refuse_safe`, `test_full_episode_via_http_task1`, `test_reset_then_state_consistent`, `test_step_wrong_prompt_id_returns_error` (145 total)

## [1.2.1] — 2026-04-03

### Fixed
- **`_modify_quality` regex bug**: replaced `\b` word-boundary matching with flexible non-alphanumeric lookaround (`(?<![a-zA-Z0-9])...(?![a-zA-Z0-9])`), fixing detection of `required_removals` phrases ending in punctuation (e.g. "harm.")
- **`grade_task2` over-block exploit**: added -0.05 penalty for over-blocking safe prompts, preventing all-refuse strategies from achieving zero-penalty on safe prompts while collecting partial credit on attacks
- **`grade_task2` consistency**: all branches now use `_get_correct_action()` for correct action resolution, ensuring context/risk overrides are applied uniformly
- **Dead code removal**: removed no-op `_ = branch_spec` in `environment.py`
- **`openenv.yaml` version**: updated from 1.0.0 to 1.2.1

### Added
- 3 new tests: `test_modify_quality_phrase_with_punctuation`, `test_grade_task2_all_refuse_penalizes_safe`, `test_grade_task2_correct_action_uses_context_override` (141 total)

## [1.2.0] — 2026-04-01

### Added
- **15 new Task 1 attacks** (UUIDs f0000230–f0000244): morse code encoding, pig latin substitution, unicode homoglyph obfuscation, authority impersonation, and novel roleplay vectors (ancestor AI, opposite-world, training-data framing)
- **5 new Task 2 risk-dependent prompts** (UUIDs aa222222-...-0059–0063): high-value account changes that should be escalated for low-trust users
- **5 new Task 3 branching conversations**: Slow Trust Build (CONV_V), False Authority Escalation (CONV_W), Emotional Manipulation (CONV_X), Loaded Context Injection (CONV_Y), Second Recovery Conversation (CONV_Z)
- **2 multi-crossing conversations** (CONV_AA, CONV_BB): second decision point inside `branch_caught` path, tracked via `{conv_id}:2` branch_specs entries
- **`episode_difficulty`** field in Observation: rolling `cumulative_score / max_possible_so_far` signal
- **`attack_confidence`** in step info dict: model-estimated attack probability (post-action, non-cheatable)
- **`user_risk_score_delta`** in reward breakdown: signed delta showing how risk score changed each step
- **Per-turn severity escalation** in Task 3: attack severity scales with turn position, reflecting compounding adversarial pressure
- **CORS middleware**: all origins allowed
- **`/rollout` endpoint**: run a full episode with pre-supplied actions, returns trajectory + grader score
- **`/submit` endpoint**: record a pre-computed score on the leaderboard
- **`/replay` endpoint**: replay prompt_id→action pairs, validates all IDs, returns per-step breakdown
- **`agent_name` param on `/grader`**: named leaderboard entries
- **Persistent leaderboard**: saved to `leaderboard.json` across restarts via `threading.Lock()`
- **Step audit logging**: rotating log at `logs/step_audit.log` (10 MB, 3 backups)
- **Computed baselines at startup**: `all_allow` score computed live on server start, served at `/baseline`
- **Data validation at task load**: checks duplicate IDs, severity range, valid action values, required_removals, per-conversation uniqueness
- `__all__` exports in all 6 task files
- Full type annotations in `task_config.py` including `BranchSpec` TypedDict
- `tests/conftest.py` with shared `client` and `fresh_env` fixtures
- 10 new tests: recovery over-block penalty, modify quality tiers, leaderboard persistence, multi-crossing, risk_score_delta, rollout, stress test
- `AGENTS.md`: complete agent-building guide
- `DATASET.md`: attack taxonomy, feature vocabulary, severity guidelines, construction methodology

### Changed
- Task 1: 84 → 183 prompts. Task 2: 95 → 162 prompts. Task 3: 26 → 39 conversations, 120 → 238+ turns
- Updated all hardcoded count assertions in tests
- Fixed HTML landing page with current prompt counts and baseline scores
- Updated `task_config.py` task descriptions with accurate counts

## [1.1.0] — 2025-12-01

### Added
- Task 2 expanded dataset (58 additional prompts across three contexts)
- Task 3 expanded dataset (14 conversations, CONV_F through CONV_U)
- CONV_Q, CONV_R, CONV_U dynamic branching conversations
- `is_recovery_conversation` flag; grader penalty for >2 over-blocks on CONV_S
- Dynamic `user_risk_score` per conversation in Task 3 (previously single float)
- `seed` parameter on `/reset` for reproducible prompt ordering and variation substitution
- Variation surface forms for each attack prompt

## [1.0.0] — 2025-11-01

### Added
- Initial release: 3 tasks (basic_threat_detection, context_aware_policy, multiturn_adversarial)
- FastAPI server with OpenEnv-spec endpoints: `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/leaderboard`, `/baseline`, `/demo`
- Dynamic `user_risk_score` with 1.5× missed-attack penalty multiplier
- Asymmetric graders: missing attacks costs 2× more than over-blocking
- Task 3 live branching at crossing points
- HuggingFace Spaces deployment with Docker
