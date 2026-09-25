# Graph Report - gatelaya  (2026-09-25)

## Corpus Check
- 45 files · ~31,401 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 951 nodes · 2088 edges · 50 communities (41 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 105 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9ca32f4d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- LayaRouterAgent
- FakeAgent
- test_dashboard_api.py
- GateLayaConfig
- GateLayaGuardrail
- DecisionRecord
- GuardrailConfigurationError
- test_guardrail_post_call.py
- calibration_api.py
- review.py
- test_audit.py
- test_routing.py
- run_route
- test_calibration.py
- GateLaya — Every LLM call, decided.
- InMemoryAuditSink
- make_router
- db.py
- build_questions
- decisions.py
- .from_yaml
- put_config
- TemperatureMap
- choice_probs
- RoutingPolicy
- conftest.py
- test_integration_litellm.py
- test_questions.py
- _fit_rows
- build_routing_questions
- Product
- gatelaya/guardrail.py
- .from_yaml
- .load
- get_engine
- app.js
- noul_probs
- Any
- GateLaya module
- .from_dict
- test_noul_probs_garbage_raises_value_error
- custom_guardrail/gatelaya/__init__.py
- custom_guardrail/__init__.py
- dashboard/__init__.py
- routers/__init__.py
- tests/__init__.py
- test_choice_probs_probs_dict_normalized
- test_choice_probs_rejects_bad_lengths_and_labels
- test_tier_rewrite_hard_to_frontier
- gatelaya

## God Nodes (most connected - your core abstractions)
1. `GateLayaGuardrail` - 86 edges
2. `GateLayaConfig` - 85 edges
3. `InMemoryAuditSink` - 77 edges
4. `FakeAgent` - 69 edges
5. `RoutingPolicy` - 49 edges
6. `GuardrailConfigurationError` - 41 edges
7. `LayaRouterAgent` - 39 edges
8. `TemperatureMap` - 37 edges
9. `DecisionRecord` - 33 edges
10. `make_router()` - 31 edges

## Surprising Connections (you probably didn't know these)
- `CalibrationView` --uses--> `TemperatureMap`  [INFERRED]
  dashboard/routers/calibration_api.py → gatelaya/calibration.py
- `CalibrationView` --uses--> `GuardrailConfigurationError`  [INFERRED]
  dashboard/routers/calibration_api.py → gatelaya/errors.py
- `UploadAck` --uses--> `TemperatureMap`  [INFERRED]
  dashboard/routers/calibration_api.py → gatelaya/calibration.py
- `UploadAck` --uses--> `GuardrailConfigurationError`  [INFERRED]
  dashboard/routers/calibration_api.py → gatelaya/errors.py
- `ConfigView` --uses--> `GateLayaConfig`  [INFERRED]
  dashboard/routers/config_api.py → gatelaya/config.py

## Import Cycles
- None detected.

## Communities (50 total, 9 thin omitted)

### Community 0 - "LayaRouterAgent"
Cohesion: 0.05
Nodes (59): Bucket, CalibrationView, BaseModel, Stored temperature map plus its file path., Fitted temperatures written to disk (proxy reloads at next boot)., UploadAck, detect_bucket(), LayaAgent (+51 more)

### Community 1 - "FakeAgent"
Cohesion: 0.06
Nodes (61): clean_answers(), FakeAgent, Configurable ``LayaAgent`` stand-in. Records every ``predict`` call, and…, Number of predict invocations so far., Implements the ``LayaAgent`` protocol., All four checks confidently negative, plus a neutral pii_type answer., assert_valid_record(), Any (+53 more)

### Community 2 - "test_dashboard_api.py"
Cohesion: 0.07
Nodes (50): AsyncClient, auth_client(), _clear_caches(), client(), env(), _laya_installed(), Any, fixture (+42 more)

### Community 3 - "GateLayaConfig"
Cohesion: 0.07
Nodes (38): Action, GateLayaConfig, BaseModel, field_validator, Thresholds, actions, routing, and runtime flags for the guardrail., Return the confidence threshold for a check., Return the configured action for a check., _env_overrides() (+30 more)

### Community 4 - "GateLayaGuardrail"
Cohesion: 0.11
Nodes (21): LiteLLM resolves custom_guardrail.gatelaya.guardrail.GateLayaGuardrail here., GateLayaGuardrail, _Outcome, Any, Exception, Laya-powered LiteLLM guardrail: block/mask/flag PII, injection, toxicity,…, Run the agent once for all checks; returns (outcomes, latency_ms, error)., Evaluate one check's answer against its threshold and configured action. (+13 more)

### Community 5 - "DecisionRecord"
Cohesion: 0.08
Nodes (25): LiteLLM resolves custom_guardrail.gatelaya.router.GateLayaRouter here., DecisionRecord, BaseModel, One guardrail decision: what ran, what happened, how long it took., Persist one decision record., Append a record, evicting oldest entries past the cap., Extract text from a LiteLLM/OpenAI-style response object or dict., _response_text() (+17 more)

### Community 6 - "GuardrailConfigurationError"
Cohesion: 0.09
Nodes (25): AuditSink, Protocol, Decision audit records and pluggable sinks (in-memory / SQLAlchemy)., Create the decisions table if it does not exist., Insert one decision row (lazily creating the table on first write)., Dispose the underlying engine pool., Async sink for decision records., Async SQLAlchemy sink writing to the `guardrail_decisions` table. (+17 more)

### Community 7 - "test_guardrail_post_call.py"
Cohesion: 0.10
Nodes (35): Any, skipif, Tests for GateLayaGuardrail.async_post_call_success_hook (response scanning)., Arrange: raw dict response (choices/message/content). Act: hook. Assert: text…, Arrange: response with no/blank content. Act: hook. Assert: None and agent…, Arrange: real litellm.ModelResponse. Act: hook (clean). Assert: text extracted…, Arrange: real ModelResponse leaking a secret. Act: hook. Assert: HTTPException…, Arrange: secret_leak action=flag above threshold. Act: hook. Assert: no… (+27 more)

### Community 8 - "calibration_api.py"
Cohesion: 0.10
Nodes (26): init_db(), Create the audit and review tables if they do not exist., SettingsDep, Shared route dependencies: settings, DB session, bearer-token auth., Allow the request when no token is configured or the bearer token matches., require_token(), create_app(), _ensure_static_dir() (+18 more)

### Community 9 - "review.py"
Cohesion: 0.10
Nodes (29): export_labels(), label_decision(), _label_out(), LabelAck, LabelIn, LabelOut, _labels_by_decision(), list_review() (+21 more)

### Community 10 - "test_audit.py"
Cohesion: 0.10
Nodes (25): make_record(), Any, skipif, Tests for DecisionRecord and audit sinks (in-memory + SQLAlchemy)., Arrange: sink capped at 3. Act: record 5. Assert: only the newest 3 survive…, Arrange/act: max_records < 1. Assert: GuardrailConfigurationError., Arrange: sink. Act: isinstance check against runtime-checkable protocol.…, Arrange: async sqlite in-memory sink. Act: record one decision, read rows.… (+17 more)

### Community 11 - "test_routing.py"
Cohesion: 0.10
Nodes (24): MonkeyPatch, Phase 2 tests: RoutingPolicy config, GateLayaRouter pre-call model routing., Arrange: GATELAYA_ROUTE_* env vars, no explicit policy. Act: construct router.…, Arrange: env set but policy passed explicitly. Act: construct. Assert: env NOT…, Arrange: GATELAYA_ROUTE_DEFAULT=''. Act: construct without policy. Assert:…, Arrange: GATELAYA_ROUTE_TIERS that is not JSON. Act: construct. Assert:…, Arrange: GATELAYA_ROUTE_TIERS JSON array. Act: construct. Assert:…, Arrange: GATELAYA_ROUTING_PATH pointing at a policy YAML. Act: construct… (+16 more)

### Community 12 - "run_route"
Cohesion: 0.11
Nodes (24): make_data(), Arrange: complexity=trivial, tiers maps trivial->cheap-mini. Act: hook. Assert:…, Arrange: default policy (no tiers). Act: hook. Assert: original model kept., Arrange: raw confidence 0.75 >= 0.70 but choice temperature 3.0 pulls the…, Arrange: enabled=False. Act: hook. Assert: None returned, agent never called,…, Arrange: agent raises. Act: hook. Assert: no exception, original model kept,…, Arrange: complexity answer not in options. Act: hook. Assert: treated like an…, Arrange: answers without 'complexity'. Act: hook. Assert: original model kept… (+16 more)

### Community 13 - "test_calibration.py"
Cohesion: 0.11
Nodes (21): calibrated(), Temperature-scale a probability vector (log-space) and renormalize., Tests for temperature calibration: TemperatureMap I/O, scaling, and fit()., Arrange: arbitrary probs. Act: temperature scale. Assert: sums to ~1, values in…, Arrange: probs. Act: T=1 scaling. Assert: unchanged (within float error)., Arrange: confident binary probs. Act: raise T. Assert: distribution moves…, Arrange: a zero entry (would be log(0)). Act: scale. Assert: finite result…, Arrange/act: T<=0 and empty probs. Assert: GuardrailConfigurationError. (+13 more)

### Community 14 - "GateLaya — Every LLM call, decided."
Cohesion: 0.09
Nodes (21): Actions, API endpoints, Architecture, Audit log, Calibration (required), Checks, Configuration, Dashboard (Phase 3) (+13 more)

### Community 15 - "InMemoryAuditSink"
Cohesion: 0.15
Nodes (19): InMemoryAuditSink, Capped in-memory sink for tests and audit-disabled runs., LogCaptureFixture, FakeStreamResponse, Minimal async-iterator matching what the streaming hook consumes., collect_chunks(), Any, Tests for the streaming hook: pass-through, no enforcement in v1. (+11 more)

### Community 16 - "make_router"
Cohesion: 0.14
Nodes (21): make_router(), Any, Arrange: complexity=moderate with no moderate entry. Act: hook. Assert:…, Arrange: sensitive_p=0.95 >= 0.85 with sensitive_model set (tier also matches).…, Arrange: sensitive request but sensitive_model=None. Act: hook. Assert: tier…, Arrange: raw sensitive p=0.86 >= 0.85, but noul temperature 2.0 pulls the…, Arrange: complexity confidence 0.40 < 0.70, default_model set. Act: hook.…, Arrange: low confidence, default_model=None. Act: hook. Assert: original model… (+13 more)

### Community 17 - "db.py"
Cohesion: 0.14
Nodes (18): datetime, Async engine, session factory, and dashboard schema (audit + review tables)., Attach/convert to UTC — sqlite returns naive UTC wall times., utc_dt(), _count_when(), _group_counts(), AsyncSession, BaseModel (+10 more)

### Community 18 - "build_questions"
Cohesion: 0.14
Nodes (18): fit(), Temperature calibration: maps, temperature scaling, and NLL grid fit., Write a temperature map to a JSON file., Grid-search the scalar temperature minimizing NLL over (probs, true_label)…, save_temperature_map(), build_questions(), Merge question dicts for the given checks; returns (questions, names)., Namespace (+10 more)

### Community 19 - "decisions.py"
Cohesion: 0.15
Nodes (18): Any, Build a DecisionRecord from a guardrail_decisions row mapping., row_to_record(), _clauses(), DecisionListOut, get_decision(), list_decisions(), BaseModel (+10 more)

### Community 20 - ".from_yaml"
Cohesion: 0.14
Nodes (16): Path, Write this configuration to a YAML file., Load configuration from a YAML file., Path, Arrange: config with overrides. Act: to_yaml -> from_yaml. Assert: equal., Arrange/act: nonexistent path. Assert: GuardrailConfigurationError., Arrange: YAML with bad action value. Act: from_yaml. Assert: config error., Arrange: YAML list root. Act: from_yaml. Assert: config error. (+8 more)

### Community 21 - "put_config"
Cohesion: 0.16
Nodes (17): ConfigPatch, ConfigView, get_config(), load_config(), put_config(), BaseModel, get, Path (+9 more)

### Community 22 - "TemperatureMap"
Cohesion: 0.13
Nodes (14): load_temperature_map(), Path, Per-(question_type, option_count) temperatures: {"choice": {...}, "noul":…, Return the temperature for a question type (choice keyed by option count)., Return the JSON-serializable map., Write the temperature map to a JSON file., Load a temperature map from a JSON file., TemperatureMap (+6 more)

### Community 23 - "choice_probs"
Cohesion: 0.12
Nodes (16): _aligned(), choice_probs(), Build Laya question dicts for GateLaya checks and normalize Laya answers., Normalize a Laya choice answer to probs aligned with `options` order., Arrange/act: {'choice': 'personal', 'confidence': 0.8}. Assert: selected gets…, Arrange/act: choice without confidence. Assert: one-hot., Arrange/act: plain label string. Assert: one-hot vector., Arrange/act: {'probabilities': {...}} alias. Assert: same alignment. (+8 more)

### Community 24 - "RoutingPolicy"
Cohesion: 0.12
Nodes (15): BaseModel, field_validator, Tier, sensitivity, and confidence rules for the model router., RoutingPolicy, parametrize, Arrange/act: tier key outside {trivial, moderate, hard}. Assert:…, Arrange/act: capitalized tier key. Assert: rejected (keys are exact)., Arrange/act: threshold outside [0,1]. Assert: pydantic ValidationError. (+7 more)

### Community 25 - "conftest.py"
Cohesion: 0.16
Nodes (16): audit_sink(), config(), fake_agent(), guardrail(), make_guardrail(), pre_call_data(), fixture, Shared fixtures and fakes for the GateLaya test suite. No network, no model… (+8 more)

### Community 26 - "test_integration_litellm.py"
Cohesion: 0.13
Nodes (15): MonkeyPatch, skipif, Integration checks: LiteLLM boot path for the custom guardrail pack. Verifies…, Arrange: proxy_config-shaped guardrail dict + LitellmParams. Act:…, Arrange/act: import the dotted path litellm uses in proxy_config.yaml. Assert:…, Arrange/act: class hierarchy. Assert: real litellm base class used., Arrange: no config/agent/audit — exactly what litellm passes at boot. Act:…, Arrange/act: safe_load proxy_config.yaml. Assert: mapping with guardrails. (+7 more)

### Community 27 - "test_questions.py"
Cohesion: 0.12
Nodes (15): Tests for question building and Laya answer normalizers (questions.py)., Arrange/act: all checks. Assert: every question present, pii_type included., Arrange/act: all checks. Assert: type field within the allowed set., Arrange/act: subset ['toxicity']. Assert: no other checks leak in., Arrange/act: injection only (e.g. enabled_checks restriction). Assert:…, Arrange/act: checks including pii_type. Assert: choice <= 20 options (PRODUCT…, Arrange/act: unknown check name. Assert: ValueError., Arrange/act: no checks. Assert: empty questions and names. (+7 more)

### Community 28 - "_fit_rows"
Cohesion: 0.17
Nodes (15): _fit_rows(), get_calibration(), _label_index(), _parse_rows(), Any, get, post, SettingsDep (+7 more)

### Community 29 - "build_routing_questions"
Cohesion: 0.16
Nodes (15): build_routing_questions(), injection_questions(), pii_questions(), Model-routing questions: task (choice), complexity (choice), sensitive (noul)., PII detection (noul) plus PII-type follow-up (choice)., Prompt-injection / jailbreak detection (noul)., Toxicity detection (noul)., Secret-leak detection (noul). (+7 more)

### Community 30 - "Product"
Cohesion: 0.13
Nodes (14): Brand, Constraints, Core Workflows, Functional Capabilities, Name, Non-Goals (v1), One-liner, Positioning (+6 more)

### Community 31 - "gatelaya/guardrail.py"
Cohesion: 0.19
Nodes (8): GateLaya CustomGuardrail: Laya-powered pre/post call hooks for LiteLLM Proxy., CustomGuardrail, Any, Fallback CustomGuardrail base used ONLY when litellm is not importable. CLEARLY…, Minimal stand-in matching the litellm CustomGuardrail surface GateLaya uses., Stub pre-call hook (real behavior lives in GateLayaGuardrail)., Stub post-call hook (real behavior lives in GateLayaGuardrail)., Stub streaming hook (real behavior lives in GateLayaGuardrail).

### Community 32 - ".from_yaml"
Cohesion: 0.18
Nodes (12): Path, Load a routing policy from a YAML file., Write this routing policy to a YAML file., Path, Arrange: policy with overrides. Act: to_yaml -> from_yaml. Assert: equal., Arrange/act: nonexistent path. Assert: GuardrailConfigurationError., Arrange: YAML with unknown tier key. Act: from_yaml. Assert: config error., Arrange: YAML list root. Act: from_yaml. Assert: config error. (+4 more)

### Community 33 - ".load"
Cohesion: 0.22
Nodes (10): Load a temperature map from a JSON file., Path, Arrange: populated map. Act: save -> load. Assert: identical., Arrange/act: missing path. Assert: GuardrailConfigurationError., Arrange: malformed JSON. Act: load. Assert: config error., Arrange: map. Act: save. Assert: valid JSON on disk with both sections., test_load_invalid_json_raises(), test_load_missing_file_raises() (+2 more)

### Community 34 - "get_engine"
Cohesion: 0.20
Nodes (10): async_sessionmaker, AsyncEngine, get_engine(), get_sessionmaker(), AsyncSession, Cached async engine for GATELAYA_DATABASE_URL., Cached session factory used by the per-request dependency., get_db() (+2 more)

### Community 35 - "app.js"
Cohesion: 0.28
Nodes (7): dashboard(), DEFAULT_ACTIONS, DEFAULT_THRESHOLDS, FILTER_KEYS, fmtClock(), relTime(), THRESHOLD_CHECKS

### Community 36 - "noul_probs"
Cohesion: 0.22
Nodes (9): noul_probs(), Any, Normalize a Laya noul answer to [P(false), P(true)]., Arrange/act: bool/int/float shapes. Assert: normalized pairs., Arrange/act: values outside [0,1]. Assert: clamped, still sums to 1., Arrange/act: {'noul': 0.7}. Assert: [P(false), P(true)]., test_noul_probs_bool_and_number_answers(), test_noul_probs_clamps_out_of_range() (+1 more)

### Community 37 - "Any"
Cohesion: 0.22
Nodes (4): Any, Exception, Questions dict from the most recent predict call., State dict from the most recent predict call.

### Community 38 - "GateLaya module"
Cohesion: 0.25
Nodes (7): Audit, Calibration (required before trusting probabilities), Checks and actions, GateLaya module, Install, Programmatic use, v1 limitations (documented)

### Community 39 - ".from_dict"
Cohesion: 0.33
Nodes (5): Build a map from its JSON dict form., Arrange/act: bad dict forms. Assert: GuardrailConfigurationError., Arrange: JSON-parsed dict with int keys/values. Act: from_dict. Assert: keys…, test_from_dict_coerces_keys_and_values(), test_from_dict_rejects_non_mapping_and_bad_shapes()

### Community 40 - "test_noul_probs_garbage_raises_value_error"
Cohesion: 0.50
Nodes (4): Any, parametrize, Arrange/act: garbage input. Assert: ValueError with useful message., test_noul_probs_garbage_raises_value_error()

## Knowledge Gaps
- **41 isolated node(s):** `THRESHOLD_CHECKS`, `DEFAULT_THRESHOLDS`, `DEFAULT_ACTIONS`, `FILTER_KEYS`, `gatelaya` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GateLayaConfig` connect `GateLayaConfig` to `LayaRouterAgent`, `FakeAgent`, `test_dashboard_api.py`, `GateLayaGuardrail`, `GuardrailConfigurationError`, `test_guardrail_post_call.py`, `calibration_api.py`, `InMemoryAuditSink`, `.from_yaml`, `put_config`, `conftest.py`, `gatelaya/guardrail.py`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `GateLayaGuardrail` connect `GateLayaGuardrail` to `LayaRouterAgent`, `FakeAgent`, `GateLayaConfig`, `DecisionRecord`, `GuardrailConfigurationError`, `test_guardrail_post_call.py`, `InMemoryAuditSink`, `TemperatureMap`, `conftest.py`, `test_integration_litellm.py`, `gatelaya/guardrail.py`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `DecisionRecord` connect `DecisionRecord` to `FakeAgent`, `test_dashboard_api.py`, `GateLayaGuardrail`, `GuardrailConfigurationError`, `test_audit.py`, `test_routing.py`, `db.py`, `decisions.py`, `RoutingPolicy`, `gatelaya/guardrail.py`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `GateLayaGuardrail` (e.g. with `LayaAgent` and `LayaRouterAgent`) actually correct?**
  _`GateLayaGuardrail` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `GateLayaConfig` (e.g. with `ConfigPatch` and `ConfigView`) actually correct?**
  _`GateLayaConfig` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `InMemoryAuditSink` (e.g. with `GuardrailConfigurationError` and `GateLayaGuardrail`) actually correct?**
  _`InMemoryAuditSink` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FakeAgent` (e.g. with `InMemoryAuditSink` and `GateLayaConfig`) actually correct?**
  _`FakeAgent` has 5 INFERRED edges - model-reasoned connections that need verification._