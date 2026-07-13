# Plan: Add `echo` + prompt logprobs support to C++ `llama-server` `/v1/completions`

## Goal
Make `llama-server`'s `/v1/completions` endpoint honor `echo: true` + `logprobs: N`, returning the **legacy OpenAI format** (`text_offset`, `token_logprobs`, `tokens`, `top_logprobs`) that lm-eval-harness's `gguf`/`local-completions` backends expect. This lets all loglikelihood tasks (piqa, hellaswag, gpqa, arc, winogrande) run on the fast C++ server with `-np N` true concurrency, instead of the slow single-request llama-cpp-python server.

## Approach: Port PR #15189 to current codebase
PR #15189 (fo40225, open) implements exactly this and was validated against lm-eval-harness on PIQA/HellaSwag/OpenBookQA/WinoGrande with correct results. It's based on the old monolithic `server.cpp`; the current code is split across `server-context.cpp` / `server-task.cpp` / `server-task.h` / `server-schema.cpp` / `server-common.cpp`. I'll port the logic, adapting field names.

### Key field mappings (old PR → current code)
- `slot.prompt_tokens` → `slot.prompt.tokens` (type `server_tokens`; index via `.get_tokens()[i]` since no `operator[]`)
- `slot.params.echo` → `slot.task->params.chat_parser_params.echo` (already parsed by `server-schema.cpp:329`)
- `ctx` / `vocab` → `ctx_tgt` / `ctx_server.vocab`
- `slot.prompt_tokens.detokenize(ctx, true)` → `slot.task->tokens.detokenize(ctx_tgt, true)`
- `slot.prompt_tokens.size()` → `slot.prompt.tokens.size()`
- `llama_get_logits_ith` / `llama_get_logits` / `llama_n_batch` / `common_batch_add` / `validate_utf8` — all already used in the codebase

## Changes (4 files)

### 1. `tools/server/server-common.cpp` — remove the echo rejection
- In `oaicompat_completion_params_parse` (line ~805): delete the block that throws `"Only no echo is supported"`. `echo` is already stored into `chat_parser_params.echo` by `server-schema.cpp:329`, so parsing just needs to stop rejecting it.

### 2. `tools/server/server-task.h` — add fields
- `completion_token_output`: add static method `oaicompat_probs_vector_to_json(probs_out, post_sampling_probs, echo, prompt_probs={})` producing the legacy format.
- `server_task_result_cmpl_final`: add `bool echo = false;` and `std::vector<completion_token_output> prompt_probs_output;`
- `server_task_result_cmpl_partial`: add `bool echo = false;`, `std::string prompt_text;`, `bool is_first_chunk = false;` (for streaming echo; keeps parity with PR even if we mainly use non-stream)
- `server_slot` (in server-context.cpp): add `std::string prompt_text;` and `std::vector<completion_token_output> prompt_token_probs;`; clear them in the slot reset method.

### 3. `tools/server/server-task.cpp` — implement the JSON formatter + wire echo into responses
- Implement `oaicompat_probs_vector_to_json`: concatenate `prompt_probs` (if echo) + `probs_out`, build `text_offset`/`token_logprobs` (with `std::optional<float>`, null for first/invalid), `tokens`, `top_logprobs` (map<string,float>). Port directly from the PR diff.
- `to_json_oaicompat()` (final, ~line 398): when `echo && prompt_probs_output.size() > 0`, call `oaicompat_probs_vector_to_json` instead of the `content`-array `probs_vector_to_json`. Also set `text` to `prompt + content` when `echo && !stream`.
- `to_json_oaicompat()` (partial, ~line 1080s): wire `response_text = prompt_text + content` when `echo && is_first_chunk`.

### 4. `tools/server/server-context.cpp` — capture prompt logprobs during launch
- In `launch_slot_with_task` (after `task.tokens.validate`, ~line 1810, before samplers): port the PR's prefill block. When `slot.task->params.chat_parser_params.echo` is true (note: `slot.task` is set at line 1849, so read echo from the local `task.params.chat_parser_params.echo` before the move, or set `slot.prompt_text` after):
  - `slot.prompt_text = task.tokens.detokenize(ctx_tgt, true);`
  - If `n_probs > 0 && size > 1 && prompt_token_probs.empty()`: do a **separate prefill pass** with `llama_memory_clear` + batched `llama_decode` (output=true for all), capture logits, compute per-token logprob = `logit[tok] - logsumexp(logits)` for positions 1..n (position 0 = null/-inf), and top-N candidates via `std::partial_sort`. Store into `slot.prompt_token_probs`.
  - Else (no n_probs): fill `prompt_token_probs` with placeholder entries (tok + text + -inf prob).
- In `send_partial_response` / `process_token` (where `server_task_result_cmpl_partial` is built, ~line 2084-2145): set `res->echo`, `res->prompt_text`, `res->is_first_chunk = (slot.n_decoded == 1)`.
- In the final-response builder (~line 2145-2185): set `res->echo = slot.task->params.chat_parser_params.echo;`, `res->prompt = echo ? slot.prompt_text : detokenize(...)`; and `if (echo && !prompt_token_probs.empty()) res->prompt_probs_output = slot.prompt_token_probs;`.

## Important correctness details
- The separate prefill pass clears KV memory (`llama_memory_clear`) and re-decodes the prompt to get logits at every position. This **doubles prompt processing cost** for echo requests but is the only way to get per-token prompt logprobs. Acceptable for eval (echo requests are loglikelihood tasks where correctness > speed).
- Must read `echo` from `task.params` BEFORE `std::move(task)` into `slot.task` at line 1849. I'll capture `const bool echo = task.params.chat_parser_params.echo;` early.
- `logprobs` field: lm_eval sends `logprobs: 1` (gguf backend) or `logprobs: 10`. The server maps `logprobs`→`n_probs` via `server-schema.cpp:180` alias. So `n_probs` is already set correctly — no change needed there.
- Non-stream path only is fine for lm_eval (it doesn't stream). I'll implement both per the PR but test non-stream.

## Build & test
1. Apply changes, rebuild: `cmake --build build --config Release -j` (GGML_CUDA already on in the existing build dir).
2. Restart server with `-np 8 -cb` (no `--skip-chat-parsing` needed for echo path, but keep `--reasoning off`).
3. curl test: `{"prompt":"Hello world","max_tokens":1,"echo":true,"logprobs":1}` → expect `token_logprobs` array with prompt tokens, HTTP 200.
4. Run lm_eval piqa with `--model gguf --model_args base_url=http://127.0.0.1:8901` — expect no KeyError, real `acc` score, fast (concurrency).

## Risk / scope
- ~120 lines added across 4 files. Largest risk is the prefill-pass logic in `launch_slot_with_task` interacting with the existing prompt-caching/checkpoint machinery — I'll place it carefully and clear/restore state to not corrupt the normal prefill that follows.
- If the separate prefill pass causes issues with the subsequent normal prefill (double-processing), fallback: skip the separate pass and instead capture logits during the normal prefill by setting `output=true` on prompt tokens. But the PR's separate-pass approach is proven, so try it first.
