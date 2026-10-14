# The miner side of signed episodes: client, renderer and bridge

A miner does inference only. It never runs a tool. It opens an episode with a validator-signed session token, sends each tool call the model wrote to the machine's gateway, shows the model the text `reliquary_sandbox.observation` renders from the signed record, and finishes. This page describes the pieces as they are in the code:

| Module | Depends on | Role |
| --- | --- | --- |
| `reliquary_sandbox.observation` | stdlib, `reliquary_sandbox.attest` | `plan_call`, `render_observation`, the pinned texts |
| `reliquary_sandbox.episode_client` | `httpx` | `EpisodeClient`, `Episode`: open, call, finish, transcript, with retries |
| `reliquary_sandbox.tool_routing` | no verifiers | `ToolRouter`: answers a model's tool calls once each, in order |
| `reliquary_sandbox_verifiers` | the `verifiers` extra | verifiers' bash harness driven through the gateway; `SandboxEpisodeRunner` |

The validator imports `observation` and `attest` too, to check the calls and observations against the proven tokens; a copy of either drifts.

## Wire rules

From the docstring of `src/reliquary_sandbox/episode_client.py`:

- Bodies are compact UTF-8 JSON (`ensure_ascii=False`, no spaces). A `\uXXXX`-escaped body is up to six times larger and can hit the call body limit while the canonical arguments fit.
- A call is identified by `(turn, k)`. Resending one is always safe: the gateway returns the stored record without running it again. Network errors, timeouts, `503 episodes are recovering` and `409 episode is busy` (this call or another still runs; the gateway answers busy instead of queueing) are retried with the same `(turn, k)`: at least `MIN_RESENDS` (3) times, then within `retry_s` (300 s; `busy_wait_s`, 900 s, for busy) counted from the end of the first failure, and never past the caller's `deadline`. Only transport errors that are network errors, timeouts or remote protocol errors are retried; other transport errors are configuration faults and raise at once.
- An open is idempotent per token: resending the token returns the same record 0 and key while the episode runs, `409 episode is busy...` while it still opens (waited out), and `409` with the signed final once it is closed (`EpisodeClosed`). A `503` without `Retry-After` means the open failed and the token is spent (a resend answers `409` with no final): ask the validator for a new token. A full machine (`503` with `Retry-After`) is not retried.
- Send only what `plan_call` says to send. Arguments with no canonical form are refused locally, with nothing recorded.
- `finish` is idempotent. A finish after the deadline answers `409` with the signed `expired` final, which `finish()` returns like any other final.
- A call may return a final (`CallResult.final`): the CPU budget ran out, the deadline passed, or the transcript came within one worst-case call and final of the machine's cap (`budget_exhausted`, reason `transcript_bytes`; that call is recorded). A signed transcript never exceeds the cap the capacity report states (`caps.max_transcript_bytes`, at most 8 MiB of compact UTF-8 JSON). Later calls raise `EpisodeClosed`.
- An episode whose machine key rotated is void and unsigned (`aborted_unsigned`): calls and `finish()` raise `EpisodeClosed` with `final` set to `None`.

Timeouts the client uses: open 900 s, finish 3900 s, a call `per_call_timeout_s + 60 s`, transcript 60 s.

## What the model sees: `reliquary-observation/1`

Valid for `tools_version` `reliquary-tools/1`. The text is a function of the signed call body's `tool`, `output`, `truncated` and `timed_out` only. Rendering is forward-only: a record determines its observation, and nothing maps an observation back to a record. A validator renders the signed record and compares; it never parses what the model saw.

| Case | Text |
| --- | --- |
| Output with visible content | the output as recorded (stdout then stderr, UTF-8 with replacement; runtime lines already replaced by `[runtime output removed]` before signing), never altered |
| Output empty or whitespace only (`output.strip() == ""`) | `(no output)` |
| `bash` and `truncated` | note `[output truncated]` on its own line (a newline is added first unless the output ends with one) |
| `bash` and `timed_out` | after the truncation note, on its own line: `[command timed out; its process group was killed. Run long jobs in the background with their output redirected to a file, or narrow the command.]` |
| `edit` | the output only; the notes apply to `bash` only |

The exit code is not shown, as in verifiers' bash harness. An observation is never empty: verifiers' tool hook answers "allow" when a rewritten message equals the incoming one (content `""`), and the bash program would then run the command on the miner's host. Whitespace-only output also renders `(no output)`, because Qwen's chat template strips tool content. `render_observation` raises `ValueError` for a body whose `tool` or `output` is not text or whose flags are not booleans.

Calls refused locally by `plan_call(tool, arguments_text, offered=None)` get a pinned text and consume no record and no `(turn, k)`:

| Reason | Observation |
| --- | --- |
| `bad_tool_name` (not 1 to 64 characters of valid text) | `error: the tool name must be 1 to 64 characters of valid text` |
| `not_offered` (a name outside the offered set; checked before the arguments) | `error: that tool is not available in this episode; use one of the tools listed` |
| `not_text` (arguments not a string) | `error: the tool arguments must be sent as JSON text; resend the call` |
| `invalid_json` | `error: the tool arguments are not valid JSON; resend the call with a JSON object` |
| `not_an_object` | `error: the tool arguments must be a JSON object; resend the call as an object` |
| `not_canonical` (unpaired surrogate, non-finite number, integer outside 64 bits, nesting over 64) | `error: the tool arguments hold a value that cannot be represented exactly (an unpaired surrogate, a non-finite number, an integer outside 64 bits or nesting deeper than 64 levels); resend them` |
| `too_large` (canonical arguments over 120 KiB, a protocol constant) | `error: the tool arguments exceed 122880 bytes; split the work into smaller calls` |

Empty or missing arguments mean `{}`. Everything else is sent as parsed, byte for byte, blank commands and unknown tools included (the gateway records them as errors).

Positions: within a turn, sent calls take `k = 0, 1, ...` and refused calls take none (`sent_positions(plans)` returns `None` for them). `turn` is the number of assistant messages before the one that made the call.

## Record 0 and the offered tools

Record 0 signs the task's offered tools (`tools`: sorted, unique, a subset of `bash`, `edit`). The harness lists exactly those, and the router passes them as `offered` to `plan_call`; there is no separate `tools` parameter to get wrong. `verify_transcript` exposes them as `VerificationResult.tools` and refuses a call record naming a tool outside them (`tool_not_offered`). The bash harness always lists `bash`, so `env_config` accepts `{bash}` or `{bash, edit}`, never an edit-only set.

## The bridge

`reliquary_sandbox_verifiers` drives verifiers' pinned bash harness. Install it with the extra: `uv sync --extra verifiers`, which resolves `verifiers` at git commit `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`.

```python
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from reliquary_sandbox_verifiers import SandboxEpisodeRunner

ctx = ModelContext(model="my-model",
                   client=EvalClientConfig(base_url="http://127.0.0.1:8001/v1",
                                           api_key_var="MODEL_KEY"))
async with SandboxEpisodeRunner(ctx=ctx, gateway_url=machine_url, max_turns=48) as runner:
    result = await runner.run(token=token, prompt=prompt)   # prompt: the env's sandbox_prompt
if result.error is None:
    submit(result.transcript, result.final, result.state)
```

- `run` opens the token's episode, runs the agent, always finishes the episode, then reads the transcript. The agent stops `FINISH_MARGIN_S` (60 s) before the episode's wall deadline, measured from before the open on a monotonic clock, and as soon as the gateway closes the episode. On cancellation it tries to finish for up to 120 s, then re-raises.
- `SandboxEpisodeResult` holds `episode_id`, `trace`, `final`, `state` (the diff, if the task publishes state), `transcript`, `refusals` (`(turn, j, reason)`) and `error`. If `error` is set, the agent run did not complete cleanly (a framework, harness or routing failure, or the wall budget): do not submit.
- Mechanism: the program runs on the miner under verifiers' `subprocess` runtime, and a task `@vf.intercept` hook (`route_request`) answers every tool message through the `ToolRouter`. Answers are cached by `(turn, call id)`, because models reuse ids such as `call_0` across turns, and served one at a time. The system prompt and tool schemas are verifiers' bash harness's own.
- Defence in depth: the harness module patches `run_bash` and `run_edit` in the pinned program with functions that exit, so a call that ever reached local execution stops the program instead of running on the miner's host.
- The pin is enforced at import: the installed `verifiers` must come from the pinned commit (its `direct_url.json`), the bash harness module must be byte-identical to the pinned one, and the program's hash must match. Any other verifiers is refused.
- Ids: taskset and harness id `reliquary-sandbox-verifiers`.

## Why tools never run on the miner

The tools (`bash`, `edit`) are implemented in the gateway, versioned by `tools_version` in record 0, and they run in the sandbox box next to the task's files. The miner has no box, and nothing it ran locally could be signed or reproduced. The client, the renderer and the gateway's `tools.py` change together under one `tools_version`, which is why they live in this repository. verifiers' bash harness cannot be pointed at the gateway as a runtime: it runs `program.py` inside the runtime box and executes the tools there, and our boxes have no network and no model access. So the bridge runs the program on the miner and answers every tool call by interception. A miner-side runtime that re-implemented the tools would not reproduce the signed observations (see "Tools" in `deployment.md`).

## The first user message

The first user message is the env's `sandbox_prompt(split, index)`, exactly. It never goes through the sandbox, and it carries no restricted-network notice, because the subprocess runtime does not add one. The validator renders the same prompt.

## What a validator and a miner must mirror

A short checklist for the validator and the miner (calls, arguments and observations against the proven tokens):

- [ ] Pin one tool-call parser, used by miner and validator, that yields `(tool, arguments_text)` per assistant span and covers invalid JSON and missing names identically on both sides.
- [ ] Apply `plan_call(tool, arguments_text, offered=record0.tools)` to each pair on both sides; a refusal where the miner sent, or a send where the validator refuses, rejects the submission.
- [ ] A `Send` consumes the next call record and requires `record.turn == t`, `record.k == sent_positions(plans)[j]`, the same tool and canonical-equal arguments; a `Refuse` consumes no record. Reject records left over, and fewer records than sends (except on the last turn).
- [ ] Last turn: when the episode stops on context length or max turns, count the final span's sent calls but compare no observation. When the gateway closed the episode mid-turn (`episode_closed`), the records match a prefix of the final span's sent calls and the final record follows them; compare no observation for that span.
- [ ] Compare token IDs, not decoded text: `next_prompt(prompt, completion, expected_observations)` must equal the proven token IDs, with `expected_observations` from `render_observation(record["body"])` and the refusal texts, in call order. Never decode tokens to find observations.
- [ ] Fix one encoding rule for chat special-token literals in output (`<|im_end|>`, `<tool_response>`), the same on both sides, and never segment observations by searching for those tokens.
- [ ] Render the same first user message (`sandbox_prompt`), and the `reliquary-observation/1` format for `tools_version` `reliquary-tools/1`; refuse any other `tools_version` (`require_tools_version`).
- [ ] Check the call records against record 0's `tools` (`verify_transcript` does: `tool_not_offered`).
- [ ] Check that `sha256` of the submitted state equals the final record's `state_sha256`, and that the reward is the final record's.
