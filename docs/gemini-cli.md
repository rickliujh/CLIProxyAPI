# Gemini CLI provider: design and pitfalls

Notes for agents working on the `gemini-cli` provider. Read this before changing
`internal/runtime/executor/gemini_cli_executor*.go`, the Antigravity translators,
or anything that shapes Code Assist requests.

## Design in one paragraph

Gemini CLI and Antigravity call the same backend: the Cloud Code Assist API
(`cloudcode-pa.googleapis.com/v1internal:{streamGenerateContent,generateContent,countTokens}`)
with the same `{model, project, request{contents, ...}}` envelope and the same
`{response{candidates, usageMetadata}}` SSE shape. Upstream removed Gemini CLI in
`78ba8ba7` and kept hardening Antigravity. This branch therefore does **not** carry
its own Gemini CLI translators. `GeminiCLIExecutor` translates to the `antigravity`
format and reuses Antigravity's signature validation, sanitizing, reasoning replay
cache, boundary-turn handling and response translators. It only differs in client
identity, and it fixes up the wire payload in `geminiCLIEnvelope` just before sending.

| | gemini-cli | antigravity |
|---|---|---|
| OAuth client | `681255809395-...` (`internal/auth/gemini`) | `1071006060591-...` |
| Host | `cloudcode-pa` (prod) | `daily-cloudcode-pa` by default |
| Headers | `User-Agent: GeminiCLI/<ver>/<model> (...)`, `X-Goog-Api-Client` | `User-Agent: antigravity/...` |
| Envelope extras | `user_prompt_id`, `request.session_id` | `userAgent`, `requestType`, `requestId`, `request.sessionId` |
| Thinking applier | `internal/thinking/provider/geminicli` | `internal/thinking/provider/antigravity` |
| countTokens body | `{"request":{"model":"models/<m>","contents":[...]}}` | full request minus envelope fields |

The gemini-cli envelope fields were checked against Google's gemini-cli source
(`packages/core/src/code_assist/converter.ts`).

## Code Assist rules for the Gemini CLI identity

The Antigravity backend accepts request shapes that Code Assist rejects (or silently
ignores) for the Gemini CLI identity. Each rule below is enforced in the gemini-cli
executor and has a regression test in `gemini_cli_executor_test.go`.

1. **`thinkingLevel` must be the upper-case proto enum name** (`HIGH`, not `high`).
   A lower-case value is silently dropped: no error, just `thoughtsTokenCount=0`.
   That is why thinking is applied with the `gemini-cli` applier (format
   `"gemini-cli"`), not the Antigravity one, even though the payload is in
   Antigravity format. Found on the old branch (`9d50e66d` on `feat/restore-gemini-cli`).
2. **Tool results are user turns.** Since `72886356` Antigravity rewrites
   functionResponse-only turns to `role: "model"`, and
   `EnsureGeminiTrailingUserContent` deliberately appends nothing after a
   functionResponse. Every request after a tool call then ends with a model turn.
   The real Gemini CLI sends tool results as `role: "user"`.
3. **Text must come before tool results in the same turn.** Claude Code follows a
   `tool_result` with a mid-conversation `system` message, which becomes a text part
   in the same user turn. Code Assist rejects `[functionResponse, text]` with
   `400 Requests ending with a model turn are not supported`, **even though the last
   turn is a user turn**. This is upstream issue #5607, fixed for the Gemini
   translator by `ReorderGeminiUserParts` (`4fde97f4`). Antigravity orders tool
   results first, which its own backend accepts.
4. **The request must end with a user turn.** After rules 2 and 3,
   `geminiCLIEnvelope` also runs `EnsureGeminiTrailingUserContent`, so this holds
   whatever the pipeline produced.

Rules 2 to 4 are applied to the wire payload only. `plan.requestPayload`, which
the reasoning replay accumulator records, keeps the Antigravity form, so replay
context fingerprints still match on the next turn. Do not move these fix-ups
earlier in the pipeline.

## How the model-turn 400 was debugged (September 2026)

Worth reading because each step's assumption was wrong in a new way:

1. The first report came minutes after a fix. The obvious theory was a stale build.
   The field log proved otherwise: it contained a log line that only the new
   executor emits.
2. Local reproductions passed. They used simplified Claude requests and fake Gemini
   `functionCall`s **without an `id`**. Without `functionCall.id` the Antigravity
   response translator does not mint reserved `cpa_gemini_*` tool IDs, so the
   reasoning replay path never runs. Fake upstream responses in tests must include
   `functionCall.id` to exercise the real path.
3. Even with realistic multi-turn reproductions passing, the cause was only found
   from structure logs of the real request. The executor logs these at debug level
   (roles and part kinds, never message text):
   - `inbound claude messages shape` is what Claude Code sent,
   - `pipeline contents shape` is after the Antigravity pipeline and replay,
   - `sent contents shape` is what went to Code Assist,
   - `rejected request contents shape` is logged again on a 400.

   The decisive line was `sent: [... | model:functionCall+sig | user:functionResponse,text]`,
   which pointed at rule 3.

Lesson: when Code Assist returns a 400 that the code "cannot produce", ask for the
debug shape lines before guessing. Code Assist error messages can name the wrong
cause.

## Other things to keep

- **Model list:** `internal/registry/models/models.json` → `gemini-cli` must keep
  `gemini-3.8-flash` (Gemini CLI's `LATEST_GEMINI_FLASH_MODEL` on Code Assist) and
  `gemini-3.5-flash`. The remote catalog has no `gemini-cli` section. Only
  `mergeEmbeddedExtras` in `internal/registry/model_updater.go` keeps these models
  served after a remote refresh, so do not remove it.
- **Stream termination:** `[DONE]` is translated only after a clean end of stream
  (`scanner.Err() == nil`). Otherwise a truncated stream would be reported to Claude
  clients as a completed message.
- **Missing project:** requests fail with 400 before calling upstream when the
  credential has no `project_id`. Multi-project logins are synthesized into virtual
  auths (`internal/runtime/geminicli`, `internal/watcher/synthesizer/file.go`).
- **Endpoint override:** a credential may set `base_url` (attribute or metadata),
  which the tests use to point at `httptest` servers.

## Deliberately not carried over from `feat/restore-gemini-cli`

- The Gemini CLI translator packages (`internal/translator/gemini-cli/*` and the
  `*/gemini-cli` source-format packages). They are replaced by the Antigravity
  translators.
- The inbound `/v1internal:method` endpoint (`enable-gemini-cli-endpoint`), which lets
  the Gemini CLI tool itself use the proxy. It is a separate feature from using
  Gemini CLI accounts.
- Grouped round-robin (rotating across credentials before their projects). Upstream
  rewrote the scheduler; virtual project auths now rotate flat.
- The empty-turn retry from `78dcae19`. Antigravity has no equivalent. Re-add it
  only if blank turns are observed again.

## Where to look

- `internal/runtime/executor/gemini_cli_executor.go`: identity, token refresh, project,
  `geminiCLIEnvelope` and the wire fix-ups, shape helpers.
- `internal/runtime/executor/gemini_cli_executor_execute.go`: the `planRequest`
  pipeline, Execute, ExecuteStream, CountTokens.
- `internal/runtime/executor/antigravity_*.go`: the shared pipeline pieces being reused.
- `internal/translator/common/gemini.go`: `ReorderGeminiUserParts`.
