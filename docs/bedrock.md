# Amazon Bedrock

CLIProxyAPI can serve Claude models from Amazon Bedrock, signing requests with
credentials from an AWS CLI profile instead of an Anthropic API key. Every client
format the proxy accepts (Claude, OpenAI Chat/Responses, Gemini, ...) works, because
Bedrock entries reuse the Claude pipeline.

## Configuration

A Bedrock credential is a Claude API-key entry with a `bedrock` block and no
`api-key`:

```yaml
api-keys:
  claude:
    - name: bedrock-eu
      keys:
        - bedrock:
            profile: "my-sso-profile"
            region: "eu-central-1"
```

| Field      | Meaning |
|------------|---------|
| `profile`  | AWS CLI profile in `~/.aws/config`. SSO, assume-role, `credential_process` and static keys all work. Empty uses the default credential chain (`AWS_PROFILE`, environment variables, `default`). |
| `region`   | Bedrock Runtime region. Empty uses the profile's region. |
| `geo`      | Inference profile geography to discover (`eu`, `us`, `apac`, `global`, ...). Empty derives it from the region: `eu-*` gives `eu`, `us-*` gives `us`, `ap-*` gives `apac`. |
| `endpoint` | Optional Bedrock Runtime endpoint override, e.g. a VPC interface endpoint. |
| `control-endpoint` | Optional Bedrock control-plane endpoint override for discovery. |
| `betas`    | `anthropic-beta` values forwarded to Bedrock. Omit for the built-in list; `[]` forwards none. |

## Models

Without `models`, the entry discovers its models. It calls `ListInferenceProfiles`
(IAM permission `bedrock:ListInferenceProfiles`) and keeps the active Claude
profiles of its geography, so an EU entry sees only `eu.anthropic.*` profiles.
Each one is served under its Anthropic name, with the geography prefix and the
version suffix removed: `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` becomes
`claude-sonnet-4-5-20250929`. These names get the same catalog metadata (thinking
support, context size) as the Anthropic API models of that name, and clients use
them unchanged.

Discovery runs in the background when the entry is registered, and the models
appear once it finishes. The list is then refreshed every 6 hours, so profiles
AWS adds or retires appear or disappear without a restart. A client asking for a
model none of your credentials offers gets `400 unknown provider for model ...`
from the proxy before Bedrock is called.

If discovery fails, for example because the SSO session has expired, a warning is
logged and discovery is retried after 30 seconds, then at doubling intervals up to
every 10 minutes, until it succeeds. Until the first success the entry serves no
models; a failed refresh keeps serving the previous list. Run `aws sso login` and
the next retry picks it up.

To choose the models yourself, list them instead. Each `name` is a Bedrock model
or inference profile ID and `alias` the name clients send:

```yaml
      models:
        - name: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
          alias: "claude-sonnet-4-5"
```

For SSO profiles, run `aws sso login --profile <name>` before starting the proxy and
whenever the SSO session expires. Credentials are refreshed automatically while the
session is valid.

## Keeping inference in the EU

Use an EU `region`. Discovery then only serves `eu.anthropic.*` cross-region
inference profiles; with `models`, list `eu.` profile IDs. An EU inference profile
only routes requests to EU regions, and it can only be called from an EU source
region. To see the Claude profiles available to your account:

```bash
aws bedrock list-inference-profiles --region eu-central-1 --profile my-sso-profile \
  --query "inferenceProfileSummaries[?starts_with(inferenceProfileId, 'eu.anthropic.')].inferenceProfileId"
```

## How it works

The Claude executor builds an Anthropic Messages request as usual. For a Bedrock
credential its HTTP client gets the transport in `internal/bedrock`, which:

- sends `/v1/messages` to `InvokeModel`, or `InvokeModelWithResponseStream` when
  the body asks to stream, with the model ID in the URL;
- removes `model` and `stream` from the body, adds
  `anthropic_version: bedrock-2023-05-31`, and moves allowed `anthropic-beta`
  values into `anthropic_beta`;
- drops Anthropic headers and forwards only `X-Amzn-Bedrock-*` headers;
- signs the request with SigV4 (service `bedrock`);
- converts Bedrock's binary event stream back to Anthropic SSE, a mid-stream
  exception into an SSE `error` event, and Bedrock error bodies into Anthropic
  error bodies with the same status code;
- sends `/v1/messages/count_tokens` to `CountTokens` with the foundation model ID
  (the `eu.` prefix removed);
- maps a plain Anthropic model name to its discovered inference profile, and
  answers `GET /v1/models` with the discovered models.

Bedrock rejects requests that carry an `anthropic-beta` value it does not support,
so other values are dropped (logged at debug level). Bedrock also does not offer
every Anthropic API feature; a request that uses an unsupported feature, such as a
server tool Bedrock lacks, fails with Bedrock's validation error.
