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
      models:
        - name: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
          alias: "claude-sonnet-4-5"
      keys:
        - bedrock:
            profile: "my-sso-profile"
            region: "eu-central-1"
```

| Field      | Meaning |
|------------|---------|
| `profile`  | AWS CLI profile in `~/.aws/config`. SSO, assume-role, `credential_process` and static keys all work. Empty uses the default credential chain (`AWS_PROFILE`, environment variables, `default`). |
| `region`   | Bedrock Runtime region. Empty uses the profile's region. |
| `endpoint` | Optional endpoint override, e.g. a VPC interface endpoint. |
| `betas`    | `anthropic-beta` values forwarded to Bedrock. Omit for the built-in list; `[]` forwards none. |

`models` is required in practice: it maps the Bedrock model ID (`name`) to the name
clients send (`alias`). Without it the entry advertises Anthropic model names, which
Bedrock does not accept, and a warning is logged.

For SSO profiles, run `aws sso login --profile <name>` before starting the proxy and
whenever the SSO session expires. Credentials are refreshed automatically while the
session is valid.

## Keeping inference in the EU

Use the `eu.` cross-region inference profiles (for example
`eu.anthropic.claude-sonnet-4-5-20250929-v1:0`) and an EU `region`. An EU inference
profile only routes requests to EU regions, and it can only be called from an EU
source region. List the Claude profiles available to your account:

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
  (the `eu.` prefix removed).

Bedrock rejects requests that carry an `anthropic-beta` value it does not support,
so other values are dropped (logged at debug level). Bedrock also does not offer
every Anthropic API feature; a request that uses an unsupported feature, such as a
server tool Bedrock lacks, fails with Bedrock's validation error.
