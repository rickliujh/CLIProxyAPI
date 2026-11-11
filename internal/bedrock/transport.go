package bedrock

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/MoeForge/moego/ojson"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	log "github.com/sirupsen/logrus"
)

const (
	anthropicVersion = "bedrock-2023-05-31"
	signingService   = "bedrock"
	// countTokensMaxTokens fills max_tokens on a count_tokens request, which the
	// Anthropic API does not require but a Bedrock InvokeModel body does.
	countTokensMaxTokens = 1
)

type operation int

const (
	opMessages operation = iota
	opCountTokens
)

// inferenceProfilePrefix matches the geography prefix of a cross-region inference
// profile ID such as "eu.anthropic.claude-...".
var inferenceProfilePrefix = regexp.MustCompile(`^[a-z]{2,6}\.(anthropic\.)`)

type transport struct {
	settings Settings
	base     http.RoundTripper
	creds    credentialSource
	signer   *v4.Signer
	now      func() time.Time
}

func newTransport(settings Settings, base http.RoundTripper, creds credentialSource) *transport {
	return &transport{settings: settings, base: base, creds: creds, signer: v4.NewSigner(), now: time.Now}
}

// RoundTrip implements http.RoundTripper.
func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	op, ok := operationForPath(req.URL.Path)
	if !ok {
		closeRequestBody(req)
		return nil, fmt.Errorf("bedrock: unsupported Anthropic endpoint %s", req.URL.Path)
	}
	body, errRead := readRequestBody(req)
	if errRead != nil {
		return nil, errRead
	}
	model := strings.TrimSpace(ojson.GetBytes(body, "model").String())
	if model == "" {
		return nil, fmt.Errorf("bedrock: request has no model")
	}
	stream := ojson.GetBytes(body, "stream").Bool()

	creds, region, errCreds := t.creds(req.Context())
	if errCreds != nil {
		return nil, errCreds
	}

	payload, errPayload := t.invokeBody(body, req.Header)
	if errPayload != nil {
		return nil, errPayload
	}
	var action string
	switch {
	case op == opCountTokens:
		model = foundationModelID(model)
		action = "count-tokens"
		payload, errPayload = countTokensBody(payload)
		if errPayload != nil {
			return nil, errPayload
		}
	case stream:
		action = "invoke-with-response-stream"
	default:
		action = "invoke"
	}

	outReq, errBuild := t.newUpstreamRequest(req.Context(), region, model, action, payload, req.Header)
	if errBuild != nil {
		return nil, errBuild
	}
	payloadHash := sha256.Sum256(payload)
	if errSign := t.signer.SignHTTP(req.Context(), creds, outReq, hex.EncodeToString(payloadHash[:]), signingService, region, t.now()); errSign != nil {
		return nil, fmt.Errorf("bedrock: sign request: %w", errSign)
	}

	resp, errDo := t.base.RoundTrip(outReq)
	if errDo != nil {
		return nil, errDo
	}
	resp.Request = req
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return convertErrorResponse(resp), nil
	}
	switch {
	case op == opCountTokens:
		return convertCountTokensResponse(resp), nil
	case stream:
		return convertStreamResponse(resp), nil
	default:
		return resp, nil
	}
}

func operationForPath(path string) (operation, bool) {
	switch strings.TrimRight(path, "/") {
	case "/v1/messages":
		return opMessages, true
	case "/v1/messages/count_tokens":
		return opCountTokens, true
	default:
		return 0, false
	}
}

func readRequestBody(req *http.Request) ([]byte, error) {
	if req.Body == nil {
		return nil, fmt.Errorf("bedrock: request has no body")
	}
	defer closeRequestBody(req)
	body, errRead := io.ReadAll(req.Body)
	if errRead != nil {
		return nil, fmt.Errorf("bedrock: read request body: %w", errRead)
	}
	return body, nil
}

func closeRequestBody(req *http.Request) {
	if req.Body == nil {
		return
	}
	if errClose := req.Body.Close(); errClose != nil {
		log.Debugf("bedrock: close request body: %v", errClose)
	}
}

// invokeBody turns an Anthropic Messages body into a Bedrock InvokeModel body:
// the model moves into the URL, streaming is chosen by the endpoint, and the
// anthropic-beta header becomes the anthropic_beta field.
func (t *transport) invokeBody(body []byte, header http.Header) ([]byte, error) {
	out, errDelete := ojson.DeleteBytes(body, "model")
	if errDelete != nil {
		return nil, fmt.Errorf("bedrock: rewrite request body: %w", errDelete)
	}
	if out, errDelete = ojson.DeleteBytes(out, "stream"); errDelete != nil {
		return nil, fmt.Errorf("bedrock: rewrite request body: %w", errDelete)
	}
	var errSet error
	if out, errSet = ojson.SetBytes(out, "anthropic_version", anthropicVersion); errSet != nil {
		return nil, fmt.Errorf("bedrock: rewrite request body: %w", errSet)
	}
	requested := splitList(strings.Join(header.Values("Anthropic-Beta"), ","))
	for _, existing := range ojson.GetBytes(out, "anthropic_beta").Array() {
		requested = append(requested, existing.String())
	}
	betas := t.allowedBetas(requested)
	if len(betas) == 0 {
		if out, errDelete = ojson.DeleteBytes(out, "anthropic_beta"); errDelete != nil {
			return nil, fmt.Errorf("bedrock: rewrite request body: %w", errDelete)
		}
		return out, nil
	}
	if out, errSet = ojson.SetBytes(out, "anthropic_beta", betas); errSet != nil {
		return nil, fmt.Errorf("bedrock: rewrite request body: %w", errSet)
	}
	return out, nil
}

func (t *transport) allowedBetas(requested []string) []string {
	allowList := DefaultBetas
	if t.settings.BetasSet {
		allowList = t.settings.Betas
	}
	allowed := make(map[string]struct{}, len(allowList))
	for _, beta := range allowList {
		allowed[beta] = struct{}{}
	}
	seen := make(map[string]struct{}, len(requested))
	var out, dropped []string
	for _, beta := range requested {
		beta = strings.TrimSpace(beta)
		if beta == "" {
			continue
		}
		if _, dup := seen[beta]; dup {
			continue
		}
		seen[beta] = struct{}{}
		if _, ok := allowed[beta]; ok {
			out = append(out, beta)
		} else {
			dropped = append(dropped, beta)
		}
	}
	if len(dropped) > 0 {
		log.Debugf("bedrock: dropping anthropic-beta values Bedrock does not accept: %s", strings.Join(dropped, ","))
	}
	return out
}

// countTokensBody wraps an InvokeModel body in the CountTokens request shape.
func countTokensBody(invokeBody []byte) ([]byte, error) {
	if !ojson.GetBytes(invokeBody, "max_tokens").Exists() {
		var errSet error
		if invokeBody, errSet = ojson.SetBytes(invokeBody, "max_tokens", countTokensMaxTokens); errSet != nil {
			return nil, fmt.Errorf("bedrock: rewrite count_tokens body: %w", errSet)
		}
	}
	out, errSet := ojson.SetBytes([]byte(`{}`), "input.invokeModel.body", string(invokeBody))
	if errSet != nil {
		return nil, fmt.Errorf("bedrock: rewrite count_tokens body: %w", errSet)
	}
	return out, nil
}

// foundationModelID strips a cross-region inference profile prefix, since
// CountTokens takes a foundation model ID.
func foundationModelID(model string) string {
	return inferenceProfilePrefix.ReplaceAllString(model, "$1")
}

func (t *transport) endpoint(region string) (*url.URL, error) {
	raw := t.settings.Endpoint
	if raw == "" {
		raw = "https://bedrock-runtime." + region + ".amazonaws.com"
	}
	endpoint, errParse := url.Parse(raw)
	if errParse != nil || endpoint.Scheme == "" || endpoint.Host == "" {
		return nil, fmt.Errorf("bedrock: invalid endpoint %q", raw)
	}
	return endpoint, nil
}

func (t *transport) newUpstreamRequest(ctx context.Context, region, model, action string, payload []byte, inHeader http.Header) (*http.Request, error) {
	endpoint, errEndpoint := t.endpoint(region)
	if errEndpoint != nil {
		return nil, errEndpoint
	}
	prefix := strings.TrimRight(endpoint.Path, "/")
	target := *endpoint
	target.RawQuery = ""
	target.Path = prefix + "/model/" + model + "/" + action
	// Bedrock expects the model ID as one escaped path segment, ":" included.
	target.RawPath = prefix + "/model/" + strings.ReplaceAll(url.PathEscape(model), ":", "%3A") + "/" + action

	outReq, errNew := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(payload))
	if errNew != nil {
		return nil, fmt.Errorf("bedrock: build request: %w", errNew)
	}
	outReq.Header.Set("Content-Type", "application/json")
	if action == "invoke-with-response-stream" {
		outReq.Header.Set("Accept", "application/vnd.amazon.eventstream")
		outReq.Header.Set("X-Amzn-Bedrock-Accept", "application/json")
	} else {
		outReq.Header.Set("Accept", "application/json")
	}
	// Anthropic headers mean nothing to Bedrock; only Bedrock's own request
	// headers (guardrails, trace, performance config) are forwarded.
	for name, values := range inHeader {
		if strings.HasPrefix(strings.ToLower(name), "x-amzn-bedrock-") && !strings.EqualFold(name, "X-Amzn-Bedrock-Accept") {
			for _, value := range values {
				outReq.Header.Add(name, value)
			}
		}
	}
	return outReq, nil
}
