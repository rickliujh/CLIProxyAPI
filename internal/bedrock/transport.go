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
	opListModels
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
	op, ok := operationForRequest(req.Method, req.URL.Path)
	if !ok {
		closeRequestBody(req)
		return nil, fmt.Errorf("bedrock: unsupported Anthropic endpoint %s %s", req.Method, req.URL.Path)
	}
	if op == opListModels {
		closeRequestBody(req)
		return t.listModels(req)
	}
	body, errRead := readRequestBody(req)
	if errRead != nil {
		return nil, errRead
	}
	doc := ojson.NewDoc(body)
	model := strings.TrimSpace(doc.Get("model").String())
	if model == "" {
		return nil, fmt.Errorf("bedrock: request has no model")
	}
	stream := doc.Get("stream").Bool()

	creds, region, errCreds := t.creds(req.Context())
	if errCreds != nil {
		return nil, errCreds
	}
	if needsModelMapping(model) {
		profileID, found, errResolve := t.resolveModel(req.Context(), creds, region, model)
		if errResolve != nil {
			return nil, errResolve
		}
		if !found {
			message := fmt.Sprintf("model %s has no %s inference profile on Amazon Bedrock in this account", model, t.geo(region))
			return syntheticResponse(req, http.StatusNotFound, anthropicErrorJSON("not_found_error", message)), nil
		}
		model = profileID
	}

	if errPrepare := t.prepareInvokeDoc(doc, req.Header); errPrepare != nil {
		return nil, errPrepare
	}
	var (
		action  string
		payload []byte
	)
	switch {
	case op == opCountTokens:
		model = foundationModelID(model)
		action = "count-tokens"
		var errPayload error
		if payload, errPayload = countTokensPayload(doc); errPayload != nil {
			return nil, errPayload
		}
	case stream:
		action = "invoke-with-response-stream"
		payload = doc.Bytes()
	default:
		action = "invoke"
		payload = doc.Bytes()
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

func operationForRequest(method, path string) (operation, bool) {
	switch path = strings.TrimRight(path, "/"); {
	case method == http.MethodPost && path == "/v1/messages":
		return opMessages, true
	case method == http.MethodPost && path == "/v1/messages/count_tokens":
		return opCountTokens, true
	case method == http.MethodGet && path == "/v1/models":
		return opListModels, true
	default:
		return 0, false
	}
}

// listModels answers GET /v1/models with the discovered Claude inference
// profiles, under their client-facing Anthropic names.
func (t *transport) listModels(req *http.Request) (*http.Response, error) {
	creds, region, errCreds := t.creds(req.Context())
	if errCreds != nil {
		return nil, errCreds
	}
	models, errDiscover := t.discover(req.Context(), creds, region, rediscoverAfter)
	if errDiscover != nil {
		return nil, errDiscover
	}
	return syntheticResponse(req, http.StatusOK, modelListResponse(models)), nil
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

// prepareInvokeDoc turns an Anthropic Messages body into a Bedrock InvokeModel
// body in place: the model moves into the URL, streaming is chosen by the
// endpoint, and the anthropic-beta header becomes the anthropic_beta field.
// These are transport framing changes; the business payload is untouched.
func (t *transport) prepareInvokeDoc(doc *ojson.Doc, header http.Header) error {
	requested := splitList(strings.Join(header.Values("Anthropic-Beta"), ","))
	for _, existing := range doc.Get("anthropic_beta").Array() {
		requested = append(requested, existing.String())
	}
	betas := t.allowedBetas(requested)
	edits := []func() error{
		func() error { return doc.Delete("model") },
		func() error { return doc.Delete("stream") },
		func() error { return doc.Set("anthropic_version", anthropicVersion) },
	}
	if len(betas) == 0 {
		edits = append(edits, func() error { return doc.Delete("anthropic_beta") })
	} else {
		edits = append(edits, func() error { return doc.Set("anthropic_beta", betas) })
	}
	for _, edit := range edits {
		if errEdit := edit(); errEdit != nil {
			return fmt.Errorf("bedrock: rewrite request body: %w", errEdit)
		}
	}
	return nil
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

// countTokensPayload wraps an InvokeModel body in the CountTokens request
// shape, which carries that body as a JSON string.
func countTokensPayload(invokeDoc *ojson.Doc) ([]byte, error) {
	if !invokeDoc.Get("max_tokens").Exists() {
		if errSet := invokeDoc.Set("max_tokens", countTokensMaxTokens); errSet != nil {
			return nil, fmt.Errorf("bedrock: rewrite count_tokens body: %w", errSet)
		}
	}
	out := ojson.NewDocString(`{}`)
	if errSet := out.Set("input.invokeModel.body", invokeDoc.String()); errSet != nil {
		return nil, fmt.Errorf("bedrock: rewrite count_tokens body: %w", errSet)
	}
	return out.Bytes(), nil
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
