package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
)

const testModel = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

type capturedRequest struct {
	method  string
	rawPath string
	header  http.Header
	body    []byte
}

// fakeBedrock records every request and answers with handler.
func fakeBedrock(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*httptest.Server, func() []capturedRequest) {
	t.Helper()
	var (
		mu       sync.Mutex
		requests []capturedRequest
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		requests = append(requests, capturedRequest{method: r.Method, rawPath: r.URL.EscapedPath(), header: r.Header.Clone(), body: body})
		mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	return server, func() []capturedRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]capturedRequest(nil), requests...)
	}
}

func staticCredentials(region string) credentialSource {
	return func(context.Context) (aws.Credentials, string, error) {
		return aws.Credentials{AccessKeyID: "AKIDTEST", SecretAccessKey: "secret", SessionToken: "session-token"}, region, nil
	}
}

func newTestTransport(endpoint string, settings Settings) *transport {
	settings.Endpoint = endpoint
	tr := newTransport(settings, http.DefaultTransport, staticCredentials("eu-central-1"))
	tr.now = func() time.Time { return time.Date(2026, 11, 11, 10, 0, 0, 0, time.UTC) }
	return tr
}

func anthropicRequest(t *testing.T, path, body string) *http.Request {
	t.Helper()
	req, errNew := http.NewRequest(http.MethodPost, "https://bedrock-runtime.amazonaws.com"+path, strings.NewReader(body))
	if errNew != nil {
		t.Fatal(errNew)
	}
	req.Header.Set("X-Api-Key", "should-not-leak")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	req.Header.Set("Anthropic-Beta", "claude-code-20250219,interleaved-thinking-2025-05-14")
	req.Header.Add("Anthropic-Beta", "context-1m-2025-08-07")
	req.Header.Set("X-Amzn-Bedrock-GuardrailIdentifier", "guard-1")
	return req
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	data, errRead := io.ReadAll(resp.Body)
	if errRead != nil {
		t.Fatalf("read response: %v", errRead)
	}
	return string(data)
}

func TestInvokeRewritesRequestAndSigns(t *testing.T) {
	server, requests := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}]}`))
	})
	tr := newTestTransport(server.URL, Settings{})

	req := anthropicRequest(t, "/v1/messages?beta=true", `{"model":"`+testModel+`","stream":false,"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	resp, errDo := tr.RoundTrip(req)
	if errDo != nil {
		t.Fatal(errDo)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got := readAll(t, resp); !strings.Contains(got, `"text":"hi"`) {
		t.Fatalf("response body = %s", got)
	}

	got := requests()
	if len(got) != 1 {
		t.Fatalf("requests = %d", len(got))
	}
	sent := got[0]
	if want := "/model/eu.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"; sent.rawPath != want {
		t.Fatalf("path = %s, want %s", sent.rawPath, want)
	}
	if ojson.GetBytes(sent.body, "model").Exists() || ojson.GetBytes(sent.body, "stream").Exists() {
		t.Fatalf("model/stream must move out of the body: %s", sent.body)
	}
	if v := ojson.GetBytes(sent.body, "anthropic_version").String(); v != anthropicVersion {
		t.Fatalf("anthropic_version = %q", v)
	}
	betas := ojson.GetBytes(sent.body, "anthropic_beta").Array()
	if len(betas) != 2 || betas[0].String() != "interleaved-thinking-2025-05-14" || betas[1].String() != "context-1m-2025-08-07" {
		t.Fatalf("anthropic_beta = %s", ojson.GetBytes(sent.body, "anthropic_beta").Raw)
	}
	auth := sent.header.Get("Authorization")
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIDTEST/20261111/eu-central-1/bedrock/aws4_request") {
		t.Fatalf("Authorization = %s", auth)
	}
	if sent.header.Get("X-Amz-Security-Token") != "session-token" {
		t.Fatal("session token header missing")
	}
	if sent.header.Get("X-Api-Key") != "" || sent.header.Get("Anthropic-Version") != "" || sent.header.Get("Anthropic-Beta") != "" {
		t.Fatalf("Anthropic headers leaked upstream: %v", sent.header)
	}
	if sent.header.Get("X-Amzn-Bedrock-Guardrailidentifier") != "guard-1" {
		t.Fatal("Bedrock request header was not forwarded")
	}
}

func TestConfiguredBetasReplaceDefaults(t *testing.T) {
	tr := newTestTransport("https://example.invalid", Settings{BetasSet: true, Betas: []string{"claude-code-20250219"}})
	header := http.Header{}
	header.Set("Anthropic-Beta", "claude-code-20250219,interleaved-thinking-2025-05-14")
	out, errBody := tr.invokeBody([]byte(`{"model":"m","messages":[]}`), header)
	if errBody != nil {
		t.Fatal(errBody)
	}
	if raw := ojson.GetBytes(out, "anthropic_beta").Raw; raw != `["claude-code-20250219"]` {
		t.Fatalf("anthropic_beta = %s", raw)
	}

	tr = newTestTransport("https://example.invalid", Settings{BetasSet: true})
	if out, errBody = tr.invokeBody([]byte(`{"model":"m","anthropic_beta":["x"]}`), header); errBody != nil {
		t.Fatal(errBody)
	}
	if ojson.GetBytes(out, "anthropic_beta").Exists() {
		t.Fatalf("an empty beta list must forward none: %s", out)
	}
}

func encodeEvent(t *testing.T, w io.Writer, headers map[string]string, payload []byte) {
	t.Helper()
	var hs eventstream.Headers
	for name, value := range headers {
		hs.Set(name, eventstream.StringValue(value))
	}
	if errEncode := eventstream.NewEncoder().Encode(w, eventstream.Message{Headers: hs, Payload: payload}); errEncode != nil {
		t.Fatal(errEncode)
	}
}

func chunk(t *testing.T, w io.Writer, event string) {
	t.Helper()
	payload := []byte(`{"bytes":"` + base64.StdEncoding.EncodeToString([]byte(event)) + `"}`)
	encodeEvent(t, w, map[string]string{":message-type": "event", ":event-type": "chunk", ":content-type": "application/json"}, payload)
}

func TestStreamConvertsEventStreamToSSE(t *testing.T) {
	server, requests := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		chunk(t, w, `{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[]}}`)
		chunk(t, w, "{\"type\":\"content_block_delta\",\n \"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}")
		chunk(t, w, `{"type":"message_stop","amazon-bedrock-invocationMetrics":{"inputTokenCount":3}}`)
	})
	tr := newTestTransport(server.URL, Settings{})

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages?beta=true", `{"model":"`+testModel+`","stream":true,"messages":[]}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %s", ct)
	}
	want := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\",\"amazon-bedrock-invocationMetrics\":{\"inputTokenCount\":3}}\n\n"
	if got := readAll(t, resp); got != want {
		t.Fatalf("SSE =\n%s\nwant\n%s", got, want)
	}
	if path := requests()[0].rawPath; !strings.HasSuffix(path, "/invoke-with-response-stream") {
		t.Fatalf("path = %s", path)
	}
}

func TestStreamExceptionBecomesErrorEvent(t *testing.T) {
	server, _ := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		chunk(t, w, `{"type":"message_start","message":{"id":"msg_1"}}`)
		encodeEvent(t, w, map[string]string{":message-type": "exception", ":exception-type": "throttlingException"}, []byte(`{"message":"Too many requests"}`))
		chunk(t, w, `{"type":"message_stop"}`)
	})
	tr := newTestTransport(server.URL, Settings{})

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"`+testModel+`","stream":true}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	got := readAll(t, resp)
	wantTail := "event: error\ndata: {\"error\":{\"message\":\"Too many requests\",\"type\":\"rate_limit_error\"},\"type\":\"error\"}\n\n"
	if !strings.HasSuffix(got, wantTail) || strings.Contains(got, "message_stop") {
		t.Fatalf("SSE = %s", got)
	}
}

func TestTruncatedStreamIsAnError(t *testing.T) {
	var frame bytes.Buffer
	chunk(t, &frame, `{"type":"message_start"}`)
	server, _ := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(frame.Bytes()[:frame.Len()-3])
	})
	tr := newTestTransport(server.URL, Settings{})

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"`+testModel+`","stream":true}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	defer func() { _ = resp.Body.Close() }()
	if _, errRead := io.ReadAll(resp.Body); errRead == nil {
		t.Fatal("expected a read error for a truncated event stream")
	}
}

func TestErrorResponseBecomesAnthropicError(t *testing.T) {
	server, _ := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Amzn-ErrorType", "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"prompt is too long: 300000 tokens > 200000 maximum"}`))
	})
	tr := newTestTransport(server.URL, Settings{})

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"`+testModel+`"}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	body := readAll(t, resp)
	if ojson.Get(body, "error.type").String() != "invalid_request_error" || !strings.Contains(ojson.Get(body, "error.message").String(), "prompt is too long") {
		t.Fatalf("body = %s", body)
	}
}

func TestCountTokens(t *testing.T) {
	server, requests := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"inputTokens":42}`))
	})
	tr := newTestTransport(server.URL, Settings{})

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages/count_tokens?beta=true", `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}]}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	if got := readAll(t, resp); got != `{"input_tokens":42}` {
		t.Fatalf("body = %s", got)
	}
	sent := requests()[0]
	if want := "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/count-tokens"; sent.rawPath != want {
		t.Fatalf("path = %s, want %s", sent.rawPath, want)
	}
	inner := ojson.GetBytes(sent.body, "input.invokeModel.body").String()
	if ojson.Get(inner, "model").Exists() || ojson.Get(inner, "max_tokens").Int() != countTokensMaxTokens || ojson.Get(inner, "anthropic_version").String() != anthropicVersion {
		t.Fatalf("invokeModel body = %s", inner)
	}
}

func TestUnsupportedEndpoint(t *testing.T) {
	tr := newTestTransport("https://example.invalid", Settings{})
	if _, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/models", `{}`)); errDo == nil {
		t.Fatal("expected an error for an endpoint Bedrock does not serve")
	}
}

func TestSettingsRoundTripThroughAttributes(t *testing.T) {
	in := Settings{Profile: "work", Region: "eu-west-1", BetasSet: true}
	out, ok := SettingsFromAttributes(in.Attributes())
	if !ok || out.Profile != "work" || out.Region != "eu-west-1" || !out.BetasSet || len(out.Betas) != 0 {
		t.Fatalf("settings = %+v ok=%v", out, ok)
	}
	if in.BaseURL() != "https://bedrock-runtime.eu-west-1.amazonaws.com" {
		t.Fatalf("base URL = %s", in.BaseURL())
	}
	if _, ok := SettingsFromAttributes(map[string]string{"api_key": "x"}); ok {
		t.Fatal("a plain API key must not be treated as Bedrock")
	}
	base := http.DefaultTransport
	if WrapTransport(map[string]string{"api_key": "x"}, base) != base {
		t.Fatal("non-Bedrock credentials must keep their transport")
	}
}

// writeAWSProfile points the SDK at an isolated config file with one static profile.
func writeAWSProfile(t *testing.T, profile, region string) {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config")
	content := "[profile " + profile + "]\nregion = " + region + "\naws_access_key_id = AKIDPROFILE\naws_secret_access_key = profile-secret\n"
	if errWrite := os.WriteFile(configPath, []byte(content), 0o600); errWrite != nil {
		t.Fatal(errWrite)
	}
	t.Setenv("AWS_CONFIG_FILE", configPath)
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(dir, "credentials"))
	for _, name := range []string{"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"} {
		t.Setenv(name, "")
	}
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
}

func TestWrapTransportUsesAWSProfile(t *testing.T) {
	writeAWSProfile(t, "bedrock-test-profile", "eu-west-3")
	server, requests := fakeBedrock(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"type":"message"}`))
	})
	rt := WrapTransport(Settings{Profile: "bedrock-test-profile", Endpoint: server.URL}.Attributes(), nil)

	resp, errDo := rt.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"`+testModel+`"}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	_ = readAll(t, resp)
	auth := requests()[0].header.Get("Authorization")
	if !strings.Contains(auth, "Credential=AKIDPROFILE/") || !strings.Contains(auth, "/eu-west-3/bedrock/aws4_request") {
		t.Fatalf("Authorization = %s", auth)
	}
}
