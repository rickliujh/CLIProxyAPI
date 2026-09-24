package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	_ "github.com/router-for-me/CLIProxyAPI/v7/internal/translator"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func newGeminiCLITestAuth(baseURL string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{
		ID:         "gemini-cli-test-auth",
		Provider:   "gemini-cli",
		Attributes: map[string]string{"base_url": baseURL},
		Metadata: map[string]any{
			"access_token":  "token",
			"refresh_token": "refresh",
			"token_type":    "Bearer",
			"expiry":        time.Now().Add(time.Hour).Format(time.RFC3339),
			"project_id":    "project-1",
		},
	}
}

func drainStream(t *testing.T, result *cliproxyexecutor.StreamResult) (string, error) {
	t.Helper()
	var out strings.Builder
	var streamErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			streamErr = chunk.Err
			continue
		}
		out.Write(chunk.Payload)
	}
	return out.String(), streamErr
}

// A Claude request must reach Code Assist through the Antigravity translators, carry the
// Gemini CLI identity rather than Antigravity's, and send thinkingLevel as the upper-case
// enum name the Gemini CLI backend honours.
func TestGeminiCLIExecuteStream_ClaudeRequestUsesGeminiCLIIdentity(t *testing.T) {
	var upstreamBody []byte
	var upstreamHeader http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1internal:streamGenerateContent" {
			t.Errorf("path = %q, want /v1internal:streamGenerateContent", r.URL.Path)
		}
		if got := r.URL.Query().Get("alt"); got != "sse" {
			t.Errorf("alt = %q, want sse", got)
		}
		upstreamHeader = r.Header.Clone()
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4}}}\n\n"))
	}))
	defer server.Close()

	payload := []byte(`{"model":"gemini-3.5-flash","max_tokens":1024,"stream":true,"messages":[
		{"role":"user","content":[{"type":"text","text":"hi"}]},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"secret reasoning","signature":"not-a-gemini-signature"},
			{"type":"text","text":"hello"}
		]},
		{"role":"user","content":[{"type":"text","text":"go on"}]}
	]}`)
	exec := NewGeminiCLIExecutor(&config.Config{})
	result, errExecute := exec.ExecuteStream(context.Background(), newGeminiCLITestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gemini-3.5-flash(high)",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		ResponseFormat:  sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: payload,
	})
	if errExecute != nil {
		t.Fatalf("ExecuteStream() error = %v", errExecute)
	}
	out, errStream := drainStream(t, result)
	if errStream != nil {
		t.Fatalf("stream error: %v", errStream)
	}

	if ua := upstreamHeader.Get("User-Agent"); !strings.HasPrefix(ua, "GeminiCLI/") || !strings.Contains(ua, "gemini-3.5-flash") {
		t.Errorf("User-Agent = %q, want GeminiCLI/<ver>/gemini-3.5-flash", ua)
	}
	if upstreamHeader.Get("X-Goog-Api-Client") == "" {
		t.Error("X-Goog-Api-Client header missing")
	}
	if got := upstreamHeader.Get("Authorization"); got != "Bearer token" {
		t.Errorf("Authorization = %q, want Bearer token", got)
	}

	body := gjson.ParseBytes(upstreamBody)
	if got := body.Get("model").String(); got != "gemini-3.5-flash" {
		t.Errorf("model = %q, want gemini-3.5-flash", got)
	}
	if got := body.Get("project").String(); got != "project-1" {
		t.Errorf("project = %q, want project-1", got)
	}
	if body.Get("user_prompt_id").String() == "" || body.Get("request.session_id").String() == "" {
		t.Errorf("user_prompt_id and request.session_id must be set: %s", upstreamBody)
	}
	for _, path := range []string{"userAgent", "requestType", "requestId", "request.sessionId"} {
		if body.Get(path).Exists() {
			t.Errorf("Antigravity envelope field %s must not be sent: %s", path, upstreamBody)
		}
	}
	if got := body.Get("request.generationConfig.thinkingConfig.thinkingLevel").String(); got != "HIGH" {
		t.Errorf("thinkingLevel = %q, want HIGH: %s", got, upstreamBody)
	}
	// A thinking block whose signature Gemini did not issue is dropped, text and all.
	if strings.Contains(string(upstreamBody), "not-a-gemini-signature") || strings.Contains(string(upstreamBody), "secret reasoning") {
		t.Errorf("foreign thinking block must not be replayed upstream: %s", upstreamBody)
	}

	if !strings.Contains(out, `"text":"ok"`) || !strings.Contains(out, "message_stop") {
		t.Errorf("Claude stream missing text or message_stop: %s", out)
	}
}

// A stream cut off mid-body must surface an error and must not be finalised with
// message_stop, which would report a truncated answer as complete.
func TestGeminiCLIExecuteStream_TruncatedStreamIsNotFinalised(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Length", "100000")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"partial\"}]}}]}}\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Error("response writer cannot be hijacked")
			return
		}
		conn, _, errHijack := hijacker.Hijack()
		if errHijack != nil {
			t.Errorf("hijack: %v", errHijack)
			return
		}
		_ = conn.Close()
	}))
	defer server.Close()

	payload := []byte(`{"model":"gemini-3.5-flash","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	exec := NewGeminiCLIExecutor(&config.Config{})
	result, errExecute := exec.ExecuteStream(context.Background(), newGeminiCLITestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gemini-3.5-flash",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		ResponseFormat:  sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: payload,
	})
	if errExecute != nil {
		t.Fatalf("ExecuteStream() error = %v", errExecute)
	}
	out, errStream := drainStream(t, result)
	if errStream == nil {
		t.Fatalf("expected a stream error for a truncated body, got output: %s", out)
	}
	if strings.Contains(out, "message_stop") {
		t.Fatalf("truncated stream must not emit message_stop: %s", out)
	}
}

// countTokens uses the request shape gemini-cli sends: only model and contents.
func TestGeminiCLICountTokens_UsesGeminiCLIRequestShape(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1internal:countTokens" {
			t.Errorf("path = %q, want /v1internal:countTokens", r.URL.Path)
		}
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"totalTokens":42}`))
	}))
	defer server.Close()

	payload := []byte(`{"model":"gemini-3.5-flash","system":"be brief","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	exec := NewGeminiCLIExecutor(&config.Config{})
	resp, errCount := exec.CountTokens(context.Background(), newGeminiCLITestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gemini-3.5-flash",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	})
	if errCount != nil {
		t.Fatalf("CountTokens() error = %v", errCount)
	}

	body := gjson.ParseBytes(upstreamBody)
	if got := body.Get("request.model").String(); got != "models/gemini-3.5-flash" {
		t.Errorf("request.model = %q, want models/gemini-3.5-flash", got)
	}
	if got := body.Get("request.contents.0.parts.0.text").String(); got != "hi" {
		t.Errorf("request.contents.0.parts.0.text = %q, want hi: %s", got, upstreamBody)
	}
	if keys := body.Get("request").Map(); len(keys) != 2 {
		t.Errorf("request must carry only model and contents: %s", upstreamBody)
	}
	if body.Get("project").Exists() || body.Get("model").Exists() {
		t.Errorf("countTokens must not carry the generate envelope: %s", upstreamBody)
	}
	if got := gjson.GetBytes(resp.Payload, "input_tokens").Int(); got != 42 {
		t.Errorf("input_tokens = %d, want 42: %s", got, resp.Payload)
	}
}

// Without a project the request cannot be routed, so fail before calling upstream.
func TestGeminiCLIExecute_MissingProjectFailsBeforeUpstream(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	auth := newGeminiCLITestAuth(server.URL)
	delete(auth.Metadata, "project_id")
	payload := []byte(`{"model":"gemini-3.5-flash","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	_, errExecute := NewGeminiCLIExecutor(&config.Config{}).Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gemini-3.5-flash",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		ResponseFormat:  sdktranslator.FormatClaude,
		OriginalRequest: payload,
	})
	if errExecute == nil || !strings.Contains(errExecute.Error(), "project_id") {
		t.Fatalf("Execute() error = %v, want missing project_id", errExecute)
	}
	if called {
		t.Fatal("upstream must not be called without a project")
	}
}

// Claude Code ends every tool round trip with a tool_result. Antigravity sends tool
// results as model-role turns, but Code Assist rejects a Gemini CLI request that ends
// with a model turn ("Requests ending with a model turn are not supported"), so the
// Gemini CLI request must carry them as user turns, as gemini-cli itself does.
func TestGeminiCLIExecuteStream_ToolResultTurnIsSentAsUser(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1,\"totalTokenCount\":4}}}\n\n"))
	}))
	defer server.Close()

	payload := []byte(`{"model":"gemini-3.5-flash","max_tokens":1024,"stream":true,
		"tools":[{"name":"read_file","description":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"read a.txt"}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"a.txt"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"hello"}]}
		]}`)
	exec := NewGeminiCLIExecutor(&config.Config{})
	result, errExecute := exec.ExecuteStream(context.Background(), newGeminiCLITestAuth(server.URL), cliproxyexecutor.Request{
		Model:   "gemini-3.5-flash",
		Payload: payload,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatClaude,
		ResponseFormat:  sdktranslator.FormatClaude,
		Stream:          true,
		OriginalRequest: payload,
	})
	if errExecute != nil {
		t.Fatalf("ExecuteStream() error = %v", errExecute)
	}
	if _, errStream := drainStream(t, result); errStream != nil {
		t.Fatalf("stream error: %v", errStream)
	}

	contents := gjson.GetBytes(upstreamBody, "request.contents").Array()
	if len(contents) == 0 {
		t.Fatalf("no contents sent: %s", upstreamBody)
	}
	last := contents[len(contents)-1]
	if got := last.Get("role").String(); got != "user" {
		t.Fatalf("last turn role = %q, want user: %s", got, upstreamBody)
	}
	if !last.Get("parts.0.functionResponse").Exists() {
		t.Fatalf("last turn should carry the tool result: %s", upstreamBody)
	}
	for _, content := range contents {
		if content.Get("parts.0.functionResponse").Exists() && content.Get("role").String() != "user" {
			t.Fatalf("tool result turn sent with role %q: %s", content.Get("role").String(), upstreamBody)
		}
	}
}

// Whatever shape the Antigravity pipeline produces, the Gemini CLI wire request must
// never end with a model turn, and a turn carrying a tool result is always a user turn.
func TestGeminiCLIEnvelope_NeverEndsWithModelTurn(t *testing.T) {
	cases := map[string]string{
		"model text last":                 `{"request":{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"text":"hello"}]}]}}`,
		"model tool result mixed w/ text": `{"request":{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"functionCall":{"id":"c1","name":"ls","args":{}}}]},{"role":"model","parts":[{"functionResponse":{"id":"c1","name":"ls","response":{"result":"a"}}},{"text":"note"}]}]}}`,
		"model tool result only":          `{"request":{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":[{"functionCall":{"id":"c1","name":"ls","args":{}}}]},{"role":"model","parts":[{"functionResponse":{"id":"c1","name":"ls","response":{"result":"a"}}}]}]}}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			out := geminiCLIEnvelope("gemini-3.8-flash", []byte(raw), "project-1", "session-1")
			contents := gjson.GetBytes(out, "request.contents").Array()
			if got := contents[len(contents)-1].Get("role").String(); got != "user" {
				t.Fatalf("last role = %q, want user: %s", got, geminiCLIContentsShape(out))
			}
			for _, content := range contents {
				hasResponse := false
				for _, part := range content.Get("parts").Array() {
					hasResponse = hasResponse || part.Get("functionResponse").Exists()
				}
				if hasResponse && content.Get("role").String() != "user" {
					t.Fatalf("tool result turn has role %q: %s", content.Get("role").String(), geminiCLIContentsShape(out))
				}
			}
		})
	}
}

func TestGeminiCLIContentsShape_OmitsText(t *testing.T) {
	raw := []byte(`{"request":{"contents":[{"role":"user","parts":[{"text":"secret prompt"}]},{"role":"model","parts":[{"text":"x","thought":true},{"functionCall":{"name":"ls"},"thoughtSignature":"s"}]},{"role":"user","parts":[{"functionResponse":{"name":"ls"}},{"text":""}]}]}}`)
	want := "[user:text | model:thought,functionCall+sig | user:functionResponse,emptyText]"
	got := geminiCLIContentsShape(raw)
	if got != want {
		t.Fatalf("shape = %q, want %q", got, want)
	}
	if strings.Contains(got, "secret") {
		t.Fatal("shape must not leak message text")
	}
}
