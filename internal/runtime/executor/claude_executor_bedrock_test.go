package executor

import (
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

	"github.com/MoeForge/moego/ojson"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

const bedrockTestModel = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

func writeBedrockTestProfile(t *testing.T, profile string) {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config")
	content := "[profile " + profile + "]\nregion = eu-central-1\naws_access_key_id = AKIDEXEC\naws_secret_access_key = exec-secret\n"
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

func writeBedrockChunk(t *testing.T, w io.Writer, event string) {
	t.Helper()
	var headers eventstream.Headers
	headers.Set(":message-type", eventstream.StringValue("event"))
	headers.Set(":event-type", eventstream.StringValue("chunk"))
	payload := []byte(`{"bytes":"` + base64.StdEncoding.EncodeToString([]byte(event)) + `"}`)
	if errEncode := eventstream.NewEncoder().Encode(w, eventstream.Message{Headers: headers, Payload: payload}); errEncode != nil {
		t.Fatal(errEncode)
	}
}

// A claude-api-key entry with a bedrock block runs through the Claude executor
// and reaches Bedrock InvokeModel with SigV4 and the Bedrock body shape.
func TestClaudeExecutorSendsBedrockCredentialsToBedrock(t *testing.T) {
	writeBedrockTestProfile(t, "claude-executor-bedrock-test")

	var (
		mu    sync.Mutex
		paths []string
		auths []string
		body  []byte
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.EscapedPath())
		auths = append(auths, r.Header.Get("Authorization"))
		body = raw
		mu.Unlock()
		if strings.HasSuffix(r.URL.Path, "/invoke-with-response-stream") {
			writeBedrockChunk(t, w, `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}`)
			writeBedrockChunk(t, w, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
			writeBedrockChunk(t, w, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello from bedrock"}}`)
			writeBedrockChunk(t, w, `{"type":"content_block_stop","index":0}`)
			writeBedrockChunk(t, w, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`)
			writeBedrockChunk(t, w, `{"type":"message_stop"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[{"type":"text","text":"hello from bedrock"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":4}}`))
	}))
	defer server.Close()

	settings := bedrock.Settings{Profile: "claude-executor-bedrock-test", Endpoint: server.URL}
	attrs := settings.Attributes()
	attrs["base_url"] = server.URL
	auth := &cliproxyauth.Auth{ID: "bedrock-auth", Provider: "claude", Attributes: attrs}
	executor := NewClaudeExecutor(&config.Config{})

	cases := []struct {
		name   string
		format sdktranslator.Format
		stream bool
		// action is the Bedrock operation the executor's upstream request maps to.
		// A translated non-stream response is still read from an upstream stream.
		action string
	}{
		{name: "claude non-stream", format: sdktranslator.FormatClaude, stream: false, action: "invoke"},
		{name: "openai non-stream", format: sdktranslator.FormatOpenAI, stream: false, action: "invoke-with-response-stream"},
		{name: "openai stream", format: sdktranslator.FormatOpenAI, stream: true, action: "invoke-with-response-stream"},
	}
	for _, tc := range cases {
		stream := tc.stream
		t.Run(tc.name, func(t *testing.T) {
			payload := []byte(`{"model":"` + bedrockTestModel + `","max_tokens":64,"stream":` + map[bool]string{false: "false", true: "true"}[stream] + `,"messages":[{"role":"user","content":"hi"}]}`)
			req := cliproxyexecutor.Request{Model: bedrockTestModel, Payload: payload}
			opts := cliproxyexecutor.Options{
				SourceFormat: tc.format, ResponseFormat: tc.format,
				Stream: stream, OriginalRequest: payload,
			}
			var text string
			if stream {
				result, errStream := executor.ExecuteStream(context.Background(), auth, req, opts)
				if errStream != nil {
					t.Fatal(errStream)
				}
				var sb strings.Builder
				for chunk := range result.Chunks {
					if chunk.Err != nil {
						t.Fatal(chunk.Err)
					}
					sb.Write(chunk.Payload)
				}
				text = sb.String()
			} else {
				resp, errExecute := executor.Execute(context.Background(), auth, req, opts)
				if errExecute != nil {
					t.Fatal(errExecute)
				}
				text = string(resp.Payload)
			}
			if !strings.Contains(text, "hello from bedrock") {
				t.Fatalf("translated response = %s", text)
			}

			mu.Lock()
			defer mu.Unlock()
			if want := "/model/eu.anthropic.claude-sonnet-4-5-20250929-v1%3A0/" + tc.action; paths[len(paths)-1] != want {
				t.Fatalf("path = %s, want %s", paths[len(paths)-1], want)
			}
			if a := auths[len(auths)-1]; !strings.Contains(a, "Credential=AKIDEXEC/") || !strings.Contains(a, "/eu-central-1/bedrock/aws4_request") {
				t.Fatalf("Authorization = %s", a)
			}
			if ojson.GetBytes(body, "model").Exists() || ojson.GetBytes(body, "anthropic_version").String() != "bedrock-2023-05-31" {
				t.Fatalf("upstream body = %s", body)
			}
		})
	}
}
