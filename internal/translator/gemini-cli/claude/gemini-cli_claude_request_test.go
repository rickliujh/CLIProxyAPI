package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertClaudeRequestToCLI_ToolChoice_SpecificTool(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "json"}
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	if got := gjson.GetBytes(output, "request.toolConfig.functionCallingConfig.mode").String(); got != "ANY" {
		t.Fatalf("Expected request.toolConfig.functionCallingConfig.mode 'ANY', got '%s'", got)
	}
	allowed := gjson.GetBytes(output, "request.toolConfig.functionCallingConfig.allowedFunctionNames").Array()
	if len(allowed) != 1 || allowed[0].String() != "json" {
		t.Fatalf("Expected allowedFunctionNames ['json'], got %s", gjson.GetBytes(output, "request.toolConfig.functionCallingConfig.allowedFunctionNames").Raw)
	}
}

func TestConvertClaudeRequestToCLI_StripsClaudeCodeAttribution(t *testing.T) {
	inputJSON := []byte(`{
		"model": "claude-sonnet-4-5",
		"system": [
			{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.63.abc; cc_entrypoint=cli; cch=12345;"},
			{"type": "text", "text": "User system prompt"}
		],
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	parts := gjson.GetBytes(output, "request.systemInstruction.parts").Array()
	if len(parts) != 1 {
		t.Fatalf("Expected 1 system part after attribution strip, got %d: %s", len(parts), gjson.GetBytes(output, "request.systemInstruction.parts").Raw)
	}
	if got := parts[0].Get("text").String(); got != "User system prompt" {
		t.Fatalf("Unexpected system part: %q", got)
	}
}

func TestConvertClaudeRequestToCLI_ConvertsMessageSystemRoleToUserContent(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"system": [{"type": "text", "text": "Top-level rules"}],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]},
			{"role": "system", "content": "String mid-conversation rule"},
			{"role": "system", "content": [{"type": "text", "text": "Array mid-conversation rule"}]}
		]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	if systemContent := gjson.GetBytes(output, `request.contents.#(role=="system")`); systemContent.Exists() {
		t.Fatalf("system role should not be emitted in request.contents: %s", systemContent.Raw)
	}

	contents := gjson.GetBytes(output, "request.contents").Array()
	if len(contents) != 3 {
		t.Fatalf("Expected the user and message-level system turns in request.contents, got %d: %s", len(contents), gjson.GetBytes(output, "request.contents").Raw)
	}
	if got := contents[0].Get("role").String(); got != "user" {
		t.Fatalf("Expected first content role user, got %q", got)
	}
	if got := contents[1].Get("role").String(); got != "user" {
		t.Fatalf("Expected message-level string system content to be downgraded to user role, got %q", got)
	}
	if got := contents[1].Get("parts.0.text").String(); got != "String mid-conversation rule" {
		t.Fatalf("Unexpected string message-level system content text: %q", got)
	}
	if got := contents[2].Get("role").String(); got != "user" {
		t.Fatalf("Expected message-level array system content to be downgraded to user role, got %q", got)
	}
	if got := contents[2].Get("parts.0.text").String(); got != "Array mid-conversation rule" {
		t.Fatalf("Unexpected array message-level system content text: %q", got)
	}

	parts := gjson.GetBytes(output, "request.systemInstruction.parts").Array()
	if len(parts) != 1 {
		t.Fatalf("Expected only top-level system parts, got %d: %s", len(parts), gjson.GetBytes(output, "request.systemInstruction.parts").Raw)
	}
	if got := parts[0].Get("text").String(); got != "Top-level rules" {
		t.Fatalf("Unexpected first system part: %q", got)
	}
}

func TestConvertClaudeRequestToCLI_StructuredToolResult(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "json-call-1", "name": "json", "input": {"ok": true}}
				]
			},
			{
				"role": "user",
				"content": [
					{
						"type": "tool_result",
						"tool_use_id": "json-call-1",
						"content": [
							{"type": "text", "text": "alpha"},
							{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
						]
					}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	fr := gjson.GetBytes(output, "request.contents.1.parts.0.functionResponse")
	if !fr.Exists() {
		t.Fatalf("expected functionResponse part, contents=%s", gjson.GetBytes(output, "request.contents").Raw)
	}
	// The text block must remain structured JSON, not a double-encoded string blob.
	if got := fr.Get("response.result.text").String(); got != "alpha" {
		t.Fatalf("expected structured result text 'alpha', got result=%s", fr.Get("response.result").Raw)
	}
	// The image block must be emitted as a separate inlineData part, not embedded in result.
	img := gjson.GetBytes(output, "request.contents.1.parts.1.inlineData")
	if got := img.Get("mime_type").String(); got != "image/png" {
		t.Fatalf("expected image mime type 'image/png', got '%s'", got)
	}
	if got := img.Get("data").String(); got != "aGVsbG8=" {
		t.Fatalf("expected image data 'aGVsbG8=', got '%s'", got)
	}
}

func TestConvertClaudeRequestToCLI_StringToolResult(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "json-call-1", "name": "json", "input": {"ok": true}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "json-call-1", "content": "alpha"}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	fr := gjson.GetBytes(output, "request.contents.1.parts.0.functionResponse")
	if !fr.Exists() {
		t.Fatalf("expected functionResponse part, contents=%s", gjson.GetBytes(output, "request.contents").Raw)
	}
	// String content must not be double-encoded: result should be exactly "alpha".
	if got := fr.Get("response.result").String(); got != "alpha" {
		t.Fatalf("expected result 'alpha', got '%s' (raw=%s)", got, fr.Get("response.result").Raw)
	}
}

// Gemini 3 models reason whether or not they are asked to. When the client did not
// opt in to extended thinking, the backend must be told to withhold thought parts
// so no reasoning comes back that has no valid place in a Claude response.
func TestConvertClaudeRequestToCLI_DisablesThoughtsWhenThinkingNotRequested(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	include := gjson.GetBytes(output, "request.generationConfig.thinkingConfig.includeThoughts")
	if !include.Exists() {
		t.Fatalf("expected includeThoughts to be set, generationConfig=%s",
			gjson.GetBytes(output, "request.generationConfig").Raw)
	}
	if include.Bool() {
		t.Fatalf("expected includeThoughts=false when thinking was not requested, got true")
	}
}

func TestConvertClaudeRequestToCLI_KeepsThoughtsWhenThinkingRequested(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"thinking": {"type": "enabled", "budget_tokens": 2048},
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)

	output := ConvertClaudeRequestToCLI("gemini-3-flash-preview", inputJSON, false)

	if !gjson.GetBytes(output, "request.generationConfig.thinkingConfig.includeThoughts").Bool() {
		t.Fatalf("expected includeThoughts=true when thinking was requested, generationConfig=%s",
			gjson.GetBytes(output, "request.generationConfig").Raw)
	}
	if got := gjson.GetBytes(output, "request.generationConfig.thinkingConfig.thinkingBudget").Int(); got != 2048 {
		t.Fatalf("expected thinkingBudget 2048, got %d", got)
	}
}

// Tool ids are minted as name-<nanos>-<counter>. Recovering the function name by
// splitting the id left the generated suffix attached, so every functionResponse
// named a function that was never declared. The model then sees its whole tool
// history answered by unknown functions.
func TestConvertClaudeRequestToCLI_ToolResultNameMatchesToolUse(t *testing.T) {
	for _, toolName := range []string{"Read", "Bash", "some-hyphenated-tool"} {
		toolID := toolName + "-1785194095218119402-1"
		inputJSON := []byte(`{
			"model": "gemini-3.5-flash",
			"messages": [
				{"role":"user","content":[{"type":"text","text":"hi"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"` + toolID + `","name":"` + toolName + `","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + toolID + `","content":"ok"}]}
			]
		}`)

		out := ConvertClaudeRequestToCLI("gemini-3.5-flash", inputJSON, true)

		call := gjson.GetBytes(out, "request.contents.1.parts.0.functionCall.name").String()
		response := gjson.GetBytes(out, "request.contents.2.parts.0.functionResponse.name").String()
		if call != response {
			t.Errorf("tool %q: functionCall.name=%q but functionResponse.name=%q; the names must match or the model sees a reply from an undeclared function",
				toolName, call, response)
		}
		if response != toolName {
			t.Errorf("tool %q: functionResponse.name=%q, want %q", toolName, response, toolName)
		}
	}
}

// A tool_use_id with no matching tool_use block still has to degrade sensibly.
func TestConvertClaudeRequestToCLI_ToolResultWithoutToolUseFallsBack(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3.5-flash",
		"messages": [
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"Read-1785194095218119402-1","content":"ok"}]}
		]
	}`)

	out := ConvertClaudeRequestToCLI("gemini-3.5-flash", inputJSON, true)
	if got := gjson.GetBytes(out, "request.contents.0.parts.0.functionResponse.name").String(); got != "Read" {
		t.Errorf("fallback recovery = %q, want %q", got, "Read")
	}
}

// Gemini 3 carries reasoning state across turns in the thought signature attached
// to a function call. Anthropic tool_use blocks have nowhere to carry it, so it is
// stashed against the tool id at response time and replayed here. Sending the
// synthetic constant instead only suppresses validation and loses the state, which
// forces the model to re-reason in visible text on every turn.
func TestConvertClaudeRequestToCLI_ReplaysRealThoughtSignature(t *testing.T) {
	const model = "gemini-3.5-flash"
	req := []byte(`{"model":"` + model + `","thinking":{"type":"adaptive"},
		"tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"read it"}]}]}`)
	callChunk := []byte(`{"response":{"candidates":[{"content":{"parts":[
		{"functionCall":{"name":"Read","args":{"path":"/tmp/x"}},"thoughtSignature":"CvcBAdHtim9Xr4uPq2mKzR8wJvLb3NcQeT5yHgFdSaZxMvBnKjHgFdEwQzXcVbNmAsDfGhJkLpOiUyTrEwQzXcVbNm"}]},
		"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5},
		"modelVersion":"` + model + `","responseId":"r"}}`)

	// Response side mints the id and should stash the signature against it.
	var param any
	var sse strings.Builder
	for _, seg := range ConvertGeminiCLIResponseToClaude(context.Background(), model, req, req, callChunk, &param) {
		sse.Write(seg)
	}
	toolID := gjson.Get(sse.String()[strings.Index(sse.String(), `{"type":"content_block_start"`):], "content_block.id").String()
	if toolID == "" {
		t.Fatalf("no tool_use id minted: %s", sse.String())
	}

	// Next turn: the client echoes the tool_use back.
	follow := []byte(`{"model":"` + model + `","thinking":{"type":"adaptive"},
		"messages":[
			{"role":"user","content":[{"type":"text","text":"read it"}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"` + toolID + `","name":"Read","input":{"path":"/tmp/x"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + toolID + `","content":"body"}]}
		]}`)

	out := ConvertClaudeRequestToCLI(model, follow, true)
	got := gjson.GetBytes(out, "request.contents.1.parts.0.thoughtSignature").String()
	if got == geminiCLIClaudeThoughtSignature {
		t.Fatalf("replayed the synthetic placeholder instead of the real signature; reasoning state is lost")
	}
	if got != "CvcBAdHtim9Xr4uPq2mKzR8wJvLb3NcQeT5yHgFdSaZxMvBnKjHgFdEwQzXcVbNmAsDfGhJkLpOiUyTrEwQzXcVbNm" {
		t.Fatalf("thoughtSignature = %q, want %q", got, "CvcBAdHtim9Xr4uPq2mKzR8wJvLb3NcQeT5yHgFdSaZxMvBnKjHgFdEwQzXcVbNmAsDfGhJkLpOiUyTrEwQzXcVbNm")
	}
}

// With no cached signature the synthetic constant is still the correct fallback.
func TestConvertClaudeRequestToCLI_FallsBackToSyntheticSignature(t *testing.T) {
	req := []byte(`{"model":"gemini-3.5-flash","messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"Unknown-999-1","name":"Read","input":{}}]}
	]}`)
	out := ConvertClaudeRequestToCLI("gemini-3.5-flash", req, true)
	if got := gjson.GetBytes(out, "request.contents.0.parts.0.thoughtSignature").String(); got != geminiCLIClaudeThoughtSignature {
		t.Fatalf("fallback signature = %q, want %q", got, geminiCLIClaudeThoughtSignature)
	}
}
