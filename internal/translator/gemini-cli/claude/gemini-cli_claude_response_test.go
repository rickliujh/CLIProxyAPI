package claude

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

// requestWithThinking is a Claude request that opted in to extended thinking.
var requestWithThinking = []byte(`{"model":"gemini-test","thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

// requestWithoutThinking is a Claude request that did not opt in, which is what
// a client sends when extended thinking is off.
var requestWithoutThinking = []byte(`{"model":"gemini-test","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

func convertStream(t *testing.T, requestJSON []byte, chunks ...[]byte) string {
	t.Helper()
	ctx := context.Background()
	var param any
	var output []byte
	for _, chunk := range chunks {
		output = append(output, bytes.Join(
			ConvertGeminiCLIResponseToClaude(ctx, "gemini-test", requestJSON, requestJSON, chunk, &param), nil)...)
	}
	return string(output)
}

const (
	thinkingChunk = `{"response":{
		"candidates":[{"content":{"parts":[{"text":"reasoning about it","thought":true}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`
	answerChunk = `{"response":{
		"candidates":[{"content":{"parts":[{"text":"the answer"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":7,"totalTokenCount":22},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`
)

// The Anthropic API only returns thinking blocks when the client opts in. A client
// with thinking disabled cannot render one and shows the reasoning as the answer,
// so reasoning returned by the backend must be dropped.
func TestConvertGeminiCLIResponseToClaude_ThinkingSuppressedWhenNotRequested(t *testing.T) {
	out := convertStream(t, requestWithoutThinking, []byte(thinkingChunk), []byte(answerChunk), []byte("[DONE]"))

	if strings.Contains(out, `"type":"thinking"`) || strings.Contains(out, `"thinking_delta"`) {
		t.Fatalf("thinking must not be emitted when the client did not request it: %s", out)
	}
	if !strings.Contains(out, `"text_delta","text":"the answer"`) {
		t.Fatalf("the visible answer must still be delivered: %s", out)
	}
	if !strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Fatalf("the turn must still be finalized: %s", out)
	}
	if !strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("DONE must emit message_stop: %s", out)
	}
}

// A signature-only part must also stay invisible when thinking was not requested.
func TestConvertGeminiCLIResponseToClaude_SignatureSuppressedWhenNotRequested(t *testing.T) {
	signatureChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig-test"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithoutThinking, []byte(thinkingChunk), signatureChunk, []byte(answerChunk), []byte("[DONE]"))

	if strings.Contains(out, `"signature_delta"`) || strings.Contains(out, `sig-test`) {
		t.Fatalf("signature must not be emitted when thinking was not requested: %s", out)
	}
	if got := strings.Count(out, `"content_block":{"type":"text"`); got != 1 {
		t.Fatalf("expected exactly one text block, got %d: %s", got, out)
	}
}

// When the client did opt in, reasoning is surfaced as a thinking block.
func TestConvertGeminiCLIResponseToClaude_ThinkingEmittedWhenRequested(t *testing.T) {
	out := convertStream(t, requestWithThinking, []byte(thinkingChunk), []byte(answerChunk), []byte("[DONE]"))

	if !strings.Contains(out, `"content_block":{"type":"thinking"`) {
		t.Fatalf("thinking must be emitted when the client requested it: %s", out)
	}
	if !strings.Contains(out, `"thinking_delta"`) {
		t.Fatalf("expected a thinking delta: %s", out)
	}
	if !strings.Contains(out, `"text_delta","text":"the answer"`) {
		t.Fatalf("the visible answer must follow the thinking block: %s", out)
	}
}

// A turn that spends its entire budget on thinking reports thoughtsTokenCount but
// no candidatesTokenCount. The terminal events must still be emitted, otherwise the
// client never receives a stop_reason and treats the thinking as a finished answer.
func TestConvertGeminiCLIResponseToClaude_ThinkingOnlyTurnStillFinalizes(t *testing.T) {
	finishChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}],
		"usageMetadata":{"promptTokenCount":10,"thoughtsTokenCount":64,"totalTokenCount":74},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithThinking, []byte(thinkingChunk), finishChunk, []byte("[DONE]"))

	if !strings.Contains(out, `"type":"message_delta"`) {
		t.Fatalf("finish chunk without candidatesTokenCount must still emit message_delta: %s", out)
	}
	if !strings.Contains(out, `"stop_reason":"max_tokens"`) {
		t.Fatalf("MAX_TOKENS finish must map to the max_tokens stop reason: %s", out)
	}
	if !strings.Contains(out, `"output_tokens":64`) {
		t.Fatalf("thinking tokens must be counted as output tokens: %s", out)
	}
	if !strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("DONE must emit message_stop: %s", out)
	}
}

// The terminal events must be emitted exactly once even if more than one chunk
// carries both usageMetadata and a finishReason.
func TestConvertGeminiCLIResponseToClaude_FinalEventsEmittedOnce(t *testing.T) {
	out := convertStream(t, requestWithoutThinking, []byte(answerChunk), []byte(answerChunk), []byte("[DONE]"))

	if got := strings.Count(out, `"type":"message_delta"`); got != 1 {
		t.Fatalf("expected exactly one message_delta, got %d: %s", got, out)
	}
	if got := strings.Count(out, `"type":"content_block_stop"`); got != 1 {
		t.Fatalf("expected exactly one content_block_stop, got %d: %s", got, out)
	}
}

// A tool call seen in an earlier chunk must still drive the stop reason when the
// finishReason arrives in a later chunk.
func TestConvertGeminiCLIResponseToClaude_ToolCallAcrossChunksSetsStopReason(t *testing.T) {
	toolChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"London"}}}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)
	finishChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithoutThinking, toolChunk, finishChunk, []byte("[DONE]"))

	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("tool call in an earlier chunk must yield the tool_use stop reason: %s", out)
	}
}

// Gemini emits the thought signature as its own part with empty text. Treating it
// as regular text closes the thinking block and opens an empty text block, so the
// turn ends with the reasoning as its only visible content and no answer.
func TestConvertGeminiCLIResponseToClaude_SignatureOnlyPartDoesNotOpenEmptyTextBlock(t *testing.T) {
	signatureChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig-test"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithThinking, []byte(thinkingChunk), signatureChunk, []byte(answerChunk), []byte("[DONE]"))

	if !strings.Contains(out, `"type":"signature_delta"`) || !strings.Contains(out, `"signature":"sig-test"`) {
		t.Fatalf("signature-only part must be emitted as a signature_delta: %s", out)
	}
	if got := strings.Count(out, `"content_block":{"type":"text"`); got != 1 {
		t.Fatalf("expected exactly one text block, got %d: %s", got, out)
	}
	if got := strings.Count(out, `"content_block":{"type":"thinking"`); got != 1 {
		t.Fatalf("expected exactly one thinking block, got %d: %s", got, out)
	}
	if !strings.Contains(out, `"text_delta","text":"the answer"`) {
		t.Fatalf("the visible answer must survive the signature part: %s", out)
	}
}

// thoughtSignature is a multi-turn continuity token, not a thinking marker: Gemini
// attaches it to function calls and to ordinary answer parts too. Treating it as a
// marker routes the visible answer into a thinking block, which is precisely the
// "reasoning shown instead of the answer" failure.
// See https://ai.google.dev/gemini-api/docs/thinking.
func TestConvertGeminiCLIResponseToClaude_SignedAnswerPartStaysVisibleText(t *testing.T) {
	chunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"the visible answer","thoughtSignature":"sig-x"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	// Thinking enabled is the strictest case: the signature is meaningful here, so
	// the part must still be classified by its thought flag alone.
	out := convertStream(t, requestWithThinking, chunk, []byte("[DONE]"))

	if strings.Contains(out, `"thinking_delta","thinking":"the visible answer"`) {
		t.Fatalf("a signed answer part must not be routed into a thinking block: %s", out)
	}
	if !strings.Contains(out, `"text_delta","text":"the visible answer"`) {
		t.Fatalf("a signed answer part must be delivered as visible text: %s", out)
	}
	if strings.Contains(out, `"content_block":{"type":"thinking"`) {
		t.Fatalf("no thinking block should be opened for a non-thought part: %s", out)
	}
}

// A signature arriving on the same part as thinking text must annotate that block
// rather than being dropped.
func TestConvertGeminiCLIResponseToClaude_SignatureOnThinkingPartIsEmitted(t *testing.T) {
	chunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"reasoning","thought":true,"thoughtSignature":"sig-inline"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithThinking, chunk)

	if !strings.Contains(out, `"content_block":{"type":"thinking"`) {
		t.Fatalf("expected a thinking block: %s", out)
	}
	if !strings.Contains(out, `"signature":"sig-inline"`) {
		t.Fatalf("inline thought signature must be emitted: %s", out)
	}
}

// A turn that produced no content at all must not emit terminal events, so an
// empty response is not reported to the client as a completed message.
func TestConvertGeminiCLIResponseToClaude_NoContentEmitsNoFinalEvents(t *testing.T) {
	finishChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"totalTokenCount":10},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, requestWithoutThinking, finishChunk, []byte("[DONE]"))

	if strings.Contains(out, `"type":"message_delta"`) {
		t.Fatalf("a turn with no content must not emit message_delta: %s", out)
	}
	if strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("a turn with no content must not emit message_stop: %s", out)
	}
}

// The non-streaming path must apply the same rule.
func TestConvertGeminiCLIResponseToClaudeNonStream_ThinkingSuppressedWhenNotRequested(t *testing.T) {
	response := []byte(`{"response":{
		"candidates":[{"content":{"parts":[
			{"text":"reasoning about it","thought":true},
			{"text":"the answer"}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":7},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	var param any
	out := string(ConvertGeminiCLIResponseToClaudeNonStream(
		context.Background(), "gemini-test", requestWithoutThinking, requestWithoutThinking, response, &param))

	if strings.Contains(out, `"type":"thinking"`) {
		t.Fatalf("thinking must not appear when the client did not request it: %s", out)
	}
	if !strings.Contains(out, `"the answer"`) {
		t.Fatalf("the visible answer must be preserved: %s", out)
	}

	out = string(ConvertGeminiCLIResponseToClaudeNonStream(
		context.Background(), "gemini-test", requestWithThinking, requestWithThinking, response, &param))
	if !strings.Contains(out, `"type":"thinking"`) {
		t.Fatalf("thinking must appear when the client requested it: %s", out)
	}
}

// The shape observed from cloudcode-pa for a tool-calling turn: an empty-text
// part carrying a thought signature, then a functionCall with finishReason STOP.
// It must translate into a complete Anthropic tool_use turn.
func TestConvertGeminiCLIResponseToClaude_ObservedToolCallStream(t *testing.T) {
	req := []byte(`{"model":"gemini-3.5-flash","thinking":{"type":"adaptive"},
		"tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}}],
		"messages":[{"role":"user","content":[{"type":"text","text":"read the file"}]}]}`)
	signatureChunk := []byte(`{"response":{"candidates":[{"content":{"parts":[
		{"text":"","thoughtSignature":"sig-abc"}]}}],
		"modelVersion":"gemini-3.5-flash","responseId":"r"}}`)
	callChunk := []byte(`{"response":{"candidates":[{"content":{"parts":[
		{"functionCall":{"name":"Read","args":{"path":"/tmp/x"}}}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":220056,"candidatesTokenCount":39},
		"modelVersion":"gemini-3.5-flash","responseId":"r"}}`)

	out := convertStream(t, req, signatureChunk, callChunk, []byte("[DONE]"))

	for _, want := range []string{
		`"content_block":{"type":"tool_use"`,
		`"name":"Read"`,
		`"type":"input_json_delta"`,
		`"stop_reason":"tool_use"`,
		`"type":"message_stop"`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %s in translated stream: %s", want, out)
		}
	}
	// The signature carrier must not become a visible text block.
	if strings.Contains(out, `"content_block":{"type":"text"`) {
		t.Fatalf("signature carrier opened a text block: %s", out)
	}
}
