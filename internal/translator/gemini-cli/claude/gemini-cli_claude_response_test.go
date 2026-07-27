package claude

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func convertStream(t *testing.T, chunks ...[]byte) string {
	t.Helper()
	requestJSON := []byte(`{"model":"gemini-test","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	ctx := context.Background()
	var param any
	var output []byte
	for _, chunk := range chunks {
		output = append(output, bytes.Join(
			ConvertGeminiCLIResponseToClaude(ctx, "gemini-test", requestJSON, requestJSON, chunk, &param), nil)...)
	}
	return string(output)
}

// A turn that spends its entire budget on thinking reports thoughtsTokenCount but
// no candidatesTokenCount. The terminal events must still be emitted, otherwise the
// client never receives a stop_reason and treats the thinking as a finished answer.
func TestConvertGeminiCLIResponseToClaude_ThinkingOnlyTurnStillFinalizes(t *testing.T) {
	thinkingChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"reasoning about it","thought":true}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)
	finishChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}],
		"usageMetadata":{"promptTokenCount":10,"thoughtsTokenCount":64,"totalTokenCount":74},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, thinkingChunk, finishChunk, []byte("[DONE]"))

	if !strings.Contains(out, `"type":"message_delta"`) {
		t.Fatalf("finish chunk without candidatesTokenCount must still emit message_delta: %s", out)
	}
	if !strings.Contains(out, `"stop_reason":"max_tokens"`) {
		t.Fatalf("MAX_TOKENS finish must map to the max_tokens stop reason: %s", out)
	}
	if !strings.Contains(out, `"output_tokens":64`) {
		t.Fatalf("thinking tokens must be counted as output tokens: %s", out)
	}
	if !strings.Contains(out, `"type":"content_block_stop"`) {
		t.Fatalf("the open thinking block must be closed: %s", out)
	}
	if !strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("DONE must emit message_stop: %s", out)
	}
}

// The terminal events must be emitted exactly once even if more than one chunk
// carries both usageMetadata and a finishReason.
func TestConvertGeminiCLIResponseToClaude_FinalEventsEmittedOnce(t *testing.T) {
	textChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"the answer"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)
	finishChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, textChunk, finishChunk, finishChunk, []byte("[DONE]"))

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

	out := convertStream(t, toolChunk, finishChunk, []byte("[DONE]"))

	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("tool call in an earlier chunk must yield the tool_use stop reason: %s", out)
	}
}

// Gemini emits the thought signature as its own part with empty text. Treating it
// as regular text closes the thinking block and opens an empty text block, so the
// turn ends with the reasoning as its only visible content and no answer.
func TestConvertGeminiCLIResponseToClaude_SignatureOnlyPartDoesNotOpenEmptyTextBlock(t *testing.T) {
	thinkingChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"reasoning about it","thought":true}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)
	signatureChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig-test"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)
	answerChunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"the answer"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":7,"totalTokenCount":22},
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, thinkingChunk, signatureChunk, answerChunk, []byte("[DONE]"))

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

// A signature arriving on the same part as thinking text must annotate that block
// rather than being dropped.
func TestConvertGeminiCLIResponseToClaude_SignatureOnThinkingPartIsEmitted(t *testing.T) {
	chunk := []byte(`{"response":{
		"candidates":[{"content":{"parts":[{"text":"reasoning","thought":true,"thoughtSignature":"sig-inline"}]}}],
		"modelVersion":"gemini-test","responseId":"resp-test"}}`)

	out := convertStream(t, chunk)

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

	out := convertStream(t, finishChunk, []byte("[DONE]"))

	if strings.Contains(out, `"type":"message_delta"`) {
		t.Fatalf("a turn with no content must not emit message_delta: %s", out)
	}
	if strings.Contains(out, `"type":"message_stop"`) {
		t.Fatalf("a turn with no content must not emit message_stop: %s", out)
	}
}
