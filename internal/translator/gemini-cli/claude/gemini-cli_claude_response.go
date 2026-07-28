// Package claude provides response translation functionality for Claude Code API compatibility.
// This package handles the conversion of backend client responses into Claude Code-compatible
// Server-Sent Events (SSE) format, implementing a sophisticated state machine that manages
// different response types including text content, thinking processes, and function calls.
// The translation ensures proper sequencing of SSE events and maintains state across
// multiple response chunks to provide a seamless streaming experience.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/cache"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Params holds parameters for response conversion and maintains state across streaming chunks.
// This structure tracks the current state of the response translation process to ensure
// proper sequencing of SSE events and transitions between different content types.
type Params struct {
	HasFirstResponse bool // Indicates if the initial message_start event has been sent
	ResponseType     int  // Current response type: 0=none, 1=content, 2=thinking, 3=function
	ResponseIndex    int  // Index counter for content blocks in the streaming response
	HasContent       bool // Tracks whether any content (text, thinking, or tool use) has been output
	SawToolCall      bool // Tracks whether any chunk in this stream emitted a tool call
	HasFinalEvents   bool // Guards against emitting the terminal events more than once

	// ThinkingRequested records whether the originating Claude request opted in to
	// extended thinking. When it did not, reasoning returned by the backend must
	// not be surfaced as thinking blocks.
	ThinkingRequested bool

	// Reverse map: sanitized Gemini function name → original Claude tool name.
	ToolNameMap map[string]string
}

// toolUseIDCounter provides a process-wide unique counter for tool use identifiers.
var toolUseIDCounter uint64

// geminiCLIPartSignature returns the thought signature attached to a part, in
// either the camelCase or snake_case spelling.
func geminiCLIPartSignature(part gjson.Result) string {
	signature := part.Get("thoughtSignature")
	if !signature.Exists() {
		signature = part.Get("thought_signature")
	}
	return strings.TrimSpace(signature.String())
}

// claudeThinkingRequested reports whether the originating Claude request opted in
// to extended thinking. The Anthropic API only returns thinking blocks when the
// client asks for them, so a client that did not opt in has no way to render one
// and will treat the reasoning as the assistant's answer.
func claudeThinkingRequested(originalRequestRawJSON []byte) bool {
	thinkingResult := gjson.GetBytes(originalRequestRawJSON, "thinking")
	if !thinkingResult.Exists() || !thinkingResult.IsObject() {
		return false
	}
	switch thinkingResult.Get("type").String() {
	case "enabled", "adaptive", "auto":
		return true
	default:
		return false
	}
}

// ConvertGeminiCLIResponseToClaude performs sophisticated streaming response format conversion.
// This function implements a complex state machine that translates backend client responses
// into Claude Code-compatible Server-Sent Events (SSE) format. It manages different response types
// and handles state transitions between content blocks, thinking processes, and function calls.
//
// Response type states: 0=none, 1=content, 2=thinking, 3=function
// The function maintains state across multiple calls to ensure proper SSE event sequencing.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response (unused in current implementation)
//   - rawJSON: The raw JSON response from the Gemini CLI API
//   - param: A pointer to a parameter object for maintaining state between calls
//
// Returns:
//   - [][]byte: A slice of bytes, each containing a Claude Code-compatible SSE payload.
func ConvertGeminiCLIResponseToClaude(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) [][]byte {
	if *param == nil {
		*param = &Params{
			HasFirstResponse:  false,
			ResponseType:      0,
			ResponseIndex:     0,
			ToolNameMap:       util.SanitizedToolNameMap(originalRequestRawJSON),
			ThinkingRequested: claudeThinkingRequested(originalRequestRawJSON),
		}
	}

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		// A message that was opened must be closed, even when the turn produced no
		// content blocks. Returning nothing here leaves the client with a stream
		// that has no stop_reason and no message_stop.
		if !(*param).(*Params).HasFirstResponse {
			return [][]byte{}
		}
		out := appendClaudeTerminalEvents(nil, (*param).(*Params), terminalStopReason((*param).(*Params), ""), gjson.Result{})
		out = translatorcommon.AppendSSEEventString(out, "message_stop", `{"type":"message_stop"}`, 3)
		return [][]byte{out}
	}

	output := make([]byte, 0, 1024)
	appendEvent := func(event, payload string) {
		output = translatorcommon.AppendSSEEventString(output, event, payload, 3)
	}
	// appendSignatureDelta attaches a thought signature to the thinking block that
	// is currently open. Signatures are meaningless outside a thinking block, so
	// this is a no-op in any other state.
	appendSignatureDelta := func(signature string) {
		if signature == "" || (*param).(*Params).ResponseType != 2 {
			return
		}
		data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":""}}`, (*param).(*Params).ResponseIndex)), "delta.signature", signature)
		appendEvent("content_block_delta", string(data))
		(*param).(*Params).HasContent = true
	}

	// Initialize the streaming session with a message_start event
	// This is only sent for the very first response chunk to establish the streaming session
	if !(*param).(*Params).HasFirstResponse {
		// Create the initial message structure with default values according to Claude Code API specification
		// This follows the Claude Code API specification for streaming message initialization
		messageStartTemplate := []byte(`{"type":"message_start","message":{"id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`)

		// Override default values with actual response metadata if available from the Gemini CLI response
		if modelVersionResult := gjson.GetBytes(rawJSON, "response.modelVersion"); modelVersionResult.Exists() {
			messageStartTemplate, _ = sjson.SetBytes(messageStartTemplate, "message.model", modelVersionResult.String())
		}
		if responseIDResult := gjson.GetBytes(rawJSON, "response.responseId"); responseIDResult.Exists() {
			messageStartTemplate, _ = sjson.SetBytes(messageStartTemplate, "message.id", responseIDResult.String())
		}
		appendEvent("message_start", string(messageStartTemplate))

		(*param).(*Params).HasFirstResponse = true
	}

	// Process the response parts array from the backend client
	// Each part can contain text content, thinking content, or function calls
	partsResult := gjson.GetBytes(rawJSON, "response.candidates.0.content.parts")
	if partsResult.IsArray() {
		partResults := partsResult.Array()
		for i := 0; i < len(partResults); i++ {
			partResult := partResults[i]

			// Extract the different types of content from each part
			partTextResult := partResult.Get("text")
			functionCallResult := partResult.Get("functionCall")
			thoughtSignatureResult := partResult.Get("thoughtSignature")
			if !thoughtSignatureResult.Exists() {
				thoughtSignatureResult = partResult.Get("thought_signature")
			}
			hasThoughtSignature := thoughtSignatureResult.Exists() && thoughtSignatureResult.String() != ""

			// A part carrying only a thought signature annotates the thinking block
			// that precedes it. Falling through would open an empty text block and
			// terminate the thinking block, leaving the turn with no visible answer.
			if hasThoughtSignature && !partTextResult.Exists() && !functionCallResult.Exists() {
				appendSignatureDelta(thoughtSignatureResult.String())
				continue
			}

			// Handle text content (both regular content and thinking)
			if partTextResult.Exists() {
				// Only the thought flag marks a part as reasoning. A thoughtSignature
				// is a multi-turn continuity token that Gemini also attaches to
				// function calls and to ordinary answer parts, so treating it as a
				// thinking marker would route the visible answer into a thinking
				// block. See https://ai.google.dev/gemini-api/docs/thinking.
				isThought := partResult.Get("thought").Bool()

				// A part with a signature but no text of its own only annotates the
				// thinking block that precedes it.
				if hasThoughtSignature && partTextResult.String() == "" {
					appendSignatureDelta(thoughtSignatureResult.String())
					continue
				}

				// An empty text part carries nothing the client can render. Opening a
				// block for it yields a message whose only content is an empty string,
				// which the client reports as "no visible output" and answers with a
				// retry nudge. Leave the turn genuinely empty instead.
				if partTextResult.String() == "" {
					continue
				}

				// The client did not ask for extended thinking, so reasoning has no
				// valid representation in the response. Drop it rather than emitting
				// a thinking block the client cannot render.
				if isThought && !(*param).(*Params).ThinkingRequested {
					continue
				}
				// Process thinking content (internal reasoning)
				if isThought {
					// Continue existing thinking block if already in thinking state
					if (*param).(*Params).ResponseType == 2 {
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":""}}`, (*param).(*Params).ResponseIndex)), "delta.thinking", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).HasContent = true
					} else {
						// Transition from another state to thinking
						// First, close any existing content block
						if (*param).(*Params).ResponseType != 0 {
							if (*param).(*Params).ResponseType == 2 {
								// output = output + "event: content_block_delta\n"
								// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
								// output = output + "\n\n\n"
							}
							appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
							(*param).(*Params).ResponseIndex++
						}

						// Start a new thinking content block
						appendEvent("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"thinking","thinking":""}}`, (*param).(*Params).ResponseIndex))
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":""}}`, (*param).(*Params).ResponseIndex)), "delta.thinking", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).ResponseType = 2 // Set state to thinking
						(*param).(*Params).HasContent = true
					}
					appendSignatureDelta(thoughtSignatureResult.String())
				} else {
					// Process regular text content (user-visible output)
					// Continue existing text block if already in content state
					if (*param).(*Params).ResponseType == 1 {
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":""}}`, (*param).(*Params).ResponseIndex)), "delta.text", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).HasContent = true
					} else {
						// Transition from another state to text content
						// First, close any existing content block
						if (*param).(*Params).ResponseType != 0 {
							if (*param).(*Params).ResponseType == 2 {
								// output = output + "event: content_block_delta\n"
								// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
								// output = output + "\n\n\n"
							}
							appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
							(*param).(*Params).ResponseIndex++
						}

						// Start a new text content block
						appendEvent("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`, (*param).(*Params).ResponseIndex))
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":""}}`, (*param).(*Params).ResponseIndex)), "delta.text", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).ResponseType = 1 // Set state to content
						(*param).(*Params).HasContent = true
					}
				}
			} else if functionCallResult.Exists() {
				// Handle function/tool calls from the AI model
				// This processes tool usage requests and formats them for Claude Code API compatibility
				(*param).(*Params).SawToolCall = true
				fcName := util.RestoreSanitizedToolName((*param).(*Params).ToolNameMap, functionCallResult.Get("name").String())

				// Handle state transitions when switching to function calls
				// Close any existing function call block first
				if (*param).(*Params).ResponseType == 3 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
					(*param).(*Params).ResponseType = 0
				}

				// Special handling for thinking state transition
				if (*param).(*Params).ResponseType == 2 {
					// output = output + "event: content_block_delta\n"
					// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
					// output = output + "\n\n\n"
				}

				// Close any other existing content block
				if (*param).(*Params).ResponseType != 0 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
				}

				// Start a new tool use content block
				// This creates the structure for a function call in Claude Code format
				// Create the tool use block with unique ID and function details
				data := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`, (*param).(*Params).ResponseIndex))
				toolUseID := util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d-%d", fcName, time.Now().UnixNano(), atomic.AddUint64(&toolUseIDCounter, 1)))
				// Gemini 3 carries reasoning state across turns in the thought
				// signature attached to a function call. Anthropic tool_use blocks
				// have nowhere to put it, so stash it against the tool id, which the
				// client does echo back, and replay it when the result returns.
				if signature := geminiCLIPartSignature(partResult); signature != "" {
					cache.CacheSignature(modelName, toolUseID, signature)
				}
				data, _ = sjson.SetBytes(data, "content_block.id", toolUseID)
				data, _ = sjson.SetBytes(data, "content_block.name", fcName)
				appendEvent("content_block_start", string(data))

				if fcArgsResult := functionCallResult.Get("args"); fcArgsResult.Exists() {
					data, _ = sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, (*param).(*Params).ResponseIndex)), "delta.partial_json", fcArgsResult.Raw)
					appendEvent("content_block_delta", string(data))
				}
				(*param).(*Params).ResponseType = 3
				(*param).(*Params).HasContent = true
			}
		}
	}

	usageResult := gjson.GetBytes(rawJSON, "response.usageMetadata")
	// Process usage metadata and finish reason when present in the response.
	// candidatesTokenCount is deliberately not required here: a turn that spends
	// its whole budget on thinking reports only thoughtsTokenCount, and gating on
	// candidatesTokenCount would drop the terminal events and leave the stream
	// without a stop_reason.
	if usageResult.Exists() && bytes.Contains(rawJSON, []byte(`"finishReason"`)) && !(*param).(*Params).HasFinalEvents {
		// The terminal events are emitted whether or not the turn produced content.
		// A turn that yields nothing still needs a stop_reason: without one the
		// client sees an unterminated stream, reports no visible output, and asks
		// the model to answer again, which is what drives reasoning into the
		// visible channel.
		finish := gjson.GetBytes(rawJSON, "response.candidates.0.finishReason").String()
		output = appendClaudeTerminalEvents(output, (*param).(*Params), terminalStopReason((*param).(*Params), finish), usageResult)
	}

	return [][]byte{output}
}

// terminalStopReason maps the stream state and the upstream finish reason onto the
// Anthropic stop_reason vocabulary.
func terminalStopReason(p *Params, finishReason string) string {
	switch {
	case p.SawToolCall:
		return "tool_use"
	case finishReason == "MAX_TOKENS":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

// appendClaudeTerminalEvents closes any open content block and emits the
// message_delta carrying the stop reason. It is a no-op once the terminal events
// have already been sent, so the [DONE] path can safely call it as a backstop for
// streams whose last chunk carried no usage metadata.
func appendClaudeTerminalEvents(output []byte, p *Params, stopReason string, usage gjson.Result) []byte {
	if p == nil || p.HasFinalEvents {
		return output
	}
	if p.ResponseType != 0 {
		output = translatorcommon.AppendSSEEventString(output, "content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, p.ResponseIndex), 3)
		p.ResponseType = 0
	}
	template := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
	template, _ = sjson.SetBytes(template, "delta.stop_reason", stopReason)
	if usage.Exists() {
		// Thinking tokens count toward output even when no thought part was surfaced.
		template, _ = sjson.SetBytes(template, "usage.output_tokens", usage.Get("candidatesTokenCount").Int()+usage.Get("thoughtsTokenCount").Int())
		template, _ = sjson.SetBytes(template, "usage.input_tokens", usage.Get("promptTokenCount").Int())
	}
	output = translatorcommon.AppendSSEEventString(output, "message_delta", string(template), 3)
	p.HasFinalEvents = true
	return output
}

// ConvertGeminiCLIResponseToClaudeNonStream converts a non-streaming Gemini CLI response to a non-streaming Claude response.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the Gemini CLI API.
//   - param: A pointer to a parameter object for the conversion.
//
// Returns:
//   - []byte: A Claude-compatible JSON response.
func ConvertGeminiCLIResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []byte {
	toolNameMap := util.SanitizedToolNameMap(originalRequestRawJSON)
	thinkingRequested := claudeThinkingRequested(originalRequestRawJSON)
	_ = requestRawJSON

	root := gjson.ParseBytes(rawJSON)

	out := []byte(`{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`)
	out, _ = sjson.SetBytes(out, "id", root.Get("response.responseId").String())
	out, _ = sjson.SetBytes(out, "model", root.Get("response.modelVersion").String())

	inputTokens := root.Get("response.usageMetadata.promptTokenCount").Int()
	outputTokens := root.Get("response.usageMetadata.candidatesTokenCount").Int() + root.Get("response.usageMetadata.thoughtsTokenCount").Int()
	out, _ = sjson.SetBytes(out, "usage.input_tokens", inputTokens)
	out, _ = sjson.SetBytes(out, "usage.output_tokens", outputTokens)

	parts := root.Get("response.candidates.0.content.parts")
	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
	toolIDCounter := 0
	hasToolCall := false

	flushText := func() {
		if textBuilder.Len() == 0 {
			return
		}
		block := []byte(`{"type":"text","text":""}`)
		block, _ = sjson.SetBytes(block, "text", textBuilder.String())
		out, _ = sjson.SetRawBytes(out, "content.-1", block)
		textBuilder.Reset()
	}

	flushThinking := func() {
		if thinkingBuilder.Len() == 0 {
			return
		}
		block := []byte(`{"type":"thinking","thinking":""}`)
		block, _ = sjson.SetBytes(block, "thinking", thinkingBuilder.String())
		out, _ = sjson.SetRawBytes(out, "content.-1", block)
		thinkingBuilder.Reset()
	}

	if parts.IsArray() {
		for _, part := range parts.Array() {
			if text := part.Get("text"); text.Exists() && text.String() != "" {
				if part.Get("thought").Bool() {
					// Drop reasoning when the client did not opt in to thinking.
					if !thinkingRequested {
						continue
					}
					flushText()
					thinkingBuilder.WriteString(text.String())
					continue
				}
				flushThinking()
				textBuilder.WriteString(text.String())
				continue
			}

			if functionCall := part.Get("functionCall"); functionCall.Exists() {
				flushThinking()
				flushText()
				hasToolCall = true

				name := util.RestoreSanitizedToolName(toolNameMap, functionCall.Get("name").String())
				toolIDCounter++
				toolBlock := []byte(`{"type":"tool_use","id":"","name":"","input":{}}`)
				toolBlock, _ = sjson.SetBytes(toolBlock, "id", fmt.Sprintf("tool_%d", toolIDCounter))
				toolBlock, _ = sjson.SetBytes(toolBlock, "name", name)
				inputRaw := "{}"
				if args := functionCall.Get("args"); args.Exists() && gjson.Valid(args.Raw) && args.IsObject() {
					inputRaw = args.Raw
				}
				toolBlock, _ = sjson.SetRawBytes(toolBlock, "input", []byte(inputRaw))
				out, _ = sjson.SetRawBytes(out, "content.-1", toolBlock)
				continue
			}
		}
	}

	flushThinking()
	flushText()

	stopReason := "end_turn"
	if hasToolCall {
		stopReason = "tool_use"
	} else {
		if finish := root.Get("response.candidates.0.finishReason"); finish.Exists() {
			switch finish.String() {
			case "MAX_TOKENS":
				stopReason = "max_tokens"
			case "STOP", "FINISH_REASON_UNSPECIFIED", "UNKNOWN":
				stopReason = "end_turn"
			default:
				stopReason = "end_turn"
			}
		}
	}
	out, _ = sjson.SetBytes(out, "stop_reason", stopReason)

	if inputTokens == int64(0) && outputTokens == int64(0) && !root.Get("response.usageMetadata").Exists() {
		out, _ = sjson.DeleteBytes(out, "usage")
	}

	return out
}

func ClaudeTokenCount(ctx context.Context, count int64) []byte {
	return translatorcommon.ClaudeInputTokensJSON(count)
}
