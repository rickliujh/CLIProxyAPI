package bedrock

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/MoeForge/moego/base64"
	"github.com/MoeForge/moego/ojson"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	log "github.com/sirupsen/logrus"
)

// anthropicErrorType maps a Bedrock exception name to the Anthropic error type
// clients and the Claude executor already understand.
func anthropicErrorType(exception string, status int) string {
	switch strings.ToLower(exception) {
	case "validationexception":
		return "invalid_request_error"
	case "accessdeniedexception", "unrecognizedclientexception":
		return "permission_error"
	case "resourcenotfoundexception":
		return "not_found_error"
	case "throttlingexception", "servicequotaexceededexception", "toomanyrequestsexception":
		return "rate_limit_error"
	case "serviceunavailableexception", "modelnotreadyexception":
		return "overloaded_error"
	case "modeltimeoutexception", "modelerrorexception", "modelstreamerrorexception", "internalserverexception":
		return "api_error"
	}
	switch {
	case status == http.StatusBadRequest:
		return "invalid_request_error"
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusServiceUnavailable:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

func anthropicErrorJSON(errorType, message string) []byte {
	out, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]string{"type": errorType, "message": message},
	})
	return out
}

// exceptionName extracts the exception from "ValidationException:http://..." or
// a "com.amazon...#ValidationException" style __type value.
func exceptionName(raw string) string {
	raw = strings.TrimSpace(raw)
	if idx := strings.Index(raw, ":"); idx >= 0 {
		raw = raw[:idx]
	}
	if idx := strings.LastIndex(raw, "#"); idx >= 0 {
		raw = raw[idx+1:]
	}
	return raw
}

func replaceBody(resp *http.Response, body []byte, contentType string) {
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	resp.Header.Del("Content-Encoding")
	resp.Header.Set("Content-Type", contentType)
	resp.Header.Set("Content-Length", strconv.Itoa(len(body)))
}

// syntheticResponse answers a request without calling Bedrock.
func syntheticResponse(req *http.Request, status int, body []byte) *http.Response {
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", status, http.StatusText(status)),
		StatusCode:    status,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        http.Header{"Content-Type": {"application/json"}},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
		Request:       req,
	}
}

func closeResponseBody(resp *http.Response) {
	if errClose := resp.Body.Close(); errClose != nil {
		log.Debugf("bedrock: close response body: %v", errClose)
	}
}

// convertErrorResponse rewrites a Bedrock error body ({"message": "..."}) into
// an Anthropic error body with the same status code.
func convertErrorResponse(resp *http.Response) *http.Response {
	raw, errRead := io.ReadAll(resp.Body)
	closeResponseBody(resp)
	message := strings.TrimSpace(ojson.GetBytes(raw, "message").String())
	if message == "" {
		message = strings.TrimSpace(ojson.GetBytes(raw, "Message").String())
	}
	if message == "" {
		message = strings.TrimSpace(string(raw))
	}
	if message == "" && errRead != nil {
		message = errRead.Error()
	}
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	exception := exceptionName(resp.Header.Get("X-Amzn-Errortype"))
	if exception == "" {
		exception = exceptionName(ojson.GetBytes(raw, "__type").String())
	}
	replaceBody(resp, anthropicErrorJSON(anthropicErrorType(exception, resp.StatusCode), message), "application/json")
	return resp
}

// convertCountTokensResponse rewrites {"inputTokens": n} as {"input_tokens": n}.
func convertCountTokensResponse(resp *http.Response) *http.Response {
	raw, errRead := io.ReadAll(resp.Body)
	closeResponseBody(resp)
	if errRead != nil {
		resp.StatusCode = http.StatusBadGateway
		replaceBody(resp, anthropicErrorJSON("api_error", "bedrock: read count_tokens response: "+errRead.Error()), "application/json")
		return resp
	}
	out, _ := json.Marshal(map[string]int64{"input_tokens": ojson.GetBytes(raw, "inputTokens").Int()})
	replaceBody(resp, out, "application/json")
	return resp
}

// convertStreamResponse decodes Bedrock's binary event stream and re-emits each
// Anthropic event as the SSE frame the Anthropic API would have sent.
func convertStreamResponse(resp *http.Response) *http.Response {
	upstream := resp.Body
	reader, writer := io.Pipe()
	go func() {
		defer func() {
			if errClose := upstream.Close(); errClose != nil {
				log.Debugf("bedrock: close stream body: %v", errClose)
			}
		}()
		writer.CloseWithError(pumpEventStream(upstream, writer))
	}()
	resp.Body = reader
	resp.ContentLength = -1
	resp.Header.Del("Content-Length")
	resp.Header.Del("Content-Encoding")
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp
}

// pumpEventStream copies events until the stream ends. A nil return closes the
// SSE stream cleanly; a truncated event stream is returned as an error so the
// reader sees an unexpected end rather than a clean one.
func pumpEventStream(upstream io.Reader, out io.Writer) error {
	decoder := eventstream.NewDecoder()
	payloadBuf := make([]byte, 0, 16*1024)
	for {
		msg, errDecode := decoder.Decode(upstream, payloadBuf)
		if errDecode != nil {
			if errors.Is(errDecode, io.EOF) {
				return nil
			}
			return fmt.Errorf("bedrock: decode event stream: %w", errDecode)
		}
		frame, last := sseFrame(msg)
		if len(frame) > 0 {
			if _, errWrite := out.Write(frame); errWrite != nil {
				return errWrite
			}
		}
		if last {
			return nil
		}
	}
}

func headerString(msg eventstream.Message, name string) string {
	if value := msg.Headers.Get(name); value != nil {
		return value.String()
	}
	return ""
}

// sseFrame converts one event-stream message. last reports a terminal error.
func sseFrame(msg eventstream.Message) (frame []byte, last bool) {
	switch headerString(msg, ":message-type") {
	case "event":
		if headerString(msg, ":event-type") != "chunk" {
			return nil, false
		}
		encoded := ojson.GetBytes(msg.Payload, "bytes").String()
		data, errDecode := base64.StdEncoding.DecodeString(encoded)
		if errDecode != nil {
			return errorFrame("api_error", "bedrock: invalid stream chunk: "+errDecode.Error()), true
		}
		var compact bytes.Buffer
		if errCompact := json.Compact(&compact, data); errCompact != nil {
			return errorFrame("api_error", "bedrock: invalid stream chunk: "+errCompact.Error()), true
		}
		eventType := ojson.GetBytes(compact.Bytes(), "type").String()
		var buf bytes.Buffer
		if eventType != "" {
			buf.WriteString("event: ")
			buf.WriteString(eventType)
			buf.WriteByte('\n')
		}
		buf.WriteString("data: ")
		buf.Write(compact.Bytes())
		buf.WriteString("\n\n")
		return buf.Bytes(), false
	case "exception":
		exception := headerString(msg, ":exception-type")
		message := strings.TrimSpace(ojson.GetBytes(msg.Payload, "message").String())
		if message == "" {
			message = strings.TrimSpace(string(msg.Payload))
		}
		if message == "" {
			message = exception
		}
		return errorFrame(anthropicErrorType(exception, 0), message), true
	case "error":
		code := headerString(msg, ":error-code")
		message := headerString(msg, ":error-message")
		if message == "" {
			message = code
		}
		return errorFrame(anthropicErrorType(code, 0), message), true
	default:
		return nil, false
	}
}

func errorFrame(errorType, message string) []byte {
	return []byte("event: error\ndata: " + string(anthropicErrorJSON(errorType, message)) + "\n\n")
}
