package executor

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// geminiCLIRequestPlan is a request translated and prepared for Cloud Code Assist.
type geminiCLIRequestPlan struct {
	token           string
	from            sdktranslator.Format
	to              sdktranslator.Format
	responseFormat  sdktranslator.Format
	originalPayload []byte
	translated      []byte
	requestPayload  []byte
	replayScope     antigravityReasoningReplayScope
	sessionID       string
}

// planRequest runs the Antigravity request pipeline: signature validation, translation,
// thinking, payload rules, signature sanitizing and reasoning replay. Thinking is applied
// with the gemini-cli applier, which renders thinkingLevel as the upper-case proto enum
// name that the Gemini CLI identity requires.
func (e *GeminiCLIExecutor) planRequest(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, stream bool, reporter *helps.UsageReporter) (geminiCLIRequestPlan, error) {
	plan := geminiCLIRequestPlan{
		from:           opts.SourceFormat,
		to:             sdktranslator.FormatAntigravity,
		responseFormat: cliproxyexecutor.ResponseFormatOrSource(opts),
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	originalPayload := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayload = opts.OriginalRequest
	}
	if plan.from == sdktranslator.FormatClaude {
		log.Debugf("gemini-cli executor: inbound claude messages shape: %s", geminiCLIClaudeMessagesShape(originalPayload))
	}
	originalPayload, errValidate := validateAntigravityRequestSignatures(ctx, baseModel, plan.from, originalPayload)
	if errValidate != nil {
		return plan, errValidate
	}
	req.Payload = originalPayload
	plan.originalPayload = originalPayload

	token, errToken := e.accessToken(ctx, auth)
	if errToken != nil {
		return plan, errToken
	}
	plan.token = token
	reporter.UpdateAccessTokenFingerprint(auth)

	modelInfo, _ := cliproxyauth.ResolvedModelInfo(req)
	translationReq := sdktranslator.RequestEnvelope{Format: plan.from, Model: baseModel, Stream: stream, ModelInfo: modelInfo}
	originalTranslated, translated := helps.TranslateRequestEnvelopePairWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, plan.from, plan.to, translationReq, originalPayload, req.Payload)

	translated, errThinking := helps.ApplyRequestThinking(translated, req, opts, plan.from.String(), geminiCLIAuthType, e.Identifier())
	if errThinking != nil {
		return plan, errThinking
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	translated = helps.ApplyPayloadConfigWithRequest(e.cfg, baseModel, "gemini", plan.from.String(), "request", translated, originalTranslated, requestedModel, requestPath, opts.Headers)
	translated = sanitizeAntigravityGeminiRequestSignatures(baseModel, translated)
	translated, _ = sjson.DeleteBytes(translated, "request.stream")
	reporter.SetTranslatedReasoningEffort(translated, plan.to.String())
	plan.translated = translated

	requestPayload := translated
	if antigravityUsesReasoningReplayCache(baseModel) {
		var errReplay error
		requestPayload, plan.replayScope, errReplay = prepareAntigravityGeminiReasoningReplayPayload(ctx, baseModel, req, opts, requestPayload)
		if errReplay != nil {
			return plan, errReplay
		}
	}
	plan.requestPayload = ensureAntigravityGeminiBoundaryUserContent(baseModel, requestPayload)
	plan.sessionID = helps.DerivedAntigravitySessionID(opts.Metadata, req.Metadata)
	return plan, nil
}

func (e *GeminiCLIExecutor) resolveWebSearchGroundingURLs(ctx context.Context, auth *cliproxyauth.Auth, plan geminiCLIRequestPlan, responseRawJSON []byte) []byte {
	if !shouldResolveAntigravityWebSearchGroundingURLs(plan.from, plan.originalPayload, plan.translated) {
		return responseRawJSON
	}
	return helps.ResolveAntigravityGroundingURLs(ctx, e.cfg, auth, responseRawJSON)
}

// handleErrorResponse reads a non-2xx response, drops reasoning replay state that
// upstream rejected, and returns the status error.
func (e *GeminiCLIExecutor) handleErrorResponse(ctx context.Context, plan geminiCLIRequestPlan, httpResp *http.Response, sentBody []byte) error {
	bodyBytes, errStatus := readGeminiCLIErrorBody(ctx, e.cfg, httpResp)
	if httpResp.StatusCode == http.StatusBadRequest {
		log.Debugf("gemini-cli executor: rejected request contents shape: %s", geminiCLIContentsShape(sentBody))
	}
	if bodyBytes != nil {
		if errClear := clearAntigravityReasoningReplayOnInvalidSignature(ctx, plan.replayScope, httpResp.StatusCode, bodyBytes); errClear != nil {
			// Report the upstream failure rather than the cleanup failure.
			logAntigravityReasoningReplayDegraded(plan.replayScope, "invalidate", errClear)
		}
	}
	return errStatus
}

// Execute performs a non-streaming request to Cloud Code Assist.
func (e *GeminiCLIExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	if opts.Alt == "responses/compact" {
		return resp, statusErr{code: http.StatusNotImplemented, msg: "/responses/compact not supported"}
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	plan, err := e.planRequest(ctx, auth, req, opts, false, reporter)
	if err != nil {
		return resp, err
	}
	httpClient := reporter.TrackHTTPClient(helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0))
	httpReq, sentBody, err := e.buildRequest(ctx, auth, plan.token, baseModel, plan.requestPayload, false, opts.Alt, plan.sessionID)
	if err != nil {
		return resp, err
	}
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		err = errDo
		return resp, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		err = e.handleErrorResponse(ctx, plan, httpResp, sentBody)
		return resp, err
	}
	bodyBytes, errRead := io.ReadAll(httpResp.Body)
	if errClose := httpResp.Body.Close(); errClose != nil {
		log.Errorf("gemini-cli executor: close response body error: %v", errClose)
	}
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		err = errRead
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, bodyBytes)

	cacheAntigravityReasoningReplayFromResponse(ctx, plan.replayScope, plan.requestPayload, bodyBytes)
	bodyBytes = e.resolveWebSearchGroundingURLs(ctx, auth, plan, bodyBytes)
	reporter.ObserveResponseModel(bodyBytes)
	reporter.Publish(ctx, helps.ParseAntigravityUsage(bodyBytes))
	var param any
	converted := sdktranslator.TranslateNonStream(ctx, plan.to, plan.responseFormat, req.Model, opts.OriginalRequest, plan.translated, bodyBytes, &param)
	if plan.responseFormat == sdktranslator.FormatOpenAIResponse {
		converted = helps.EnsureResponsesUsageDetails(converted)
	}
	resp = cliproxyexecutor.Response{Payload: converted, Headers: httpResp.Header.Clone()}
	reporter.EnsurePublished(ctx)
	return resp, nil
}

// ExecuteStream performs a streaming request to Cloud Code Assist.
func (e *GeminiCLIExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	if opts.Alt == "responses/compact" {
		return nil, statusErr{code: http.StatusBadRequest, msg: "streaming not supported for /responses/compact"}
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	ctx = context.WithValue(ctx, "alt", "")
	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	plan, err := e.planRequest(ctx, auth, req, opts, true, reporter)
	if err != nil {
		return nil, err
	}
	httpClient := reporter.TrackHTTPClient(helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0))
	httpReq, sentBody, err := e.buildRequest(ctx, auth, plan.token, baseModel, plan.requestPayload, true, opts.Alt, plan.sessionID)
	if err != nil {
		return nil, err
	}
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		if errors.Is(errDo, context.Canceled) || errors.Is(errDo, context.DeadlineExceeded) {
			return nil, errDo
		}
		err = errDo
		return nil, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		err = e.handleErrorResponse(ctx, plan, httpResp, sentBody)
		return nil, err
	}

	replayAccumulator := newAntigravityReasoningReplayAccumulator(plan.replayScope, plan.requestPayload)
	out := make(chan cliproxyexecutor.StreamChunk)
	go func(resp *http.Response) {
		defer close(out)
		defer func() {
			if errClose := resp.Body.Close(); errClose != nil {
				log.Errorf("gemini-cli executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(nil, streamScannerBuffer)
		claudeInputTokens := helps.NewClaudeInputTokenState(plan.from, plan.to, plan.responseFormat, plan.originalPayload)
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			if replayAccumulator != nil {
				replayAccumulator.ObserveSSELine(line)
			}

			// Only the terminal chunk keeps usageMetadata; earlier ones are renamed so
			// translators do not report partial usage as final.
			line = helps.FilterSSEUsageMetadata(line)

			payload := helps.JSONPayload(line)
			if payload == nil {
				continue
			}
			reporter.ObserveResponseModel(payload)
			if detail, ok := helps.ParseAntigravityStreamUsage(payload); ok {
				reporter.Publish(ctx, detail)
			}

			payload = e.resolveWebSearchGroundingURLs(ctx, auth, plan, payload)
			chunks := helps.TranslateStreamWithClaudeInputTokens(ctx, plan.to, plan.responseFormat, req.Model, opts.OriginalRequest, plan.translated, bytes.Clone(payload), &param, claudeInputTokens)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					return
				}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			reporter.PublishFailure(ctx, errScan)
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errScan}:
			case <-ctx.Done():
			}
			return
		}
		// Only a clean end of stream may produce a synthetic terminal event; translating
		// [DONE] after a read error would report a truncated stream as complete.
		tail := helps.TranslateStreamWithClaudeInputTokens(ctx, plan.to, plan.responseFormat, req.Model, opts.OriginalRequest, plan.translated, []byte("[DONE]"), &param, claudeInputTokens)
		for i := range tail {
			select {
			case out <- cliproxyexecutor.StreamChunk{Payload: tail[i]}:
			case <-ctx.Done():
				return
			}
		}
		if replayAccumulator != nil {
			replayAccumulator.Commit(ctx)
		}
		reporter.EnsurePublished(ctx)
	}(httpResp)
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

// CountTokens counts tokens with the request shape gemini-cli sends:
// {"request":{"model":"models/<model>","contents":[...]}}.
func (e *GeminiCLIExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	from := opts.SourceFormat
	to := sdktranslator.FormatAntigravity
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	respCtx := context.WithValue(ctx, "alt", opts.Alt)

	token, errToken := e.accessToken(ctx, auth)
	if errToken != nil {
		return cliproxyexecutor.Response{}, errToken
	}
	cliproxyauth.NotifyAccessTokenFingerprint(ctx, auth)

	modelInfo, _ := cliproxyauth.ResolvedModelInfo(req)
	translationReq := sdktranslator.RequestEnvelope{Format: from, Model: baseModel, Body: req.Payload, ModelInfo: modelInfo}
	translated := helps.TranslateRequestEnvelopeWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, translationReq).Body
	translated = ensureAntigravityGeminiLeadingUserContent(baseModel, translated)

	payload := []byte(`{"request":{"model":"","contents":[]}}`)
	payload, _ = sjson.SetBytes(payload, "request.model", "models/"+baseModel)
	if contents := gjson.GetBytes(translated, "request.contents"); contents.IsArray() {
		payload, _ = sjson.SetRawBytes(payload, "request.contents", []byte(contents.Raw))
	}

	requestURL := resolveGeminiCLIBaseURL(auth) + geminiCLICountTokensPath
	httpReq, errReq := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(payload))
	if errReq != nil {
		return cliproxyexecutor.Response{}, errReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	applyGeminiCLIHeaders(httpReq, baseModel)
	if auth != nil {
		util.ApplyCustomHeadersFromAttrs(httpReq, auth.Attributes)
	}
	e.recordRequest(ctx, auth, httpReq, payload)

	cliproxyexecutor.MarkUpstreamAttempt(ctx)
	httpResp, errDo := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0).Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return cliproxyexecutor.Response{}, errDo
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < http.StatusOK || httpResp.StatusCode >= http.StatusMultipleChoices {
		_, errStatus := readGeminiCLIErrorBody(ctx, e.cfg, httpResp)
		return cliproxyexecutor.Response{}, errStatus
	}
	bodyBytes, errRead := io.ReadAll(httpResp.Body)
	if errClose := httpResp.Body.Close(); errClose != nil {
		log.Errorf("gemini-cli executor: close response body error: %v", errClose)
	}
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		return cliproxyexecutor.Response{}, errRead
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, bodyBytes)
	count := gjson.GetBytes(bodyBytes, "totalTokens").Int()
	translatedCount := sdktranslator.TranslateTokenCount(respCtx, to, responseFormat, count, bodyBytes)
	return cliproxyexecutor.Response{Payload: translatedCount, Headers: httpResp.Header.Clone()}, nil
}
