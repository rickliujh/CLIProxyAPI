// Package executor provides runtime execution capabilities for various AI service providers.
// This file implements the Gemini CLI executor. Gemini CLI and Antigravity both call the
// Cloud Code Assist API (v1internal), so this executor reuses the Antigravity translators,
// signature validation and reasoning replay, and only differs in client identity: the
// OAuth client, endpoint, headers and request envelope match Google's gemini-cli.
package executor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/geminicli"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	geminiCLIAuthType         = "gemini-cli"
	geminiCLIBaseURL          = "https://cloudcode-pa.googleapis.com"
	geminiCLIStreamPath       = "/v1internal:streamGenerateContent"
	geminiCLIGeneratePath     = "/v1internal:generateContent"
	geminiCLICountTokensPath  = "/v1internal:countTokens"
	geminiOAuthClientID       = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
	geminiOAuthClientSecret   = "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"
	geminiCLITokenSafetyDelta = 30 * time.Second
)

var geminiOAuthScopes = []string{
	"https://www.googleapis.com/auth/cloud-platform",
	"https://www.googleapis.com/auth/userinfo.email",
	"https://www.googleapis.com/auth/userinfo.profile",
}

// GeminiCLIExecutor calls the Cloud Code Assist API with Gemini CLI OAuth credentials.
type GeminiCLIExecutor struct {
	cfg *config.Config
}

// NewGeminiCLIExecutor creates a new Gemini CLI executor instance.
func NewGeminiCLIExecutor(cfg *config.Config) *GeminiCLIExecutor {
	return &GeminiCLIExecutor{cfg: cfg}
}

// Identifier returns the executor identifier.
func (e *GeminiCLIExecutor) Identifier() string { return geminiCLIAuthType }

// PrepareRequest injects Gemini CLI credentials into the outgoing HTTP request.
func (e *GeminiCLIExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	token, errToken := e.accessToken(req.Context(), auth)
	if errToken != nil {
		return errToken
	}
	req.Header.Set("Authorization", "Bearer "+token)
	applyGeminiCLIHeaders(req, "unknown")
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Gemini CLI credentials into the request and executes it.
func (e *GeminiCLIExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("gemini-cli executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	return helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0).Do(httpReq)
}

// Refresh defers to Home when enabled. Otherwise tokens are refreshed lazily by the
// oauth2 token source on each request, so there is nothing to do here.
func (e *GeminiCLIExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	if refreshed, handled, err := helps.RefreshAuthViaHome(ctx, e.cfg, auth); handled {
		return refreshed, err
	}
	return auth, nil
}

// accessToken returns a valid access token, refreshing it when needed and writing the
// refreshed token back to the credential (shared across virtual project auths).
func (e *GeminiCLIExecutor) accessToken(ctx context.Context, auth *cliproxyauth.Auth) (string, error) {
	tokenSource, baseTokenData, errSource := prepareGeminiCLITokenSource(ctx, e.cfg, auth)
	if errSource != nil {
		return "", errSource
	}
	tok, errTok := tokenSource.Token()
	if errTok != nil {
		return "", errTok
	}
	updateGeminiCLITokenMetadata(auth, baseTokenData, tok)
	if strings.TrimSpace(tok.AccessToken) == "" {
		return "", statusErr{code: http.StatusUnauthorized, msg: "missing access token"}
	}
	return tok.AccessToken, nil
}

func prepareGeminiCLITokenSource(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth) (oauth2.TokenSource, map[string]any, error) {
	metadata := geminiOAuthMetadata(auth)
	if auth == nil || metadata == nil {
		return nil, nil, fmt.Errorf("gemini-cli auth metadata missing")
	}

	buildToken := func(meta map[string]any) (map[string]any, oauth2.Token) {
		var base map[string]any
		if tokenRaw, ok := meta["token"].(map[string]any); ok && tokenRaw != nil {
			base = cloneMap(tokenRaw)
		} else {
			base = make(map[string]any)
		}

		var token oauth2.Token
		if len(base) > 0 {
			if raw, errMarshal := json.Marshal(base); errMarshal == nil {
				_ = json.Unmarshal(raw, &token)
			}
		}

		if token.AccessToken == "" {
			token.AccessToken = stringValue(meta, "access_token")
		}
		if token.RefreshToken == "" {
			token.RefreshToken = stringValue(meta, "refresh_token")
		}
		if token.TokenType == "" {
			token.TokenType = stringValue(meta, "token_type")
		}
		if token.Expiry.IsZero() {
			if expiry := stringValue(meta, "expiry"); expiry != "" {
				if ts, errParse := time.Parse(time.RFC3339, expiry); errParse == nil {
					token.Expiry = ts
				}
			}
		}

		return base, token
	}

	base, token := buildToken(metadata)

	if cfg != nil && cfg.Home.Enabled {
		now := time.Now()
		if token.AccessToken == "" || (!token.Expiry.IsZero() && token.Expiry.Before(now.Add(geminiCLITokenSafetyDelta))) {
			refreshed, handled, errRefresh := helps.RefreshAuthViaHome(ctx, cfg, auth)
			if handled {
				if errRefresh != nil {
					return nil, nil, errRefresh
				}
				auth = refreshed
				metadata = geminiOAuthMetadata(auth)
				if metadata == nil {
					return nil, nil, fmt.Errorf("gemini-cli auth metadata missing")
				}
				base, token = buildToken(metadata)
			}
		}
		if token.AccessToken == "" {
			return nil, nil, fmt.Errorf("gemini-cli access token missing")
		}
		updateGeminiCLITokenMetadata(auth, base, &token)
		return oauth2.StaticTokenSource(&token), base, nil
	}

	conf := &oauth2.Config{
		ClientID:     geminiOAuthClientID,
		ClientSecret: geminiOAuthClientSecret,
		Scopes:       geminiOAuthScopes,
		Endpoint:     google.Endpoint,
	}
	ctxToken := ctx
	if httpClient := helps.NewProxyAwareHTTPClient(ctx, cfg, auth, 0); httpClient != nil {
		ctxToken = context.WithValue(ctxToken, oauth2.HTTPClient, httpClient)
	}
	src := conf.TokenSource(ctxToken, &token)
	currentToken, errToken := src.Token()
	if errToken != nil {
		return nil, nil, errToken
	}
	updateGeminiCLITokenMetadata(auth, base, currentToken)
	return oauth2.ReuseTokenSource(currentToken, src), base, nil
}

func updateGeminiCLITokenMetadata(auth *cliproxyauth.Auth, base map[string]any, tok *oauth2.Token) {
	if auth == nil || tok == nil {
		return
	}
	merged := buildGeminiTokenMap(base, tok)
	fields := buildGeminiTokenFields(tok, merged)
	if shared := geminicli.ResolveSharedCredential(auth.Runtime); shared != nil {
		snapshot := shared.MergeMetadata(fields)
		if !geminicli.IsVirtual(auth.Runtime) {
			auth.Metadata = snapshot
		}
		return
	}
	if auth.Metadata == nil {
		auth.Metadata = make(map[string]any)
	}
	for k, v := range fields {
		auth.Metadata[k] = v
	}
}

func buildGeminiTokenMap(base map[string]any, tok *oauth2.Token) map[string]any {
	merged := cloneMap(base)
	if merged == nil {
		merged = make(map[string]any)
	}
	if raw, errMarshal := json.Marshal(tok); errMarshal == nil {
		var tokenMap map[string]any
		if errUnmarshal := json.Unmarshal(raw, &tokenMap); errUnmarshal == nil {
			for k, v := range tokenMap {
				merged[k] = v
			}
		}
	}
	return merged
}

func buildGeminiTokenFields(tok *oauth2.Token, merged map[string]any) map[string]any {
	fields := make(map[string]any, 5)
	if tok.AccessToken != "" {
		fields["access_token"] = tok.AccessToken
	}
	if tok.TokenType != "" {
		fields["token_type"] = tok.TokenType
	}
	if tok.RefreshToken != "" {
		fields["refresh_token"] = tok.RefreshToken
	}
	if !tok.Expiry.IsZero() {
		fields["expiry"] = tok.Expiry.Format(time.RFC3339)
	}
	if len(merged) > 0 {
		fields["token"] = cloneMap(merged)
	}
	return fields
}

func geminiOAuthMetadata(auth *cliproxyauth.Auth) map[string]any {
	if auth == nil {
		return nil
	}
	if shared := geminicli.ResolveSharedCredential(auth.Runtime); shared != nil {
		if snapshot := shared.MetadataSnapshot(); len(snapshot) > 0 {
			return snapshot
		}
	}
	return auth.Metadata
}

// resolveGeminiProjectID returns the Code Assist project for a credential. Virtual
// auths synthesized from a multi-project credential carry their own project.
func resolveGeminiProjectID(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	if virtual, ok := auth.Runtime.(*geminicli.VirtualCredential); ok && virtual != nil {
		return strings.TrimSpace(virtual.ProjectID)
	}
	return strings.TrimSpace(stringValue(auth.Metadata, "project_id"))
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func stringValue(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key]; ok {
		switch typed := v.(type) {
		case string:
			return typed
		case fmt.Stringer:
			return typed.String()
		}
	}
	return ""
}

// resolveGeminiCLIBaseURL returns the Code Assist endpoint. A credential may override it
// with a base_url attribute or metadata field, the same way Antigravity credentials can.
func resolveGeminiCLIBaseURL(auth *cliproxyauth.Auth) string {
	if base := resolveCustomAntigravityBaseURL(auth); base != "" {
		return base
	}
	return geminiCLIBaseURL
}

// applyGeminiCLIHeaders sets the headers Google's gemini-cli sends. User-Agent is always
// replaced so the upstream sees a native Gemini CLI client.
func applyGeminiCLIHeaders(r *http.Request, model string) {
	r.Header.Set("User-Agent", misc.GeminiCLIUserAgent(model))
	r.Header.Set("X-Goog-Api-Client", misc.GeminiCLIApiClientHeader)
}

// geminiCLIEnvelope turns a translated Antigravity-format payload into the request body
// gemini-cli sends: {model, project, user_prompt_id, request{..., session_id}}. The
// Antigravity-only envelope fields are removed so the request carries one identity.
func geminiCLIEnvelope(modelName string, payload []byte, projectID, sessionID string) []byte {
	for _, field := range []string{"userAgent", "requestType", "requestId", "enabledCreditTypes", "request.sessionId"} {
		payload, _ = sjson.DeleteBytes(payload, field)
	}
	payload = helps.SetStringIfDifferent(payload, "model", modelName)
	payload = helps.SetStringIfDifferent(payload, "project", projectID)
	payload, _ = sjson.SetBytes(payload, "user_prompt_id", uuid.NewString())
	if sessionID = strings.TrimPrefix(strings.TrimSpace(sessionID), "-"); sessionID == "" {
		sessionID = strings.TrimPrefix(generateStableSessionID(payload), "-")
	}
	payload, _ = sjson.SetBytes(payload, "request.session_id", sessionID)
	if toolConfig := gjson.GetBytes(payload, "toolConfig"); toolConfig.Exists() && !gjson.GetBytes(payload, "request.toolConfig").Exists() {
		payload, _ = sjson.SetRawBytes(payload, "request.toolConfig", []byte(toolConfig.Raw))
		payload, _ = sjson.DeleteBytes(payload, "toolConfig")
	}
	payload = geminiCLIFunctionResponseRolesToUser(payload)
	// Code Assist rejects a Gemini CLI request whose last turn is a model turn.
	return helps.EnsureGeminiTrailingUserContent(payload, "request.contents")
}

// geminiCLIFunctionResponseRolesToUser shapes tool-result turns the way Code Assist
// accepts them for Gemini CLI. The Antigravity pipeline sends them as model turns with
// the tool results first and any other parts (such as Claude Code's system reminders)
// after them. Code Assist rejects both with a misleading 400 "Requests ending with a
// model turn are not supported" (see #5607). A functionResponse is never
// model-authored, so the turn becomes a user turn, and its text parts are moved ahead
// of the tool results as the Gemini translator does. Only the wire payload is changed;
// reasoning replay keeps the Antigravity form.
func geminiCLIFunctionResponseRolesToUser(payload []byte) []byte {
	contents := gjson.GetBytes(payload, "request.contents")
	if !contents.IsArray() {
		return payload
	}
	for i, content := range contents.Array() {
		if !translatorcommon.ContentHasGeminiFunctionResponse([]byte(content.Raw)) {
			continue
		}
		path := fmt.Sprintf("request.contents.%d", i)
		if content.Get("role").String() != "user" {
			payload, _ = sjson.SetBytes(payload, path+".role", "user")
		}
		parts := content.Get("parts").Array()
		rawParts := make([][]byte, 0, len(parts))
		for _, part := range parts {
			rawParts = append(rawParts, []byte(part.Raw))
		}
		original := translatorcommon.JoinRawArray(rawParts)
		if reordered := translatorcommon.JoinRawArray(translatorcommon.ReorderGeminiUserParts(rawParts)); !bytes.Equal(reordered, original) {
			payload, _ = sjson.SetRawBytes(payload, path+".parts", reordered)
		}
	}
	return payload
}

// geminiCLIClaudeMessagesShape summarizes an inbound Claude request as
// role:block-types, without any message text, for diagnosing upstream rejections.
func geminiCLIClaudeMessagesShape(payload []byte) string {
	var shape []string
	for _, message := range gjson.GetBytes(payload, "messages").Array() {
		var kinds []string
		content := message.Get("content")
		if content.Type == gjson.String {
			kinds = append(kinds, "text")
		}
		for _, block := range content.Array() {
			kind := block.Get("type").String()
			if kind == "thinking" && strings.TrimSpace(block.Get("signature").String()) == "" {
				kind += "(nosig)"
			}
			kinds = append(kinds, kind)
		}
		shape = append(shape, message.Get("role").String()+":"+strings.Join(kinds, ","))
	}
	return "[" + strings.Join(shape, " | ") + "]"
}

// geminiCLIContentsShape summarizes the turn structure of a request as role:part-kinds,
// without any message text, for diagnosing upstream rejections.
func geminiCLIContentsShape(payload []byte) string {
	var shape []string
	for _, content := range gjson.GetBytes(payload, "request.contents").Array() {
		var kinds []string
		for _, part := range content.Get("parts").Array() {
			kind := "other"
			switch {
			case part.Get("functionCall").Exists():
				kind = "functionCall"
			case part.Get("functionResponse").Exists():
				kind = "functionResponse"
			case part.Get("thought").Bool():
				kind = "thought"
			case part.Get("text").Exists():
				kind = "text"
				if part.Get("text").String() == "" {
					kind = "emptyText"
				}
			case part.Get("inlineData").Exists():
				kind = "inlineData"
			}
			if part.Get("thoughtSignature").Exists() {
				kind += "+sig"
			}
			kinds = append(kinds, kind)
		}
		shape = append(shape, content.Get("role").String()+":"+strings.Join(kinds, ","))
	}
	return "[" + strings.Join(shape, " | ") + "]"
}

// buildRequest assembles one generate or stream request against Cloud Code Assist.
// It returns the final request body alongside the request for error diagnostics.
func (e *GeminiCLIExecutor) buildRequest(ctx context.Context, auth *cliproxyauth.Auth, token, modelName string, payload []byte, stream bool, alt, sessionID string) (*http.Request, []byte, error) {
	projectID := resolveGeminiProjectID(auth)
	if projectID == "" {
		return nil, nil, statusErr{code: http.StatusBadRequest, msg: "gemini-cli auth missing project_id; log in again with -login"}
	}
	log.Debugf("gemini-cli executor: pipeline contents shape: %s", geminiCLIContentsShape(payload))
	payload = geminiCLIEnvelope(modelName, payload, projectID, sessionID)
	log.Debugf("gemini-cli executor: sent contents shape: %s", geminiCLIContentsShape(payload))
	if antigravityRequestNeedsSchemaSanitization(payload) {
		useAntigravitySchema := strings.Contains(modelName, "gemini-3-pro") || strings.Contains(modelName, "gemini-3.1-pro")
		payload = []byte(sanitizeAntigravityRequestSchemas(string(payload), useAntigravitySchema))
	}

	path := geminiCLIGeneratePath
	if stream {
		path = geminiCLIStreamPath
	}
	var requestURL strings.Builder
	requestURL.WriteString(resolveGeminiCLIBaseURL(auth))
	requestURL.WriteString(path)
	switch {
	case alt != "":
		requestURL.WriteString("?$alt=")
		requestURL.WriteString(url.QueryEscape(alt))
	case stream:
		requestURL.WriteString("?alt=sse")
	}

	httpReq, errReq := http.NewRequestWithContext(ctx, http.MethodPost, requestURL.String(), bytes.NewReader(payload))
	if errReq != nil {
		return nil, nil, errReq
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	applyGeminiCLIHeaders(httpReq, modelName)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	e.recordRequest(ctx, auth, httpReq, payload)
	return httpReq, payload, nil
}

func (e *GeminiCLIExecutor) recordRequest(ctx context.Context, auth *cliproxyauth.Auth, httpReq *http.Request, payload []byte) {
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       httpReq.URL.String(),
		Method:    httpReq.Method,
		Headers:   httpReq.Header.Clone(),
		Body:      payload,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
}

// readGeminiCLIErrorBody drains a failed response and converts it into a status error
// that carries the upstream retry delay, if any. The body is returned so callers can
// invalidate reasoning replay state when upstream rejected a signature.
func readGeminiCLIErrorBody(ctx context.Context, cfg *config.Config, httpResp *http.Response) ([]byte, error) {
	bodyBytes, errRead := io.ReadAll(httpResp.Body)
	if errClose := httpResp.Body.Close(); errClose != nil {
		log.Errorf("gemini-cli executor: close response body error: %v", errClose)
	}
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, cfg, errRead)
		if errCtx := ctx.Err(); errCtx != nil {
			return nil, errCtx
		}
		return nil, errRead
	}
	helps.AppendAPIResponseChunk(ctx, cfg, bodyBytes)
	log.Debugf("gemini-cli executor: upstream error status: %d, body: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), bodyBytes))
	return bodyBytes, newAntigravityStatusErr(httpResp.StatusCode, bodyBytes)
}
