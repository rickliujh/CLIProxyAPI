package auth

import (
	"context"
	"testing"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
)

// A Bedrock entry has no api-key or base-url in config; its credential carries
// the derived Bedrock base URL. Its configured aliases must still resolve.
func TestApplyAPIKeyModelAliasForBedrockEntry(t *testing.T) {
	cfg := &internalconfig.Config{
		ClaudeKey: []internalconfig.ClaudeKey{{
			Bedrock: &internalconfig.BedrockConfig{Profile: "work", Region: "eu-central-1"},
			Models:  []internalconfig.ClaudeModel{{Name: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", Alias: "claude-sonnet-4-5"}},
		}},
	}
	mgr := NewManager(nil, nil, nil)
	mgr.SetConfig(cfg)

	auth := &Auth{ID: "bedrock-1", Provider: "claude", Attributes: map[string]string{
		"base_url":     "https://bedrock-runtime.eu-central-1.amazonaws.com",
		"config_index": "0",
		"source":       "config:claude[token]",
		"bedrock":      "true",
		"auth_kind":    "apikey",
	}}
	_, _ = mgr.Register(context.Background(), auth)

	if got := mgr.applyAPIKeyModelAlias(auth, "claude-sonnet-4-5"); got != "eu.anthropic.claude-sonnet-4-5-20250929-v1:0" {
		t.Fatalf("model = %q, want the Bedrock inference profile ID", got)
	}
}
