package synthesizer

import (
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
)

func TestConfigSynthesizerBuildsBedrockClaudeAuth(t *testing.T) {
	cfg, errParse := config.ParseConfigBytes([]byte(`
api-keys:
  claude:
    - name: bedrock-eu
      models:
        - name: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
          alias: "claude-sonnet-4-5"
      keys:
        - bedrock:
            profile: " work-sso "
            region: eu-central-1
        - bedrock:
            profile: other
            region: eu-central-1
            betas: []
    - name: anthropic
      keys:
        - api-key: sk-test
`))
	if errParse != nil {
		t.Fatal(errParse)
	}
	if len(cfg.ClaudeKey) != 3 || cfg.ClaudeKey[0].Bedrock == nil || cfg.ClaudeKey[0].Bedrock.Profile != "work-sso" {
		t.Fatalf("claude keys = %+v", cfg.ClaudeKey)
	}

	auths, errSynth := NewConfigSynthesizer().Synthesize(&SynthesisContext{Config: cfg, Now: time.Now(), IDGenerator: NewStableIDGenerator()})
	if errSynth != nil {
		t.Fatal(errSynth)
	}
	if len(auths) != 3 {
		t.Fatalf("auths = %d, want 3", len(auths))
	}
	first, second, plain := auths[0], auths[1], auths[2]
	if first.ID == second.ID {
		t.Fatal("two Bedrock profiles in one region must get distinct auth IDs")
	}
	if first.Provider != "claude" || first.Label != "claude-bedrock" || first.Attributes["base_url"] != "https://bedrock-runtime.eu-central-1.amazonaws.com" {
		t.Fatalf("first auth = %+v", first)
	}
	if _, hasKey := first.Attributes["api_key"]; hasKey {
		t.Fatal("a Bedrock auth must not carry an api_key")
	}
	settings, ok := bedrock.SettingsFromAttributes(first.Attributes)
	if !ok || settings.Profile != "work-sso" || settings.Region != "eu-central-1" || settings.BetasSet {
		t.Fatalf("first settings = %+v ok=%v", settings, ok)
	}
	if settings, ok = bedrock.SettingsFromAttributes(second.Attributes); !ok || !settings.BetasSet || len(settings.Betas) != 0 {
		t.Fatalf("an explicit empty betas list must survive: %+v", settings)
	}
	if _, ok = bedrock.SettingsFromAttributes(plain.Attributes); ok || plain.Label != "claude-apikey" {
		t.Fatalf("a plain Claude key must not become Bedrock: %+v", plain)
	}
}
