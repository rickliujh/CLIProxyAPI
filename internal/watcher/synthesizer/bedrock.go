package synthesizer

import (
	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	log "github.com/sirupsen/logrus"
)

// claudeBedrockSettings converts a claude-api-key bedrock block into the settings
// the Bedrock transport reads from the auth attributes.
func claudeBedrockSettings(entry *config.ClaudeKey) (bedrock.Settings, bool) {
	if entry == nil || entry.Bedrock == nil {
		return bedrock.Settings{}, false
	}
	cfg := entry.Bedrock
	if len(entry.Models) == 0 {
		// Without models the entry serves the Anthropic catalog names, which are not
		// Bedrock model IDs, so every request would fail.
		log.Warnf("claude bedrock entry (profile %q) has no models; map Bedrock inference profile IDs such as eu.anthropic.* to client names under models", cfg.Profile)
	}
	return bedrock.Settings{
		Profile:  cfg.Profile,
		Region:   cfg.Region,
		Endpoint: cfg.Endpoint,
		Betas:    cfg.Betas,
		BetasSet: cfg.Betas != nil,
	}, true
}
