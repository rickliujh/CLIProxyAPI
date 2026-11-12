package synthesizer

import (
	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
)

// claudeBedrockSettings converts a claude-api-key bedrock block into the settings
// the Bedrock transport reads from the auth attributes.
func claudeBedrockSettings(entry *config.ClaudeKey) (bedrock.Settings, bool) {
	if entry == nil || entry.Bedrock == nil {
		return bedrock.Settings{}, false
	}
	cfg := entry.Bedrock
	return bedrock.Settings{
		Profile:         cfg.Profile,
		Region:          cfg.Region,
		Endpoint:        cfg.Endpoint,
		ControlEndpoint: cfg.ControlEndpoint,
		Geo:             cfg.Geo,
		Betas:           cfg.Betas,
		BetasSet:        cfg.Betas != nil,
	}, true
}
