// Package bedrock sends Anthropic Messages requests to Amazon Bedrock.
//
// A Claude API-key entry with a bedrock block keeps running through the Claude
// executor. Its HTTP client gets a RoundTripper from WrapTransport that turns the
// outgoing /v1/messages request into a SigV4-signed Bedrock InvokeModel call and
// turns the Bedrock response back into what the Anthropic API would have returned.
package bedrock

import (
	"net/http"
	"strings"
)

// Auth attribute names that carry an entry's Bedrock settings to the transport.
const (
	AttrEnabled  = "bedrock"
	AttrProfile  = "bedrock_profile"
	AttrRegion   = "bedrock_region"
	AttrEndpoint = "bedrock_endpoint"
	AttrBetas    = "bedrock_betas"
)

// DefaultBaseURL is the placeholder base URL for an entry whose region is only
// known once its AWS profile is loaded. The transport always replaces the host.
const DefaultBaseURL = "https://bedrock-runtime.amazonaws.com"

// DefaultBetas are the anthropic-beta values Bedrock accepts for Claude models.
// Bedrock rejects a request that carries any beta it does not know, so every
// other value a client sends is dropped.
var DefaultBetas = []string{
	"computer-use-2024-10-22",
	"computer-use-2025-01-24",
	"computer-use-2025-11-24",
	"token-efficient-tools-2025-02-19",
	"interleaved-thinking-2025-05-14",
	"output-128k-2025-02-19",
	"dev-full-thinking-2025-05-14",
	"context-1m-2025-08-07",
	"context-management-2025-06-27",
	"effort-2025-11-24",
	"tool-search-tool-2025-10-19",
	"tool-examples-2025-10-29",
}

// Settings is the Bedrock configuration of one credential.
type Settings struct {
	Profile  string
	Region   string
	Endpoint string
	// Betas replaces DefaultBetas when BetasSet is true.
	Betas    []string
	BetasSet bool
}

// BaseURL returns the base URL recorded on the credential for these settings.
func (s Settings) BaseURL() string {
	if s.Endpoint != "" {
		return s.Endpoint
	}
	if s.Region != "" {
		return "https://bedrock-runtime." + s.Region + ".amazonaws.com"
	}
	return DefaultBaseURL
}

// Attributes encodes the settings as auth attributes.
func (s Settings) Attributes() map[string]string {
	attrs := map[string]string{AttrEnabled: "true"}
	if s.Profile != "" {
		attrs[AttrProfile] = s.Profile
	}
	if s.Region != "" {
		attrs[AttrRegion] = s.Region
	}
	if s.Endpoint != "" {
		attrs[AttrEndpoint] = s.Endpoint
	}
	if s.BetasSet {
		attrs[AttrBetas] = strings.Join(s.Betas, ",")
	}
	return attrs
}

// SettingsFromAttributes reads the settings back from auth attributes. ok is
// false when the credential does not target Bedrock.
func SettingsFromAttributes(attrs map[string]string) (settings Settings, ok bool) {
	if !strings.EqualFold(strings.TrimSpace(attrs[AttrEnabled]), "true") {
		return Settings{}, false
	}
	settings = Settings{
		Profile:  strings.TrimSpace(attrs[AttrProfile]),
		Region:   strings.TrimSpace(attrs[AttrRegion]),
		Endpoint: strings.TrimSpace(attrs[AttrEndpoint]),
	}
	if raw, exists := attrs[AttrBetas]; exists {
		settings.BetasSet = true
		settings.Betas = splitList(raw)
	}
	return settings, true
}

// WrapTransport returns a RoundTripper that sends requests of a Bedrock
// credential to Bedrock over base. For any other credential it returns base.
func WrapTransport(attrs map[string]string, base http.RoundTripper) http.RoundTripper {
	settings, ok := SettingsFromAttributes(attrs)
	if !ok {
		return base
	}
	if base == nil {
		base = http.DefaultTransport
	}
	return newTransport(settings, base, profileCredentials(settings))
}

func splitList(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
