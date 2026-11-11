package config

import "strings"

// BedrockConfig routes a Claude API-key entry to Amazon Bedrock. Requests keep the
// Anthropic Messages shape and are signed with credentials from an AWS CLI profile,
// so the entry needs no api-key.
type BedrockConfig struct {
	// Profile names the AWS CLI profile (~/.aws/config) that supplies credentials.
	// SSO, assume-role, credential_process and static keys all work. Empty uses the
	// default AWS credential chain (AWS_PROFILE, environment, default profile).
	Profile string `yaml:"profile,omitempty" json:"profile,omitempty"`

	// Region is the Bedrock Runtime region, e.g. "eu-central-1". Empty uses the
	// profile's region. Geographic inference profiles such as "eu.anthropic.*"
	// can only be called from a region in the same geography.
	Region string `yaml:"region,omitempty" json:"region,omitempty"`

	// Endpoint optionally overrides the Bedrock Runtime endpoint, e.g. a VPC
	// interface endpoint. Empty uses https://bedrock-runtime.<region>.amazonaws.com.
	Endpoint string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`

	// Betas lists the anthropic-beta values forwarded to Bedrock. Bedrock rejects
	// betas it does not support, so other values are dropped. Omit to use the
	// built-in list; an empty list forwards none.
	Betas []string `yaml:"betas,omitempty" json:"betas,omitempty"`
}

// NormalizeBedrockConfig trims whitespace and drops empty beta names.
func NormalizeBedrockConfig(cfg *BedrockConfig) *BedrockConfig {
	if cfg == nil {
		return nil
	}
	cfg.Profile = strings.TrimSpace(cfg.Profile)
	cfg.Region = strings.TrimSpace(cfg.Region)
	cfg.Endpoint = strings.TrimRight(strings.TrimSpace(cfg.Endpoint), "/")
	if cfg.Betas != nil {
		betas := make([]string, 0, len(cfg.Betas))
		for _, beta := range cfg.Betas {
			if trimmed := strings.TrimSpace(beta); trimmed != "" {
				betas = append(betas, trimmed)
			}
		}
		cfg.Betas = betas
	}
	return cfg
}
