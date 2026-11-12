// Package geminicli implements thinking configuration for Gemini CLI API format.
//
// Gemini CLI uses request.generationConfig.thinkingConfig.* path instead of
// generationConfig.thinkingConfig.* used by standard Gemini API.
package geminicli

import (
	"strings"

	"github.com/MoeForge/moego/ojson"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/thinking"
)

// Applier applies thinking configuration for Gemini CLI API format.
type Applier struct{}

var _ thinking.ProviderDocApplier = (*Applier)(nil)

// NewApplier creates a new Gemini CLI thinking applier.
func NewApplier() *Applier {
	return &Applier{}
}

func init() {
	thinking.RegisterProvider("gemini-cli", NewApplier())
}

// thinkingConfigPath is where Gemini CLI carries the thinking configuration.
const thinkingConfigPath = "request.generationConfig.thinkingConfig."

func supportedMode(mode thinking.ThinkingMode) bool {
	return mode == thinking.ModeBudget || mode == thinking.ModeLevel || mode == thinking.ModeNone || mode == thinking.ModeAuto
}

// Apply applies thinking configuration to Gemini CLI request body.
func (a *Applier) Apply(body []byte, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) ([]byte, error) {
	if !thinking.IsUserDefinedModel(modelInfo) && modelInfo.Thinking == nil {
		return body, nil
	}
	if !supportedMode(config.Mode) {
		return body, nil
	}
	if len(body) == 0 || !ojson.ValidBytes(body) {
		body = []byte(`{}`)
	}
	doc := ojson.NewDoc(body)
	if errApply := a.ApplyDoc(doc, config, modelInfo); errApply != nil {
		return body, errApply
	}
	return doc.Bytes(), nil
}

// ApplyDoc applies thinking configuration onto doc in place.
func (a *Applier) ApplyDoc(doc *ojson.Doc, config thinking.ThinkingConfig, modelInfo *registry.ModelInfo) error {
	if doc == nil || !supportedMode(config.Mode) {
		return nil
	}
	if thinking.IsUserDefinedModel(modelInfo) {
		if config.Mode == thinking.ModeLevel || (config.Mode == thinking.ModeNone && config.Level != "") {
			return a.applyLevelFormatDoc(doc, config)
		}
		return a.applyBudgetFormatDoc(doc, config)
	}
	if modelInfo.Thinking == nil {
		return nil
	}
	// ModeAuto and ModeBudget always use the budget format (auto is thinkingBudget=-1).
	if config.Mode == thinking.ModeAuto || config.Mode == thinking.ModeBudget {
		return a.applyBudgetFormatDoc(doc, config)
	}
	// For other modes, choose the format from the model's capabilities.
	if len(modelInfo.Thinking.Levels) > 0 {
		return a.applyLevelFormatDoc(doc, config)
	}
	return a.applyBudgetFormatDoc(doc, config)
}

// wireThinkingLevel renders a canonical thinking level in the casing the Cloud
// Code Assist backend expects. thinkingLevel is a proto enum whose JSON names are
// upper case -- see the ThinkingLevel enum in googleapis/js-genai ("HIGH", "LOW",
// "MEDIUM", "MINIMAL"), which Google's own gemini-cli sends via ThinkingLevel.HIGH.
// A lower-case value is not a recognised enum name, so the backend discards the
// thinking configuration and the model returns no thinking at all.
func wireThinkingLevel(level thinking.ThinkingLevel) string {
	return strings.ToUpper(strings.TrimSpace(string(level)))
}

// explicitIncludeThoughts returns the caller's includeThoughts setting, in
// either camelCase or snake_case. It must be read before the snake_case field
// is removed.
func explicitIncludeThoughts(doc *ojson.Doc) (value bool, set bool) {
	if inc := doc.Get(thinkingConfigPath + "includeThoughts"); inc.Exists() {
		return inc.Bool(), true
	}
	if inc := doc.Get(thinkingConfigPath + "include_thoughts"); inc.Exists() {
		return inc.Bool(), true
	}
	return false, false
}

// deleteFields removes conflicting fields so the output carries exactly one of
// thinkingLevel and thinkingBudget, and normalizes includeThoughts to camelCase
// to avoid oneof conflicts in upstream JSON parsing.
func deleteFields(doc *ojson.Doc, fields ...string) {
	for _, field := range fields {
		_ = doc.Delete(thinkingConfigPath + field)
	}
}

func (a *Applier) applyLevelFormatDoc(doc *ojson.Doc, config thinking.ThinkingConfig) error {
	// Only ModeNone and ModeLevel apply here; budget conversion is done by the upper layer.
	if config.Mode != thinking.ModeNone && config.Mode != thinking.ModeLevel {
		return nil
	}
	includeThoughts, userSet := explicitIncludeThoughts(doc)
	deleteFields(doc, "thinkingBudget", "thinking_budget", "thinking_level", "include_thoughts")

	if config.Mode == thinking.ModeNone {
		_ = doc.Set(thinkingConfigPath+"includeThoughts", false)
		if config.Level != "" {
			_ = doc.Set(thinkingConfigPath+"thinkingLevel", wireThinkingLevel(config.Level))
		}
		return nil
	}

	_ = doc.Set(thinkingConfigPath+"thinkingLevel", wireThinkingLevel(config.Level))
	// Respect the caller's explicit includeThoughts; default to true.
	if !userSet {
		includeThoughts = true
	}
	_ = doc.Set(thinkingConfigPath+"includeThoughts", includeThoughts)
	return nil
}

func (a *Applier) applyBudgetFormatDoc(doc *ojson.Doc, config thinking.ThinkingConfig) error {
	includeThoughts, userSet := explicitIncludeThoughts(doc)
	deleteFields(doc, "thinkingLevel", "thinking_level", "thinking_budget", "include_thoughts")

	budget := config.Budget
	_ = doc.Set(thinkingConfigPath+"thinkingBudget", budget)

	// ModeNone always disables thought output, even when the budget is clamped
	// to the model minimum.
	if config.Mode == thinking.ModeNone {
		_ = doc.Set(thinkingConfigPath+"includeThoughts", false)
		return nil
	}
	if !userSet {
		includeThoughts = config.Mode == thinking.ModeAuto || budget > 0
	}
	_ = doc.Set(thinkingConfigPath+"includeThoughts", includeThoughts)
	return nil
}
