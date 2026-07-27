package geminicli

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/tidwall/gjson"
)

func gemini3Model() *registry.ModelInfo {
	return &registry.ModelInfo{
		ID: "gemini-3.5-flash",
		Thinking: &registry.ThinkingSupport{
			Min:            128,
			Max:            32768,
			DynamicAllowed: true,
			Levels:         []string{"minimal", "low", "medium", "high"},
		},
	}
}

// thinkingLevel is a proto enum: its JSON names are upper case. Sending "high"
// instead of "HIGH" is not a recognised enum name, and the Cloud Code Assist
// backend then discards the thinking configuration entirely -- observed as a
// response with thoughtsTokenCount=0 despite thinkingLevel being requested.
func TestApplyLevelUsesUpperCaseEnumName(t *testing.T) {
	for _, level := range []thinking.ThinkingLevel{"minimal", "low", "medium", "high"} {
		out, err := NewApplier().Apply([]byte(`{}`), thinking.ThinkingConfig{
			Mode:  thinking.ModeLevel,
			Level: level,
		}, gemini3Model())
		if err != nil {
			t.Fatalf("apply %q: %v", level, err)
		}
		got := gjson.GetBytes(out, "request.generationConfig.thinkingConfig.thinkingLevel").String()
		want := map[thinking.ThinkingLevel]string{
			"minimal": "MINIMAL",
			"low":     "LOW",
			"medium":  "MEDIUM",
			"high":    "HIGH",
		}[level]
		if got != want {
			t.Errorf("level %q: thinkingLevel = %q, want %q (body=%s)", level, got, want, out)
		}
		if !gjson.GetBytes(out, "request.generationConfig.thinkingConfig.includeThoughts").Bool() {
			t.Errorf("level %q: includeThoughts should default to true (body=%s)", level, out)
		}
	}
}

// ModeNone still names a level when one is carried, and it must use the same casing.
func TestApplyNoneUsesUpperCaseEnumName(t *testing.T) {
	out, err := NewApplier().Apply([]byte(`{}`), thinking.ThinkingConfig{
		Mode:   thinking.ModeNone,
		Level:  "minimal",
		Budget: 128,
	}, gemini3Model())
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if got := gjson.GetBytes(out, "request.generationConfig.thinkingConfig.thinkingLevel").String(); got != "MINIMAL" {
		t.Errorf("thinkingLevel = %q, want %q (body=%s)", got, "MINIMAL", out)
	}
	if gjson.GetBytes(out, "request.generationConfig.thinkingConfig.includeThoughts").Bool() {
		t.Errorf("ModeNone must not include thoughts (body=%s)", out)
	}
}

// The budget format is numeric and must be untouched by the casing change.
func TestApplyBudgetFormatUnaffected(t *testing.T) {
	model := &registry.ModelInfo{
		ID:       "gemini-2.5-flash",
		Thinking: &registry.ThinkingSupport{Max: 24576, ZeroAllowed: true, DynamicAllowed: true},
	}
	out, err := NewApplier().Apply([]byte(`{}`), thinking.ThinkingConfig{
		Mode:   thinking.ModeBudget,
		Budget: 8192,
	}, model)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if got := gjson.GetBytes(out, "request.generationConfig.thinkingConfig.thinkingBudget").Int(); got != 8192 {
		t.Errorf("thinkingBudget = %d, want 8192 (body=%s)", got, out)
	}
	if gjson.GetBytes(out, "request.generationConfig.thinkingConfig.thinkingLevel").Exists() {
		t.Errorf("budget format must not emit thinkingLevel (body=%s)", out)
	}
}
