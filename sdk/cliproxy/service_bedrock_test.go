package cliproxy

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/watcher/synthesizer"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

func writeServiceBedrockProfile(t *testing.T, profile string) {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config")
	content := "[profile " + profile + "]\nregion = eu-central-1\naws_access_key_id = AKIDSVC\naws_secret_access_key = svc-secret\n"
	if errWrite := os.WriteFile(configPath, []byte(content), 0o600); errWrite != nil {
		t.Fatal(errWrite)
	}
	t.Setenv("AWS_CONFIG_FILE", configPath)
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(dir, "credentials"))
	for _, name := range []string{"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"} {
		t.Setenv(name, "")
	}
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
}

// A Bedrock entry without models registers the Claude inference profiles
// discovered in the account once the background discovery completes.
func TestBedrockEntryWithoutModelsRegistersDiscoveredProfiles(t *testing.T) {
	writeServiceBedrockProfile(t, "service-bedrock-discovery")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/inference-profiles" || !strings.Contains(r.Header.Get("Authorization"), "Credential=AKIDSVC/") {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[
			{"inferenceProfileId":"eu.anthropic.claude-sonnet-4-5-20250929-v1:0","inferenceProfileName":"EU Claude Sonnet 4.5","status":"ACTIVE"},
			{"inferenceProfileId":"eu.anthropic.claude-opus-4-6-v1","inferenceProfileName":"EU Claude Opus 4.6","status":"ACTIVE"},
			{"inferenceProfileId":"us.anthropic.claude-haiku-4-5-20251001-v1:0","inferenceProfileName":"US Claude Haiku 4.5","status":"ACTIVE"}
		]}`))
	}))
	defer server.Close()

	cfg := &config.Config{ClaudeKey: []config.ClaudeKey{{Bedrock: &internalconfig.BedrockConfig{
		Profile: "service-bedrock-discovery", Region: "eu-central-1", Endpoint: server.URL, ControlEndpoint: server.URL,
	}}}}
	auths, errSynth := synthesizer.NewConfigSynthesizer().Synthesize(&synthesizer.SynthesisContext{Config: cfg, Now: time.Now(), IDGenerator: synthesizer.NewStableIDGenerator()})
	if errSynth != nil || len(auths) != 1 {
		t.Fatalf("synthesize: %v (%d auths)", errSynth, len(auths))
	}
	manager := coreauth.NewManager(nil, nil, nil)
	manager.SetConfig(cfg)
	svc := &Service{cfg: cfg, coreManager: manager}
	auth, errRegister := manager.Register(t.Context(), auths[0])
	if errRegister != nil {
		t.Fatal(errRegister)
	}
	t.Cleanup(func() { GlobalModelRegistry().UnregisterClient(auth.ID) })

	svc.registerModelsForAuth(t.Context(), auth)

	var ids []string
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		ids = ids[:0]
		for _, model := range GlobalModelRegistry().GetModelsForClient(auth.ID) {
			ids = append(ids, model.ID)
		}
		if len(ids) > 0 {
			break
		}
	}
	sort.Strings(ids)
	if want := "claude-opus-4-6,claude-sonnet-4-5-20250929"; strings.Join(ids, ",") != want {
		t.Fatalf("registered models = %v, want %s", ids, want)
	}
}
