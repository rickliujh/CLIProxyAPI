package cliproxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
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

// A failed discovery is retried on its own, so an entry whose first discovery
// fails (an SSO session not logged in yet, say) still gets its models without
// a restart or config reload.
func TestBedrockDiscoveryRetriesAfterFailure(t *testing.T) {
	writeServiceBedrockProfile(t, "service-bedrock-retry")
	initial := bedrockDiscoveryRetryInitial.Swap(int64(time.Millisecond))
	maxDelay := bedrockDiscoveryRetryMax.Swap(int64(2 * time.Millisecond))
	t.Cleanup(func() {
		bedrockDiscoveryRetryInitial.Store(initial)
		bedrockDiscoveryRetryMax.Store(maxDelay)
	})

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) <= 2 {
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"The SSO session associated with this profile has expired"}`))
			return
		}
		_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[{"inferenceProfileId":"eu.anthropic.claude-haiku-5-5-v1:0","status":"ACTIVE"}]}`))
	}))
	defer server.Close()

	cfg := &config.Config{ClaudeKey: []config.ClaudeKey{{Bedrock: &internalconfig.BedrockConfig{
		Profile: "service-bedrock-retry", Region: "eu-central-1", Endpoint: server.URL, ControlEndpoint: server.URL,
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
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline) && len(ids) == 0; time.Sleep(10 * time.Millisecond) {
		ids = ids[:0]
		for _, model := range GlobalModelRegistry().GetModelsForClient(auth.ID) {
			ids = append(ids, model.ID)
		}
	}
	if strings.Join(ids, ",") != "claude-haiku-5-5" || calls.Load() != 3 {
		t.Fatalf("registered models = %v after %d discovery calls, want claude-haiku-5-5 after 3", ids, calls.Load())
	}
}

// Models AWS adds while the proxy runs appear without a restart: the discovered
// list is refreshed in the background and the credential is re-registered when
// the list changes.
func TestBedrockDiscoveryRefreshesInBackground(t *testing.T) {
	writeServiceBedrockProfile(t, "service-bedrock-refresh")
	t.Cleanup(bedrock.SetDiscoveryIntervals(20*time.Millisecond, 0))

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[{"inferenceProfileId":"eu.anthropic.claude-haiku-5-5-v1:0","status":"ACTIVE"}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[{"inferenceProfileId":"eu.anthropic.claude-haiku-5-5-v1:0","status":"ACTIVE"},{"inferenceProfileId":"eu.anthropic.claude-sonnet-5-5-v1:0","status":"ACTIVE"}]}`))
	}))
	defer server.Close()

	cfg := &config.Config{ClaudeKey: []config.ClaudeKey{{Bedrock: &internalconfig.BedrockConfig{
		Profile: "service-bedrock-refresh", Region: "eu-central-1", Endpoint: server.URL, ControlEndpoint: server.URL,
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
	t.Cleanup(func() {
		// Stop the background loop before the next test reuses the registry.
		manager.Remove(context.Background(), auth.ID)
		GlobalModelRegistry().UnregisterClient(auth.ID)
	})

	svc.registerModelsForAuth(t.Context(), auth)

	var ids []string
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		ids = ids[:0]
		for _, model := range GlobalModelRegistry().GetModelsForClient(auth.ID) {
			ids = append(ids, model.ID)
		}
		sort.Strings(ids)
		if len(ids) == 2 {
			break
		}
	}
	if strings.Join(ids, ",") != "claude-haiku-5-5,claude-sonnet-5-5" || calls.Load() < 2 {
		t.Fatalf("registered models = %v after %d discovery calls, want the model added later to appear", ids, calls.Load())
	}
}
