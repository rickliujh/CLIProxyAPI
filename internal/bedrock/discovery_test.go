package bedrock

import (
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MoeForge/moego/ojson"
)

// fakeInferenceProfiles serves two pages of ListInferenceProfiles and accepts
// InvokeModel calls; it counts control-plane calls.
func fakeInferenceProfiles(t *testing.T, listCalls *atomic.Int32) (string, func() []capturedRequest) {
	t.Helper()
	server, requests := fakeBedrock(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/inference-profiles" {
			_, _ = w.Write([]byte(`{"type":"message"}`))
			return
		}
		listCalls.Add(1)
		if r.URL.Query().Get("typeEquals") != "SYSTEM_DEFINED" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if r.URL.Query().Get("nextToken") == "" {
			_, _ = w.Write([]byte(`{"nextToken":"page2","inferenceProfileSummaries":[
				{"inferenceProfileId":"eu.anthropic.claude-sonnet-4-5-20250929-v1:0","inferenceProfileName":"EU Claude Sonnet 4.5","status":"ACTIVE"},
				{"inferenceProfileId":"us.anthropic.claude-haiku-4-5-20251001-v1:0","inferenceProfileName":"US Claude Haiku 4.5","status":"ACTIVE"},
				{"inferenceProfileId":"eu.meta.llama3-2-3b-instruct-v1:0","inferenceProfileName":"EU Llama","status":"ACTIVE"},
				{"inferenceProfileId":"eu.anthropic.claude-dup-v1:0","inferenceProfileName":"Dup v1","status":"ACTIVE"}
			]}`))
			return
		}
		_, _ = w.Write([]byte(`{"inferenceProfileSummaries":[
			{"inferenceProfileId":"eu.anthropic.claude-opus-4-6-v1","inferenceProfileName":"EU Claude Opus 4.6","status":"ACTIVE"},
			{"inferenceProfileId":"eu.anthropic.claude-dup-v2:0","inferenceProfileName":"Dup v2","status":"ACTIVE"},
			{"inferenceProfileId":"eu.anthropic.claude-retired-v1:0","inferenceProfileName":"Retired","status":"LEGACY"}
		]}`))
	})
	return server.URL, requests
}

func newDiscoveryTransport(endpoint, profile string, now *time.Time) *transport {
	tr := newTransport(Settings{Profile: profile, Endpoint: endpoint, ControlEndpoint: endpoint}, http.DefaultTransport, staticCredentials("eu-central-1"))
	tr.now = func() time.Time { return *now }
	return tr
}

func TestListModelsDiscoversEUClaudeProfiles(t *testing.T) {
	var listCalls atomic.Int32
	endpoint, requests := fakeInferenceProfiles(t, &listCalls)
	now := time.Now()
	tr := newDiscoveryTransport(endpoint, "discovery-list", &now)

	req, _ := http.NewRequest(http.MethodGet, "https://bedrock-runtime.amazonaws.com/v1/models", nil)
	resp, errDo := tr.RoundTrip(req)
	if errDo != nil {
		t.Fatal(errDo)
	}
	body := readAll(t, resp)
	var ids []string
	for _, model := range ojson.Get(body, "data").Array() {
		ids = append(ids, model.Get("id").String())
	}
	if want := "claude-dup,claude-opus-4-6,claude-sonnet-4-5-20250929"; strings.Join(ids, ",") != want {
		t.Fatalf("ids = %v, want %s (body %s)", ids, want, body)
	}
	if listCalls.Load() != 2 {
		t.Fatalf("ListInferenceProfiles calls = %d, want 2 pages", listCalls.Load())
	}
	if auth := requests()[0].header.Get("Authorization"); !strings.Contains(auth, "/eu-central-1/bedrock/aws4_request") {
		t.Fatalf("Authorization = %s", auth)
	}

	cached, fresh, ok := CachedModelIDs(tr.settings.Attributes())
	if !ok || !fresh || len(cached) != 3 {
		t.Fatalf("cached = %v fresh=%v ok=%v", cached, fresh, ok)
	}
	models, _, _ := discoveryEntryFor(tr.settings).snapshot()
	for _, model := range models {
		if model.ID == "claude-dup" && model.ProfileID != "eu.anthropic.claude-dup-v2:0" {
			t.Fatalf("duplicate model kept %s, want the newest version", model.ProfileID)
		}
	}
}

func TestMessagesMapAnthropicNamesToDiscoveredProfiles(t *testing.T) {
	var listCalls atomic.Int32
	endpoint, requests := fakeInferenceProfiles(t, &listCalls)
	now := time.Now()
	tr := newDiscoveryTransport(endpoint, "discovery-map", &now)

	resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"claude-opus-4-6","max_tokens":8}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	_ = readAll(t, resp)
	var invoked string
	for _, sent := range requests() {
		if strings.HasPrefix(sent.rawPath, "/model/") {
			invoked = sent.rawPath
		}
	}
	if want := "/model/eu.anthropic.claude-opus-4-6-v1/invoke"; invoked != want {
		t.Fatalf("invoked %s, want %s", invoked, want)
	}

	// A Bedrock ID passes through without discovery.
	calls := listCalls.Load()
	resp, errDo = tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"eu.anthropic.claude-sonnet-4-5-20250929-v1:0"}`))
	if errDo != nil {
		t.Fatal(errDo)
	}
	_ = readAll(t, resp)
	if listCalls.Load() != calls {
		t.Fatal("a Bedrock model ID must not trigger discovery")
	}
}

func TestUnknownModelIsNotFoundAndRediscoveryIsRateLimited(t *testing.T) {
	var listCalls atomic.Int32
	endpoint, _ := fakeInferenceProfiles(t, &listCalls)
	now := time.Now()
	tr := newDiscoveryTransport(endpoint, "discovery-unknown", &now)

	send := func() *http.Response {
		t.Helper()
		resp, errDo := tr.RoundTrip(anthropicRequest(t, "/v1/messages", `{"model":"claude-unknown-9"}`))
		if errDo != nil {
			t.Fatal(errDo)
		}
		return resp
	}
	resp := send()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if body := readAll(t, resp); ojson.Get(body, "error.type").String() != "not_found_error" || !strings.Contains(body, "no eu inference profile") {
		t.Fatalf("body = %s", body)
	}
	afterFirst := listCalls.Load()

	_ = readAll(t, send())
	if listCalls.Load() != afterFirst {
		t.Fatal("an unknown model must not rediscover within rediscoverAfter")
	}
	now = now.Add(rediscoverAfter + time.Second)
	_ = readAll(t, send())
	if listCalls.Load() == afterFirst {
		t.Fatal("an unknown model should rediscover once the cache is older than rediscoverAfter")
	}
}

func TestGeoSelection(t *testing.T) {
	for region, want := range map[string]string{"eu-central-1": "eu", "eu-west-3": "eu", "us-east-1": "us", "ap-northeast-1": "apac"} {
		if got := defaultGeo(region); got != want {
			t.Fatalf("defaultGeo(%s) = %s, want %s", region, got, want)
		}
	}
	tr := newTransport(Settings{Geo: "global"}, http.DefaultTransport, staticCredentials("eu-central-1"))
	if tr.geo("eu-central-1") != "global" {
		t.Fatal("a configured geo must win over the region")
	}
	if !needsModelMapping("claude-opus-4-6") || needsModelMapping("eu.anthropic.claude-opus-4-6-v1") || needsModelMapping("arn:aws:bedrock:eu-central-1:1:application-inference-profile/x") {
		t.Fatal("needsModelMapping misclassified a model")
	}
}
