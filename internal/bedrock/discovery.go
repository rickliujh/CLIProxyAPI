package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/aws/aws-sdk-go-v2/aws"
	log "github.com/sirupsen/logrus"
)

const (
	// discoveryTTL is how long discovered profiles count as fresh for model
	// registration.
	discoveryTTL = 6 * time.Hour
	// rediscoverAfter limits how often a model listing or an unknown model name
	// triggers another discovery.
	rediscoverAfter = 5 * time.Minute
	// maxDiscoveryPages bounds ListInferenceProfiles paging.
	maxDiscoveryPages = 50
	// emptyPayloadHash is the SHA-256 of an empty body, for signing GET requests.
	emptyPayloadHash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// profileVersionSuffix matches the version suffix of a Bedrock model ID, e.g.
// "-v1:0" or "-v1".
var profileVersionSuffix = regexp.MustCompile(`-v\d+(:\d+)?$`)

// DiscoveredModel is a Claude inference profile found in the account.
type DiscoveredModel struct {
	// ID is the client-facing Anthropic model name, e.g. "claude-sonnet-4-5-20250929".
	ID string
	// ProfileID is the Bedrock inference profile ID, e.g.
	// "eu.anthropic.claude-sonnet-4-5-20250929-v1:0".
	ProfileID string
	// Name is the inference profile's display name.
	Name string
}

type discoveryKey struct {
	profile string
	region  string
	geo     string
	control string
}

// discoveryEntry caches one credential's discovered profiles. run serializes
// discovery so concurrent requests share one ListInferenceProfiles call.
type discoveryEntry struct {
	run     sync.Mutex
	mu      sync.RWMutex
	models  []DiscoveredModel
	fetched time.Time
	ok      bool
}

var discoveries sync.Map // discoveryKey -> *discoveryEntry

func discoveryKeyFor(settings Settings) discoveryKey {
	return discoveryKey{profile: settings.Profile, region: settings.Region, geo: settings.Geo, control: settings.ControlEndpoint}
}

func discoveryEntryFor(settings Settings) *discoveryEntry {
	value, _ := discoveries.LoadOrStore(discoveryKeyFor(settings), &discoveryEntry{})
	return value.(*discoveryEntry)
}

func (e *discoveryEntry) snapshot() ([]DiscoveredModel, time.Time, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.models, e.fetched, e.ok
}

func (e *discoveryEntry) store(models []DiscoveredModel, fetched time.Time) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.models, e.fetched, e.ok = models, fetched, true
}

// CachedModelIDs returns the client-facing model names last discovered for a
// Bedrock credential and whether that result is still fresh. ok is false for
// credentials that do not target Bedrock.
func CachedModelIDs(attrs map[string]string) (ids []string, fresh bool, ok bool) {
	settings, isBedrock := SettingsFromAttributes(attrs)
	if !isBedrock {
		return nil, false, false
	}
	value, exists := discoveries.Load(discoveryKeyFor(settings))
	if !exists {
		return nil, false, true
	}
	models, fetched, discovered := value.(*discoveryEntry).snapshot()
	if !discovered {
		return nil, false, true
	}
	ids = make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids, time.Since(fetched) < discoveryTTL, true
}

// defaultGeo derives the inference profile geography from a region.
func defaultGeo(region string) string {
	prefix, _, _ := strings.Cut(strings.ToLower(region), "-")
	if prefix == "ap" {
		return "apac"
	}
	return prefix
}

func (t *transport) geo(region string) string {
	if t.settings.Geo != "" {
		return t.settings.Geo
	}
	return defaultGeo(region)
}

// discover returns the credential's Claude inference profiles, running
// ListInferenceProfiles when nothing is cached or the cache is older than
// maxAge. A failed refresh keeps serving the previous result.
func (t *transport) discover(ctx context.Context, creds aws.Credentials, region string, maxAge time.Duration) ([]DiscoveredModel, error) {
	entry := discoveryEntryFor(t.settings)
	if models, fetched, ok := entry.snapshot(); ok && t.now().Sub(fetched) < maxAge {
		return models, nil
	}
	entry.run.Lock()
	defer entry.run.Unlock()
	if models, fetched, ok := entry.snapshot(); ok && t.now().Sub(fetched) < maxAge {
		return models, nil
	}
	models, errList := t.listInferenceProfiles(ctx, creds, region)
	if errList != nil {
		if cached, _, ok := entry.snapshot(); ok {
			log.Warnf("bedrock: refresh inference profiles for AWS profile %q failed, keeping previous result: %v", t.settings.Profile, errList)
			return cached, nil
		}
		return nil, errList
	}
	entry.store(models, t.now())
	log.Infof("bedrock: discovered %d %s Claude inference profile(s) for AWS profile %q in %s", len(models), t.geo(region), t.settings.Profile, region)
	return models, nil
}

// resolveModel maps a client-facing Anthropic model name to its discovered
// inference profile ID. found is false when the account has no such profile.
func (t *transport) resolveModel(ctx context.Context, creds aws.Credentials, region, model string) (string, bool, error) {
	lookup := func(models []DiscoveredModel) (string, bool) {
		for _, discovered := range models {
			if strings.EqualFold(discovered.ID, model) {
				return discovered.ProfileID, true
			}
		}
		return "", false
	}
	models, errDiscover := t.discover(ctx, creds, region, discoveryTTL)
	if errDiscover != nil {
		return "", false, errDiscover
	}
	if profileID, found := lookup(models); found {
		return profileID, true, nil
	}
	// The model may be newer than the cached list.
	if models, errDiscover = t.discover(ctx, creds, region, rediscoverAfter); errDiscover != nil {
		return "", false, errDiscover
	}
	profileID, found := lookup(models)
	return profileID, found, nil
}

// needsModelMapping reports whether a model name is a client-facing Anthropic
// name rather than a Bedrock model ID, inference profile ID or ARN.
func needsModelMapping(model string) bool {
	return !strings.Contains(model, ".") && !strings.HasPrefix(model, "arn:")
}

func (t *transport) controlEndpoint(region string) string {
	if t.settings.ControlEndpoint != "" {
		return t.settings.ControlEndpoint
	}
	return "https://bedrock." + region + ".amazonaws.com"
}

// listInferenceProfiles calls ListInferenceProfiles and keeps the active Claude
// profiles of the credential's geography, one per client-facing model name.
func (t *transport) listInferenceProfiles(ctx context.Context, creds aws.Credentials, region string) ([]DiscoveredModel, error) {
	prefix := t.geo(region) + ".anthropic."
	byID := make(map[string]DiscoveredModel)
	nextToken := ""
	for page := 0; page < maxDiscoveryPages; page++ {
		query := url.Values{}
		query.Set("maxResults", "1000")
		query.Set("typeEquals", "SYSTEM_DEFINED")
		if nextToken != "" {
			query.Set("nextToken", nextToken)
		}
		target := strings.TrimRight(t.controlEndpoint(region), "/") + "/inference-profiles?" + query.Encode()
		req, errNew := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
		if errNew != nil {
			return nil, fmt.Errorf("bedrock: build ListInferenceProfiles request: %w", errNew)
		}
		req.Header.Set("Accept", "application/json")
		if errSign := t.signer.SignHTTP(ctx, creds, req, emptyPayloadHash, signingService, region, t.now()); errSign != nil {
			return nil, fmt.Errorf("bedrock: sign ListInferenceProfiles request: %w", errSign)
		}
		resp, errDo := t.base.RoundTrip(req)
		if errDo != nil {
			return nil, fmt.Errorf("bedrock: ListInferenceProfiles: %w", errDo)
		}
		body, errRead := io.ReadAll(resp.Body)
		closeResponseBody(resp)
		if errRead != nil {
			return nil, fmt.Errorf("bedrock: read ListInferenceProfiles response: %w", errRead)
		}
		if resp.StatusCode != http.StatusOK {
			message := strings.TrimSpace(ojson.GetBytes(body, "message").String())
			if message == "" {
				message = strings.TrimSpace(string(body))
			}
			return nil, fmt.Errorf("bedrock: ListInferenceProfiles returned %d: %s", resp.StatusCode, message)
		}
		for _, summary := range ojson.GetBytes(body, "inferenceProfileSummaries").Array() {
			profileID := summary.Get("inferenceProfileId").String()
			if status := summary.Get("status").String(); status != "" && !strings.EqualFold(status, "ACTIVE") {
				continue
			}
			if !strings.HasPrefix(profileID, prefix) {
				continue
			}
			id := profileVersionSuffix.ReplaceAllString(strings.TrimPrefix(profileID, prefix), "")
			// Keep the newest version when one model has several profiles.
			if existing, dup := byID[id]; dup && existing.ProfileID > profileID {
				continue
			}
			byID[id] = DiscoveredModel{ID: id, ProfileID: profileID, Name: summary.Get("inferenceProfileName").String()}
		}
		nextToken = ojson.GetBytes(body, "nextToken").String()
		if nextToken == "" {
			break
		}
	}
	models := make([]DiscoveredModel, 0, len(byID))
	for _, model := range byID {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	return models, nil
}

// modelListResponse renders discovered profiles as an Anthropic GET /v1/models body.
func modelListResponse(models []DiscoveredModel) []byte {
	type modelEntry struct {
		Type        string `json:"type"`
		ID          string `json:"id"`
		DisplayName string `json:"display_name"`
		CreatedAt   string `json:"created_at"`
	}
	data := make([]modelEntry, 0, len(models))
	for _, model := range models {
		data = append(data, modelEntry{Type: "model", ID: model.ID, DisplayName: model.Name, CreatedAt: "1970-01-01T00:00:00Z"})
	}
	body := map[string]any{"data": data, "has_more": false, "first_id": nil, "last_id": nil}
	if len(data) > 0 {
		body["first_id"] = data[0].ID
		body["last_id"] = data[len(data)-1].ID
	}
	out, _ := json.Marshal(body)
	return out
}
