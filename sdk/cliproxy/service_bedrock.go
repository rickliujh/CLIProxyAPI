package cliproxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
	log "github.com/sirupsen/logrus"
)

// bedrockDiscoveries tracks credentials with a discovery loop running, so
// repeated registrations do not start parallel loops for one credential.
var bedrockDiscoveries sync.Map // auth ID -> struct{}

// bedrockDiscoveredModels returns the models to register for a Bedrock entry
// that lists no models: the Claude inference profiles discovered in the account,
// under their Anthropic names, with the same catalog metadata a configured model
// of that name gets. It makes sure a background loop keeps the list current and
// re-registers the credential whenever the list changes.
func (s *Service) bedrockDiscoveredModels(cfg *config.Config, auth *coreauth.Auth) []*ModelInfo {
	ids, _, ok := bedrock.CachedModelIDs(auth.Attributes)
	if !ok {
		return nil
	}
	s.queueBedrockDiscovery(cfg, auth)
	entries := make([]internalconfig.ClaudeModel, 0, len(ids))
	for _, id := range ids {
		entries = append(entries, internalconfig.ClaudeModel{Name: id})
	}
	return buildConfigModels(entries, "anthropic", "claude", "claude")
}

// Retry delays after a failed discovery. A failure is often temporary (an SSO
// session not logged in yet, a network blip), and without a retry the entry
// would serve no models, or a stale list, until the next restart.
// They are stored atomically because tests shorten them while loops run.
var (
	bedrockDiscoveryRetryInitial atomic.Int64
	bedrockDiscoveryRetryMax     atomic.Int64
)

func init() {
	bedrockDiscoveryRetryInitial.Store(int64(30 * time.Second))
	bedrockDiscoveryRetryMax.Store(int64(10 * time.Minute))
}

func (s *Service) queueBedrockDiscovery(cfg *config.Config, auth *coreauth.Auth) {
	if s == nil || auth == nil || auth.ID == "" {
		return
	}
	if _, running := bedrockDiscoveries.LoadOrStore(auth.ID, struct{}{}); running {
		return
	}
	go s.runBedrockDiscovery(cfg, auth.Clone())
}

// runBedrockDiscovery keeps one credential's discovered models current. It
// discovers when nothing is cached or the cached list is older than
// bedrock.RefreshInterval(), re-registers the credential when the list changes,
// and retries a failure with backoff. It runs until the credential is removed
// or disabled, or the service stops. Each round uses the current credential
// and config, so fixing the AWS profile and reloading takes effect.
func (s *Service) runBedrockDiscovery(cfg *config.Config, auth *coreauth.Auth) {
	defer bedrockDiscoveries.Delete(auth.ID)
	ctx := s.bedrockDiscoveryContext()
	delay := time.Duration(bedrockDiscoveryRetryInitial.Load())
	for {
		before, fetchedBefore, _ := bedrock.CachedModelIDs(auth.Attributes)
		if !fetchedBefore.IsZero() {
			// Another credential with the same settings may have refreshed the
			// shared list, so re-check once the wait is over.
			if wait := time.Until(fetchedBefore.Add(bedrock.RefreshInterval())); wait > 0 {
				if !sleepContext(ctx, wait) {
					return
				}
				if !s.refreshBedrockRound(&cfg, &auth) {
					return
				}
				continue
			}
		}

		errDiscover := discoverBedrockModels(ctx, cfg, auth)
		if ctx.Err() != nil {
			return
		}
		after, fetchedAfter, _ := bedrock.CachedModelIDs(auth.Attributes)
		if errDiscover == nil && !fetchedAfter.After(fetchedBefore) {
			// The transport served the previous list because the refresh failed.
			errDiscover = fmt.Errorf("refresh failed; still serving the list from %s", fetchedBefore.Format(time.RFC3339))
		}
		current := s.activeBedrockAuth(auth.ID)
		if current == nil {
			if errDiscover != nil {
				log.Warnf("bedrock: model discovery for %s failed: %v", auth.ID, errDiscover)
			}
			return
		}
		if errDiscover == nil {
			delay = time.Duration(bedrockDiscoveryRetryInitial.Load())
			if !slices.Equal(before, after) {
				s.refreshModelRegistrationForAuthWithCache(current, nil)
			}
			auth = current.Clone()
			if latest := s.currentConfig(); latest != nil {
				cfg = latest
			}
			continue
		}
		log.Warnf("bedrock: model discovery for %s failed, retrying in %s (check the AWS profile login and the bedrock:ListInferenceProfiles permission, or list models under the entry's models): %v", auth.ID, delay, errDiscover)
		if !sleepContext(ctx, delay) {
			return
		}
		delay = min(delay*2, time.Duration(bedrockDiscoveryRetryMax.Load()))
		if !s.refreshBedrockRound(&cfg, &auth) {
			return
		}
	}
}

// activeBedrockAuth returns the credential when it still exists and is enabled.
func (s *Service) activeBedrockAuth(id string) *coreauth.Auth {
	if s.coreManager == nil {
		return nil
	}
	if found, exists := s.coreManager.GetByID(id); exists && found != nil && !found.Disabled {
		return found
	}
	return nil
}

// refreshBedrockRound loads the current credential and config for the next
// round; false means the credential is gone and the loop should stop.
func (s *Service) refreshBedrockRound(cfg **config.Config, auth **coreauth.Auth) bool {
	current := s.activeBedrockAuth((*auth).ID)
	if current == nil {
		return false
	}
	*auth = current.Clone()
	if latest := s.currentConfig(); latest != nil {
		*cfg = latest
	}
	return true
}

// sleepContext waits for d; false means ctx ended first.
func sleepContext(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// bedrockDiscoveryContext is the service's run context, so retries stop when
// the service stops.
func (s *Service) bedrockDiscoveryContext() context.Context {
	s.cfgMu.RLock()
	ctx := s.antigravityContext
	s.cfgMu.RUnlock()
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

// discoverBedrockModels lists models through the credential's own HTTP client,
// whose Bedrock transport answers GET /v1/models from ListInferenceProfiles and
// caches the result for registration and request routing.
func discoverBedrockModels(ctx context.Context, cfg *config.Config, auth *coreauth.Auth) error {
	baseURL := strings.TrimRight(strings.TrimSpace(auth.Attributes["base_url"]), "/")
	if baseURL == "" {
		baseURL = internalconfig.BedrockDefaultBaseURL
	}
	req, errNew := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil)
	if errNew != nil {
		return errNew
	}
	resp, errDo := helps.NewUtlsHTTPClient(ctx, cfg, auth, 0).Do(req)
	if errDo != nil {
		return errDo
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Debugf("bedrock: close model list body: %v", errClose)
		}
	}()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("model list returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	_, errDrain := io.Copy(io.Discard, resp.Body)
	return errDrain
}
