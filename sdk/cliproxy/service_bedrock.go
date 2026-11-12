package cliproxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/bedrock"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
	log "github.com/sirupsen/logrus"
)

// bedrockDiscoveries tracks credentials with a discovery in flight, so repeated
// registrations do not start parallel discoveries for one credential.
var bedrockDiscoveries sync.Map // auth ID -> struct{}

// bedrockDiscoveredModels returns the models to register for a Bedrock entry
// that lists no models: the Claude inference profiles discovered in the account,
// under their Anthropic names, with the same catalog metadata a configured model
// of that name gets. When nothing fresh is cached it starts a background
// discovery that re-registers the credential once it completes.
func (s *Service) bedrockDiscoveredModels(cfg *config.Config, auth *coreauth.Auth) []*ModelInfo {
	ids, fresh, ok := bedrock.CachedModelIDs(auth.Attributes)
	if !ok {
		return nil
	}
	if !fresh {
		s.queueBedrockDiscovery(cfg, auth)
	}
	entries := make([]internalconfig.ClaudeModel, 0, len(ids))
	for _, id := range ids {
		entries = append(entries, internalconfig.ClaudeModel{Name: id})
	}
	return buildConfigModels(entries, "anthropic", "claude", "claude")
}

// Retry delays after a failed discovery. A failure is often temporary (an SSO
// session not logged in yet, a network blip), and without a retry the entry
// would serve no models until the next restart or config reload.
var (
	bedrockDiscoveryRetryInitial = 30 * time.Second
	bedrockDiscoveryRetryMax     = 10 * time.Minute
)

func (s *Service) queueBedrockDiscovery(cfg *config.Config, auth *coreauth.Auth) {
	if s == nil || auth == nil || auth.ID == "" {
		return
	}
	if _, running := bedrockDiscoveries.LoadOrStore(auth.ID, struct{}{}); running {
		return
	}
	go s.runBedrockDiscovery(cfg, auth.Clone())
}

// runBedrockDiscovery discovers the credential's models, retrying with backoff
// until it succeeds, the credential is removed or disabled, or the service stops.
// Each retry uses the current credential and config, so fixing the AWS profile
// and reloading takes effect.
func (s *Service) runBedrockDiscovery(cfg *config.Config, auth *coreauth.Auth) {
	defer bedrockDiscoveries.Delete(auth.ID)
	ctx := s.bedrockDiscoveryContext()
	delay := bedrockDiscoveryRetryInitial
	for {
		errDiscover := discoverBedrockModels(ctx, cfg, auth)
		if ctx.Err() != nil {
			return
		}
		var current *coreauth.Auth
		if s.coreManager != nil {
			if found, exists := s.coreManager.GetByID(auth.ID); exists && found != nil && !found.Disabled {
				current = found
			}
		}
		if errDiscover == nil {
			if current != nil {
				s.refreshModelRegistrationForAuthWithCache(current, nil)
			}
			return
		}
		if current == nil {
			log.Warnf("bedrock: model discovery for %s failed: %v", auth.ID, errDiscover)
			return
		}
		log.Warnf("bedrock: model discovery for %s failed, retrying in %s (check the AWS profile login and the bedrock:ListInferenceProfiles permission, or list models under the entry's models): %v", auth.ID, delay, errDiscover)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
		delay = min(delay*2, bedrockDiscoveryRetryMax)
		auth = current.Clone()
		if latest := s.currentConfig(); latest != nil {
			cfg = latest
		}
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
