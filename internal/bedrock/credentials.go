package bedrock

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

// credentialTimeout bounds credential acquisition (an SSO token exchange,
// AssumeRole call or credential_process run). It never applies to the
// Bedrock request itself.
const credentialTimeout = 30 * time.Second

// credentialSource returns signing credentials and the region to call.
type credentialSource func(ctx context.Context) (aws.Credentials, string, error)

type profileKey struct {
	profile string
	region  string
}

var (
	profileConfigsMu sync.Mutex
	profileConfigs   = map[profileKey]aws.Config{}
)

// profileCredentials loads the AWS CLI profile once per profile and region and
// reuses its credential cache, so refreshable credentials (SSO, assume-role)
// are renewed by the SDK only when they expire.
func profileCredentials(settings Settings) credentialSource {
	key := profileKey{profile: settings.Profile, region: settings.Region}
	return func(ctx context.Context) (aws.Credentials, string, error) {
		awsCfg, errLoad := loadProfileConfig(ctx, key)
		if errLoad != nil {
			return aws.Credentials{}, "", errLoad
		}
		if awsCfg.Region == "" {
			return aws.Credentials{}, "", fmt.Errorf("bedrock: no region configured for AWS profile %q; set bedrock.region or the profile's region", key.profile)
		}
		retrieveCtx, cancel := context.WithTimeout(ctx, credentialTimeout)
		defer cancel()
		creds, errRetrieve := awsCfg.Credentials.Retrieve(retrieveCtx)
		if errRetrieve != nil {
			return aws.Credentials{}, "", fmt.Errorf("bedrock: retrieve credentials for AWS profile %q: %w", key.profile, errRetrieve)
		}
		return creds, awsCfg.Region, nil
	}
}

func loadProfileConfig(ctx context.Context, key profileKey) (aws.Config, error) {
	profileConfigsMu.Lock()
	defer profileConfigsMu.Unlock()
	if cached, ok := profileConfigs[key]; ok {
		return cached, nil
	}
	var opts []func(*awsconfig.LoadOptions) error
	if key.profile != "" {
		opts = append(opts, awsconfig.WithSharedConfigProfile(key.profile))
	}
	if key.region != "" {
		opts = append(opts, awsconfig.WithRegion(key.region))
	}
	loadCtx, cancel := context.WithTimeout(ctx, credentialTimeout)
	defer cancel()
	awsCfg, errLoad := awsconfig.LoadDefaultConfig(loadCtx, opts...)
	if errLoad != nil {
		return aws.Config{}, fmt.Errorf("bedrock: load AWS profile %q: %w", key.profile, errLoad)
	}
	if awsCfg.Credentials == nil {
		return aws.Config{}, fmt.Errorf("bedrock: AWS profile %q has no credentials", key.profile)
	}
	profileConfigs[key] = awsCfg
	return awsCfg, nil
}
