package registry

import "testing"

// withEmbeddedCatalog swaps the process-wide embedded catalog for the duration
// of a test and restores it afterwards.
func withEmbeddedCatalog(t *testing.T, catalog *staticModelsJSON) {
	t.Helper()
	previous := embeddedCatalog
	embeddedCatalog = catalog
	t.Cleanup(func() { embeddedCatalog = previous })
}

func modelIDs(models []*ModelInfo) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

func TestMergeEmbeddedExtrasAddsModelsMissingFromRemote(t *testing.T) {
	withEmbeddedCatalog(t, &staticModelsJSON{
		GeminiCLI: []*ModelInfo{
			{ID: "gemini-2.5-pro"},
			{ID: "gemini-3.5-flash", DisplayName: "Gemini 3.5 Flash"},
		},
	})

	remote := &staticModelsJSON{
		GeminiCLI: []*ModelInfo{{ID: "gemini-2.5-pro"}},
	}
	mergeEmbeddedExtras(remote)

	got := modelIDs(remote.GeminiCLI)
	want := []string{"gemini-2.5-pro", "gemini-3.5-flash"}
	if len(got) != len(want) {
		t.Fatalf("gemini-cli models = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gemini-cli models = %v, want %v", got, want)
		}
	}
}

func TestMergeEmbeddedExtrasKeepsRemoteAuthoritative(t *testing.T) {
	withEmbeddedCatalog(t, &staticModelsJSON{
		GeminiCLI: []*ModelInfo{{ID: "gemini-2.5-pro", DisplayName: "stale embedded name"}},
	})

	remote := &staticModelsJSON{
		GeminiCLI: []*ModelInfo{{ID: "gemini-2.5-pro", DisplayName: "fresh remote name"}},
	}
	mergeEmbeddedExtras(remote)

	if len(remote.GeminiCLI) != 1 {
		t.Fatalf("expected no duplicate entries, got %v", modelIDs(remote.GeminiCLI))
	}
	if remote.GeminiCLI[0].DisplayName != "fresh remote name" {
		t.Fatalf("remote definition was overwritten: %q", remote.GeminiCLI[0].DisplayName)
	}
}

// A merged entry must be a copy, so later mutation of the served catalog cannot
// corrupt the embedded catalog that every subsequent refresh merges from.
func TestMergeEmbeddedExtrasDoesNotAliasEmbeddedCatalog(t *testing.T) {
	embedded := &staticModelsJSON{
		GeminiCLI: []*ModelInfo{{ID: "gemini-3.5-flash", DisplayName: "original"}},
	}
	withEmbeddedCatalog(t, embedded)

	remote := &staticModelsJSON{}
	mergeEmbeddedExtras(remote)

	remote.GeminiCLI[0].DisplayName = "mutated"
	if embedded.GeminiCLI[0].DisplayName != "original" {
		t.Fatalf("embedded catalog was mutated through the merged catalog: %q", embedded.GeminiCLI[0].DisplayName)
	}
}

// Repeated refreshes each start from a fresh remote parse, so the merge must not
// accumulate duplicates over time.
func TestMergeEmbeddedExtrasIsIdempotent(t *testing.T) {
	withEmbeddedCatalog(t, &staticModelsJSON{
		GeminiCLI: []*ModelInfo{{ID: "gemini-3.5-flash"}},
	})

	remote := &staticModelsJSON{}
	mergeEmbeddedExtras(remote)
	mergeEmbeddedExtras(remote)

	if got := modelIDs(remote.GeminiCLI); len(got) != 1 {
		t.Fatalf("merge accumulated duplicates: %v", got)
	}
}

func TestMergeEmbeddedExtrasHandlesNilInputs(t *testing.T) {
	withEmbeddedCatalog(t, nil)
	mergeEmbeddedExtras(&staticModelsJSON{}) // embedded catalog unavailable

	withEmbeddedCatalog(t, &staticModelsJSON{GeminiCLI: []*ModelInfo{{ID: "gemini-3.5-flash"}}})
	mergeEmbeddedExtras(nil) // must not panic
}
