package executor

import "testing"

// The peek that decides whether a turn is worth forwarding must agree with what
// the client can actually render.
func TestGeminiCLIChunkHasRenderableContent(t *testing.T) {
	cases := []struct {
		name          string
		line          string
		countThoughts bool
		want          bool
	}{
		{
			name:          "visible text",
			line:          `data: {"response":{"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}}`,
			countThoughts: true,
			want:          true,
		},
		{
			name: "function call",
			line: `data: {"response":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"Bash","args":{}}}]}}]}}`,
			want: true,
		},
		{
			name: "empty text part",
			line: `data: {"response":{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":0}}}`,
			want: false,
		},
		{
			name: "bare thought signature",
			line: `data: {"response":{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig"}]}}]}}`,
			want: false,
		},
		{
			name: "no parts at all",
			line: `data: {"response":{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10}}}`,
			want: false,
		},
		{
			name:          "thought part when the client asked for thinking",
			line:          `data: {"response":{"candidates":[{"content":{"parts":[{"text":"reasoning","thought":true}]}}]}}`,
			countThoughts: true,
			want:          true,
		},
		{
			name: "thought part when the client did not ask for thinking",
			line: `data: {"response":{"candidates":[{"content":{"parts":[{"text":"reasoning","thought":true}]}}]}}`,
			want: false,
		},
		{
			name: "usage only",
			line: `data: {"response":{"usageMetadata":{"promptTokenCount":10}}}`,
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := geminiCLIChunkHasRenderableContent([]byte(tc.line), tc.countThoughts); got != tc.want {
				t.Fatalf("geminiCLIChunkHasRenderableContent() = %v, want %v", got, tc.want)
			}
		})
	}
}
