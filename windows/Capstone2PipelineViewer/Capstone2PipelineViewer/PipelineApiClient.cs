using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Capstone2PipelineViewer;

public sealed class PipelineApiClient
{
    private static readonly JsonSerializerOptions JsonWrite = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private readonly HttpClient _http;

    public PipelineApiClient(HttpClient http)
    {
        _http = http;
    }

    public async Task<string> HealthAsync(Uri baseUri, CancellationToken ct = default)
    {
        var r = await _http.GetAsync(new Uri(baseUri, "health"), ct).ConfigureAwait(false);
        return await r.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
    }

    public async Task<(int StatusCode, string Body)> InitializeAsync(
        Uri baseUri,
        string preset,
        bool reuseExistingKb,
        CancellationToken ct = default)
    {
        var payload = new InitializeDto { Preset = preset, ReuseExistingKb = reuseExistingKb };
        var r = await _http.PostAsJsonAsync(new Uri(baseUri, "initialize"), payload, JsonWrite, ct)
            .ConfigureAwait(false);
        var body = await r.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        return ((int)r.StatusCode, body);
    }

    public async Task<(int StatusCode, string Body)> AskAsync(
        Uri baseUri,
        string query,
        bool fullResponse,
        bool? useLocalLlm = null,
        string? localLlmGguf = null,
        CancellationToken ct = default)
    {
        var payload = new AskDto
        {
            Query = query,
            FullResponse = fullResponse,
            UseLocalLlm = useLocalLlm,
            LocalLlmGguf = localLlmGguf,
        };
        var r = await _http.PostAsJsonAsync(new Uri(baseUri, "ask"), payload, JsonWrite, ct)
            .ConfigureAwait(false);
        var body = await r.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
        return ((int)r.StatusCode, body);
    }

    public static string TryFormatJson(string raw)
    {
        try
        {
            using var doc = JsonDocument.Parse(raw);
            return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions { WriteIndented = true });
        }
        catch
        {
            return raw;
        }
    }

    private sealed class InitializeDto
    {
        [JsonPropertyName("preset")]
        public string Preset { get; set; } = "";

        [JsonPropertyName("reuse_existing_kb")]
        public bool ReuseExistingKb { get; set; }
    }

    private sealed class AskDto
    {
        [JsonPropertyName("query")]
        public string Query { get; set; } = "";

        [JsonPropertyName("full_response")]
        public bool FullResponse { get; set; } = true;

        [JsonPropertyName("use_local_llm")]
        public bool? UseLocalLlm { get; set; }

        [JsonPropertyName("local_llm_gguf")]
        public string? LocalLlmGguf { get; set; }
    }
}
