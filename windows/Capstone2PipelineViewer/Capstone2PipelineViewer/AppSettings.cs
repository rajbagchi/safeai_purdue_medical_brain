using System.IO;
using System.Text.Json;

namespace Capstone2PipelineViewer;

/// <summary>Persisted UI defaults (%LocalAppData%\SafeAICapstone2PipelineViewer\settings.json).</summary>
internal static class AppSettings
{
    private static readonly JsonSerializerOptions JsonWrite = new() { WriteIndented = true };

    private static string SettingsPath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "SafeAICapstone2PipelineViewer",
            "settings.json");

    public sealed class Data
    {
        public string ApiBaseUrl { get; set; } = "http://127.0.0.1:8001";

        public string LocalGgufPath { get; set; } =
            @"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf";
    }

    public static Data Load()
    {
        try
        {
            if (!File.Exists(SettingsPath))
                return new Data();
            var json = File.ReadAllText(SettingsPath);
            return JsonSerializer.Deserialize<Data>(json) ?? new Data();
        }
        catch
        {
            return new Data();
        }
    }

    public static void Save(Data data)
    {
        try
        {
            var dir = Path.GetDirectoryName(SettingsPath);
            if (dir is not null)
                Directory.CreateDirectory(dir);
            File.WriteAllText(SettingsPath, JsonSerializer.Serialize(data, JsonWrite));
        }
        catch
        {
            // ignore persistence errors
        }
    }
}
