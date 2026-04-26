using System.Net.Http;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace Capstone2PipelineViewer;

public partial class MainWindow : Window
{
    private static readonly HttpClient SharedHttp = new()
    {
        Timeout = TimeSpan.FromMinutes(45),
    };

    private readonly PipelineApiClient _api = new(SharedHttp);

    public MainWindow()
    {
        InitializeComponent();
        var saved = AppSettings.Load();
        BaseUrlBox.Text = string.IsNullOrWhiteSpace(saved.ApiBaseUrl)
            ? BaseUrlBox.Text
            : saved.ApiBaseUrl;
        LocalGgufPathBox.Text = saved.LocalGgufPath ?? LocalGgufPathBox.Text;

        SourceCombo.ItemsSource = new[]
        {
            new SourceItem("WHO Malaria (NIH Bookshelf)", TestQueries.WhoMalariaPreset),
            new SourceItem("Uganda Clinical Guidelines 2023", TestQueries.UgandaPreset),
        };
        SourceCombo.DisplayMemberPath = nameof(SourceItem.Label);
        SourceCombo.SelectedIndex = 0;
        RefreshQueryList();
    }

    private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
    {
        await RefreshLocalLlmStatusFromApiAsync().ConfigureAwait(true);
    }

    private async Task RefreshLocalLlmStatusFromApiAsync()
    {
        if (!TryGetBaseUri(out var baseUri))
        {
            LocalLlmStatusText.Text = "Set API URL, then Health, to see local LLM status.";
            return;
        }

        try
        {
            var raw = await _api.HealthAsync(baseUri).ConfigureAwait(true);
            ApplyLocalLlmStatusFromHealthJson(raw);
        }
        catch
        {
            LocalLlmStatusText.Text = "Local LLM: API not reachable.";
        }
    }

    private void ApplyLocalLlmStatusFromHealthJson(string healthJson)
    {
        try
        {
            using var doc = JsonDocument.Parse(healthJson);
            var root = doc.RootElement;
            var gguf = root.TryGetProperty("local_llm_gguf_configured", out var g) && g.GetBoolean();
            var env = root.TryGetProperty("local_llm_env_enabled", out var en) && en.GetBoolean();
            var envUnlock = root.TryGetProperty("local_llm_client_path_env_unlocked", out var u) && u.GetBoolean();
            var pipeline = root.TryGetProperty("pipeline", out var p) ? p.GetString() : null;
            var pipeNote = string.IsNullOrEmpty(pipeline) ? "" : $" pipeline={pipeline}";
            LocalLlmStatusText.Text =
                $"API{pipeNote}: env GGUF file={gguf}, SAFEAI_USE_LOCAL_LLM={env}, " +
                $"client path via POST env-unlock={envUnlock}. " +
                (gguf && env
                    ? "Asks can use local LLM from env."
                    : "With API on 127.0.0.1, non-empty GGUF path below is sent as local_llm_gguf (no env required).");
        }
        catch
        {
            LocalLlmStatusText.Text = "Local LLM: could not parse /health JSON.";
        }
    }

    private void SourceCombo_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (IsLoaded)
            RefreshQueryList();
    }

    private void RefreshQueryList()
    {
        if (SourceCombo.SelectedItem is not SourceItem src)
            return;

        var queries = TestQueries.ForPreset(src.Preset);
        var options = new List<QueryOption>(queries.Count);
        for (var i = 0; i < queries.Count; i++)
        {
            var q = queries[i];
            var shortText = q.Length > 95 ? q[..92] + "…" : q;
            options.Add(new QueryOption(i + 1, q, $"{i + 1}. {shortText}"));
        }

        QueryCombo.ItemsSource = options;
        QueryCombo.SelectedIndex = 0;
    }

    private void SetBusy(bool busy)
    {
        InitButton.IsEnabled = !busy;
        AskButton.IsEnabled = !busy;
        HealthButton.IsEnabled = !busy;
        SourceCombo.IsEnabled = !busy;
        QueryCombo.IsEnabled = !busy;
        BaseUrlBox.IsEnabled = !busy;
        ReuseKbCheck.IsEnabled = !busy;
        UseLocalQwenCheck.IsEnabled = !busy;
        LocalGgufPathBox.IsEnabled = !busy;
        BrowseGgufButton.IsEnabled = !busy;
        Cursor = busy ? System.Windows.Input.Cursors.Wait : System.Windows.Input.Cursors.Arrow;
    }

    private bool TryGetBaseUri(out Uri uri)
    {
        var text = BaseUrlBox.Text.Trim();
        if (!Uri.TryCreate(text.EndsWith('/') ? text : text + "/", UriKind.Absolute, out var built))
        {
            uri = null!;
            MessageBox.Show("Enter a valid absolute API URL (e.g. http://127.0.0.1:8001).", "Invalid URL",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }

        uri = built;
        return true;
    }

    private async void HealthButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (!TryGetBaseUri(out var baseUri))
            return;

        SetBusy(true);
        StatusText.Text = "GET /health …";
        try
        {
            var raw = await _api.HealthAsync(baseUri).ConfigureAwait(true);
            ResultBox.Text = PipelineApiClient.TryFormatJson(raw);
            ApplyLocalLlmStatusFromHealthJson(raw);
            StatusText.Text = "Health OK.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Health failed.";
            ResultBox.Text = ex.ToString();
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void InitButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (!TryGetBaseUri(out var baseUri))
            return;
        if (SourceCombo.SelectedItem is not SourceItem src)
            return;

        SetBusy(true);
        StatusText.Text = "POST /initialize — this can take several minutes on first run …";
        ResultBox.Text = "";
        try
        {
            var reuse = ReuseKbCheck.IsChecked == true;
            var (code, body) = await _api.InitializeAsync(baseUri, src.Preset, reuse).ConfigureAwait(true);
            ResultBox.Text = PipelineApiClient.TryFormatJson(body);
            StatusText.Text = code == 200
                ? "Index ready. You can run a query."
                : $"Initialize returned HTTP {code}.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Initialize failed.";
            ResultBox.Text = ex.ToString();
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void MainWindow_OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        AppSettings.Save(
            new AppSettings.Data
            {
                ApiBaseUrl = BaseUrlBox.Text.Trim(),
                LocalGgufPath = LocalGgufPathBox.Text.Trim(),
            });
    }

    private void BrowseGguf_OnClick(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Filter = "GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*",
            Title = "Select GGUF model (path used on API host)",
        };
        if (dlg.ShowDialog(this) == true)
            LocalGgufPathBox.Text = dlg.FileName;
    }

    private async void AskButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (!TryGetBaseUri(out var baseUri))
            return;
        if (QueryCombo.SelectedItem is not QueryOption qo)
        {
            MessageBox.Show("Select a test query.", "Query", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        SetBusy(true);
        StatusText.Text = "POST /ask …";
        try
        {
            var useLocal = UseLocalQwenCheck.IsChecked == true;
            var gguf = LocalGgufPathBox.Text.Trim();
            var (code, body) = await _api.AskAsync(
                baseUri,
                qo.Query,
                fullResponse: true,
                useLocalLlm: useLocal,
                localLlmGguf: string.IsNullOrEmpty(gguf) ? null : gguf).ConfigureAwait(true);
            ResultBox.Text = PipelineApiClient.TryFormatJson(body);
            StatusText.Text = code == 200 ? "Answer received." : $"Ask returned HTTP {code}.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Ask failed.";
            ResultBox.Text = ex.ToString();
        }
        finally
        {
            SetBusy(false);
        }
    }

    private sealed record SourceItem(string Label, string Preset);

    private sealed record QueryOption(int Index, string Query, string Caption);
}
