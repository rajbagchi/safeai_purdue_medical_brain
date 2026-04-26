# Build and run the WPF client for the capstone2 API (default http://127.0.0.1:8001).
$ErrorActionPreference = "Stop"
$proj = Join-Path $PSScriptRoot "Capstone2PipelineViewer\Capstone2PipelineViewer\Capstone2PipelineViewer.csproj"
dotnet build $proj -c Release
dotnet run --project $proj -c Release --no-build
