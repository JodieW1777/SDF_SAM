param(
    [string]$Checkpoint = "result/best_sdf_sam_slice_24_plane.pth",
    [string]$Device = "cuda",
    [int]$Port = 8765
)

$env:MEDSAM_INTERACTIVE_CHECKPOINT = (Resolve-Path -LiteralPath $Checkpoint).Path
$env:MEDSAM_INTERACTIVE_DEVICE = $Device
python -m uvicorn interactive_backend.app:app --host 127.0.0.1 --port $Port
