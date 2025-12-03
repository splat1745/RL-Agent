$User = "underdog"
$IP = "192.168.1.198"
$RemotePath = "~/Auto-Farmer"
$LocalData = "data/imitation"

# Ensure local directory exists
if (-not (Test-Path $LocalData)) {
    New-Item -ItemType Directory -Force -Path $LocalData
}

# Keep track of sent files to avoid re-uploading
$SentFiles = @{}

Write-Host "Starting Sync Loop with $IP..."
Write-Host "Password: AboudBeast@2011$"

while ($true) {
    # 1. Find new .pkl files
    $Files = Get-ChildItem $LocalData -Filter "*.pkl"
    $NewFiles = $Files | Where-Object { -not $SentFiles.ContainsKey($_.Name) }

    if ($NewFiles) {
        Write-Host "Found $($NewFiles.Count) new files. Uploading..."
        foreach ($File in $NewFiles) {
            Write-Host "Uploading $($File.Name)..."
            # Suppress output for cleaner log
            scp $File.FullName ${User}@${IP}:${RemotePath}/data/imitation/
            
            if ($?) { 
                $SentFiles[$File.Name] = $true 
            } else {
                Write-Host "Failed to upload $($File.Name)" -ForegroundColor Red
            }
        }
    }

    # 2. Pull updated model
    # We only want to pull if the remote file is newer, but scp doesn't check timestamps easily.
    # We'll just pull it. It's one file.
    Write-Host "Checking for model update..."
    scp -q ${User}@${IP}:${RemotePath}/ppo_agent_imitation.pth .
    
    if ($?) {
        Write-Host "Model synced." -ForegroundColor Green
    }

    Write-Host "Sleeping for 30 seconds..."
    Start-Sleep -Seconds 30
}