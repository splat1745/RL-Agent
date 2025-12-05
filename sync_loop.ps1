$LoginInfo = ".\LinUser.txt"

if (Test-Path $LoginInfo) {
    $lines = @(Get-Content $LoginInfo)
    if ($lines.Count -ge 2) {
        $User = $lines[0].Trim()
        $IP = $lines[1].Trim()
        $Password = if ($lines.Count -ge 3) { $lines[2].Trim() } else { "Unknown" }
    } else {
        Write-Host "Error: LinUser.txt must have at least 2 lines (User, IP)." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Error: LinUser.txt not found. Please create it with User, IP, and Password on separate lines." -ForegroundColor Red
    exit 1
}

$RemotePath = "~/Auto-Farmer"
$LocalData = "D:\Auto-Farmer-Data\imitation_train"

# Ensure local directory exists
if (-not (Test-Path $LocalData)) {
    New-Item -ItemType Directory -Force -Path $LocalData
}

# Keep track of sent files to avoid re-uploading
$SentFiles = @{}
$LogFile = "uploaded_files.log"

if (Test-Path $LogFile) {
    Get-Content $LogFile | ForEach-Object { $SentFiles[$_] = $true }
    Write-Host "Loaded $($SentFiles.Count) uploaded files from history."
} else {
    # Try to fetch existing files from remote to avoid initial re-upload
    Write-Host "No history found. Checking remote files..."
    try {
        $RemoteFiles = ssh ${User}@${IP} "ls ${RemotePath}/data/imitation/"
        if ($?) {
            foreach ($file in $RemoteFiles) {
                if ($file -like "*.pkl") {
                    $SentFiles[$file] = $true
                    Add-Content -Path $LogFile -Value $file
                }
            }
            Write-Host "Discovered $($SentFiles.Count) files already on remote."
        }
    } catch {
        Write-Host "Could not check remote files. Will upload all local files."
    }
}

Write-Host "Starting Sync Loop with $IP..."
Write-Host "Password: $Password"

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
                Add-Content -Path $LogFile -Value $File.Name
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