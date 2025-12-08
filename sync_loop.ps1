$User = "underdog"
$IP = "192.168.0.173"
$Password = "AboudBeast@2011$"

$RemotePath = "~/Auto-Farmer"
# Syncing Imitation Training Data
$LocalData = "D:\Auto-Farmer-Data\imitation_train"
$RemoteDataPath = "~/Auto-Farmer/data/imitation"

# Ensure remote directory exists
Write-Host "Ensuring remote directory exists: $RemoteDataPath"
ssh ${User}@${IP} "mkdir -p $RemoteDataPath"

# Ensure local directory exists
if (-not (Test-Path $LocalData)) {
    New-Item -ItemType Directory -Force -Path $LocalData
}

# Keep track of sent files to avoid re-uploading
$SentFiles = @{}
$LogFile = "uploaded_traj.log"

# ALWAYS check remote files to ensure sync state is correct
Write-Host "Checking remote files to sync state..."
try {
    $RemoteFiles = ssh ${User}@${IP} "ls $RemoteDataPath"
    if ($?) {
        foreach ($file in $RemoteFiles) {
            # Add both traj and data files to known list
            if ($file -like "*.pkl") {
                $SentFiles[$file] = $true
            }
        }
        Write-Host "Discovered $($SentFiles.Count) files already on remote."
    }
} catch {
    Write-Host "Could not check remote files. Relying on local log."
}

# Also load local log as backup
if (Test-Path $LogFile) {
    Get-Content $LogFile | ForEach-Object { $SentFiles[$_] = $true }
}

Write-Host "Starting Sync Loop with $IP..."
Write-Host "Password: $Password"

while ($true) {
    # 1. Find new .pkl files
    $Files = Get-ChildItem $LocalData -Filter "data_*.pkl"
    $NewFiles = $Files | Where-Object { -not $SentFiles.ContainsKey($_.Name) }

    if ($NewFiles) {
        Write-Host "Found $($NewFiles.Count) new files. Uploading..."
        foreach ($File in $NewFiles) {
            Write-Host "Uploading $($File.Name)..."
            # Suppress output for cleaner log
            scp $File.FullName ${User}@${IP}:$RemoteDataPath/
            
            if ($?) { 
                $SentFiles[$File.Name] = $true 
                Add-Content -Path $LogFile -Value $File.Name
            } else {
                Write-Host "Failed to upload $($File.Name)" -ForegroundColor Red
            }
        }
    }

    # 2. Pull updated model (V2)
    # Pull ppo_v2.pth -> ppo_v2.pth
    Write-Host "Checking for model update..."
    # Suppress output (-q)
    scp -q ${User}@${IP}:${RemotePath}/ppo_v2.pth ./ppo_v2.pth
    
    if ($?) {
        Write-Host "Model upgrade downloaded! (ppo_agent_new.pth)" -ForegroundColor Green
    }

    Write-Host "Sleeping for 10 seconds..."
    Start-Sleep -Seconds 10
}