$User = "underdog"
$IP = "192.168.0.173"
$Password = "AboudBeast@2011$"

$RemotePath = "~/Auto-Farmer"
# Syncing RL Trajectories
$LocalData = "D:\Auto-Farmer-Data\rl_train"

# Ensure local directory exists
if (-not (Test-Path $LocalData)) {
    New-Item -ItemType Directory -Force -Path $LocalData
}

# Keep track of sent files to avoid re-uploading
$SentFiles = @{}
$LogFile = "uploaded_traj.log"

if (Test-Path $LogFile) {
    Get-Content $LogFile | ForEach-Object { $SentFiles[$_] = $true }
    Write-Host "Loaded $($SentFiles.Count) uploaded files from history."
} else {
    # Try to fetch existing files from remote to avoid initial re-upload
    Write-Host "No history found. Checking remote files..."
    try {
        # Check files in rl_train/
        $RemoteFiles = ssh ${User}@${IP} "ls ${RemotePath}/data/rl_train/"
        if ($?) {
            foreach ($file in $RemoteFiles) {
                if ($file -like "traj_*.pkl") {
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
    $Files = Get-ChildItem $LocalData -Filter "traj_*.pkl"
    $NewFiles = $Files | Where-Object { -not $SentFiles.ContainsKey($_.Name) }

    if ($NewFiles) {
        Write-Host "Found $($NewFiles.Count) new files. Uploading..."
        foreach ($File in $NewFiles) {
            Write-Host "Uploading $($File.Name)..."
            # Suppress output for cleaner log
            # Use data/rl_train/
            scp $File.FullName ${User}@${IP}:${RemotePath}/data/rl_train/
            
            if ($?) { 
                $SentFiles[$File.Name] = $true 
                Add-Content -Path $LogFile -Value $File.Name
            } else {
                Write-Host "Failed to upload $($File.Name)" -ForegroundColor Red
            }
        }
    }

    # 2. Pull updated model
    # Pull ppo_agent.pth -> ppo_agent_new.pth
    Write-Host "Checking for model update..."
    # Suppress output (-q)
    scp -q ${User}@${IP}:${RemotePath}/ppo_agent.pth ./ppo_agent_new.pth
    
    if ($?) {
        Write-Host "Model upgrade downloaded! (ppo_agent_new.pth)" -ForegroundColor Green
    }

    Write-Host "Sleeping for 10 seconds..."
    Start-Sleep -Seconds 10
}