# Copies V2 training files to your Ubuntu box using credentials in LinUser.txt
# Requires LinUser.txt to contain: user@host (e.g., underdog@192.168.0.8)

$linLines = Get-Content "$PSScriptRoot\LinUser.txt"
if ($linLines.Count -lt 2) {
    Write-Error "LinUser.txt should have at least two lines: username and host."
    exit 1
}

$user = $linLines[0].Trim()
$remoteHost = $linLines[1].Trim()
$linUser = "$user@$remoteHost"

# Paths
$srcRoot = "$PSScriptRoot"
$files = @(
    "$srcRoot\rl\network_v2.py",
    "$srcRoot\rl\agent_v2.py",
    "$srcRoot\rl\memory.py",
    "$srcRoot\trainer_v2.py"
)

# Ensure remote directories exist
Write-Host "Ensuring remote directories exist on $linUser ..."
ssh $linUser "mkdir -p ~/Auto-Farmer/rl"

# Copy individual files with correct destinations
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        Write-Error "Missing file: $f"
        exit 1
    }

    $leaf = Split-Path $f -Leaf
    $parentLeaf = Split-Path (Split-Path $f -Parent) -Leaf

    if ($parentLeaf -eq "rl") {
        $dest = "${linUser}:~/Auto-Farmer/rl/"
    } else {
        $dest = "${linUser}:~/Auto-Farmer/"
    }

    Write-Host "Copying $leaf to $dest"
    scp $f $dest
}

# Copy entire rl folder (optional, slower)
# scp -r "$srcRoot\rl" "$linUser:~/Auto-Farmer/"

Write-Host "Done. Verify on Ubuntu: ssh $linUser 'ls ~/Auto-Farmer/rl'"
