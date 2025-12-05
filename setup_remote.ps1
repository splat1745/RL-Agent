$User = "underdog"
$IP = "192.168.0.173"

Write-Host "Connecting to $IP as $User..."
Write-Host "NOTE: You will be asked for your password multiple times."

# 1. Create Directories (ensure rl_train exists)
Write-Host "Creating remote directories..."
ssh ${User}@${IP} "mkdir -p ~/Auto-Farmer/data/rl_train; mkdir -p ~/Auto-Farmer/rl"

# 2. Copy Scripts
Write-Host "Copying scripts..."
scp continuous_trainer.py ${User}@${IP}:~/Auto-Farmer/
scp train_imitation.py ${User}@${IP}:~/Auto-Farmer/

# 3. Copy RL Module
Write-Host "Copying RL module..."
scp -r rl ${User}@${IP}:~/Auto-Farmer/

Write-Host "---------------------------------------------------"
Write-Host "Setup Complete!"
Write-Host "Now, SSH into the box and install dependencies:"
Write-Host "ssh ${User}@${IP}"
Write-Host "pip install torch torchvision numpy opencv-python"
Write-Host "cd ~/Auto-Farmer"
Write-Host "python continuous_trainer.py"
Write-Host "---------------------------------------------------"
