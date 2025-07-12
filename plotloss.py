import matplotlib.pyplot as plt
import re

# Load log file
with open('pretrain_log.txt', 'r') as f:
    log_data = f.read()

# Extract epoch numbers and loss values
epochs = []
losses = []

for match in re.finditer(r"Epoch (\d+).*?Recon\+KL Loss:\s+([\d.]+)", log_data):
    epoch = int(match.group(1))
    loss = float(match.group(2))
    if epoch > 0:  # Only consider epochs after 10
        epochs.append(epoch)
        losses.append(loss)

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linewidth=2, color='seagreen')
plt.title("GAE Pretraining: Recon + KL Loss vs. Epoch (After Epoch 10)")
plt.xlabel("Epoch")
plt.ylabel("Recon + KL Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("pretrain_loss_after_epoch10.png", dpi=300)
plt.show()
