import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Config values (from voc.yaml / train.py)
lr_base       = 0.001
warmup_epochs = 3
num_epochs    = 50
start_factor  = 0.1
eta_min       = lr_base * 0.01   # 1e-5
T_max         = num_epochs - warmup_epochs  # 47

def lr_at_epoch(epoch):
    """LR seen at the *start* of `epoch` (0-indexed), matching how train.py reads it."""
    if epoch < warmup_epochs:
        # LinearLR: factor linearly from start_factor to 1.0 over warmup_epochs steps
        factor = start_factor + (1.0 - start_factor) * epoch / warmup_epochs
        return lr_base * factor
    else:
        # CosineAnnealingLR: t counts from 0 inside the cosine phase
        t = epoch - warmup_epochs
        return eta_min + 0.5 * (lr_base - eta_min) * (1 + math.cos(math.pi * t / T_max))

epochs = list(range(num_epochs))
lrs    = [lr_at_epoch(e) for e in epochs]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, lrs, color='steelblue', linewidth=2)

# Annotate the warmup boundary
ax.axvline(warmup_epochs, color='grey', linestyle='--', linewidth=1)
ax.text(warmup_epochs + 0.4, lr_base * 0.92, 'warmup ends\n(epoch 3)',
        fontsize=8, color='grey', va='top')

# Annotate key LR values
ax.annotate(f'LR = {lrs[0]:.4f}', xy=(0, lrs[0]),
            xytext=(4, lrs[0] * 1.6), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='black'), color='black')
ax.annotate(f'LR = {lr_base:.4f}', xy=(warmup_epochs, lr_base),
            xytext=(10, lr_base * 1.05), fontsize=8, color='black')
ax.annotate(f'LR = {eta_min:.5f}', xy=(num_epochs - 1, lrs[-1]),
            xytext=(35, eta_min * 12), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='black'), color='black')

ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('LR schedule: linear warmup (3 epochs) + cosine annealing (47 epochs)')
ax.set_xlim(0, num_epochs - 1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = 'lr_schedule.png'
fig.savefig(out, dpi=150)
print(f'Saved to {out}')
plt.show()
