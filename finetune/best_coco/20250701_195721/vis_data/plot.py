import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurar o estilo do seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

# Ler o arquivo JSON
with open('finetune/best_coco/20250701_195721/vis_data/20250701_195721.json', 'r') as f:
    data = []
    for line in f:
        try:
            entry = json.loads(line.strip())
            data.append(entry)
        except json.JSONDecodeError:
            continue

# Filtrar dados até epoch 20 e extrair epochs e losses
epochs = []
losses = []

for entry in data:
    if 'epoch' in entry and 'loss' in entry:
        epoch = entry['epoch']
        if epoch <= 25:
            epochs.append(epoch)
            losses.append(entry['loss'])

# Criar o gráfico com seaborn
plt.figure(figsize=(12, 8))
ax = sns.lineplot(x=epochs, y=losses, marker='o', markersize=6, linewidth=2.5, color='steelblue', ci=None)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xlim(0, 21)


plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('loss_vs_epoch_20.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Número de pontos plotados: {len(epochs)}")
print(f"Epoch inicial: {epochs[0]}")
print(f"Epoch final: {epochs[-1]}")
print(f"Loss inicial: {losses[0]:.4f}")