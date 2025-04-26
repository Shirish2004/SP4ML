import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['DenseNet121', 'DenseNet161', 'ResNet18', 'ResNet34', 'ResNet50']

# Integer order accuracies
integer_acc = [0.87, 0.83, 0.82, 0.78, 0.85]

# Fractional order accuracies
fractional_acc = [0.79, 0.78, 0.78, 0.81, 0.74]

# Plot
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width/2, integer_acc, width, label='Integer Order', alpha=0.8)
rects2 = ax.bar(x + width/2, fractional_acc, width, label='Fractional Order (α=0.9)', alpha=0.8)

# Labels and titles
ax.set_ylabel('Accuracy')
ax.set_title('Integer Order vs Fractional Order Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Annotate
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
