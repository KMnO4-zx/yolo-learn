import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the uploaded CSV file
file_path = './runs/obb/train/results.csv'
data = pd.read_csv(file_path)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# List of metrics to plot
metrics = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
    'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 
    'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2'
]

# Define a 3x4 grid for the subplots without markers
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# Flatten the axes array for easy iteration
axs = axs.flatten()

# Plot each metric on its respective subplot without markers
for i, metric in enumerate(metrics):
    if i < len(axs):
        axs[i].plot(data['epoch'], data[metric], linestyle='-')
        axs[i].set_title(metric)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric)
        axs[i].grid(True)

# Add a main title for the entire figure
fig.suptitle('YOLOv8_metric', fontsize=20)

# plt.show()
plt.savefig('metric_result.png')