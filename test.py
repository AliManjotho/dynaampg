import matplotlib.pyplot as plt
import numpy as np

# Data from the image
models = ["1D-CNN", "FS-Net", "AppNet", "MIMETIC", "PEAN", "PEAN-L", "Ours"]
inference_time = [15.68, 618.56, 261.42, 34.72, 255.65, 38.62, 42.3]
gpu_memory_training = [1139, 3121, 8957, 1501, 29876, 1211, 18678]
gpu_memory_testing = [1073, 1825, 3057, 1113, 2579, 1157, 2541]
training_time = [5, 14, 30, 6, 12, 69, 54]

font_size = 26
bar_width = 0.4
color1 = '#008FD5'
color1_dark = '#006A9E'
color2 = '#FC4F30'
color2_dark = '#AB3621'

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Figure and axes setup
fig, axs = plt.subplots(1, 3, figsize=(35, 6))






# (a) Inference time for different models
axs[0].bar(models, inference_time, bar_width, color=color1, edgecolor=color1_dark)
axs[0].set_ylabel("Inference time (us/session)", fontsize=font_size)
axs[0].set_title("(a) Inference time for different models", fontsize=font_size, x=0.5, y=-0.2)
axs[0].set_ylim(0, 700)
axs[0].set_xticklabels(models, fontsize=font_size)
axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=font_size)
for i, v in enumerate(inference_time):
    axs[0].text(i, v + 30, f"{v}", ha='center', fontsize=font_size)

# (b) GPU Memory usage for different models
x = np.arange(len(models))
width = 0.48
axs[1].bar(x - width/2, gpu_memory_training, bar_width, label='Training', color=color1, edgecolor=color1_dark)
axs[1].bar(x + width/2, gpu_memory_testing, bar_width, label='Testing', color=color2, edgecolor=color2_dark)
axs[1].set_ylabel("GPU Memory (MiB)", fontsize=font_size)
axs[1].set_title("(b) GPU memory usage for different models", fontsize=font_size, x=0.5, y=-0.2)
axs[1].set_ylim(0, 35000)
axs[1].set_xticks(x)
axs[1].set_xticklabels(models, fontsize=font_size)
axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=font_size)
axs[1].legend(fontsize=font_size)
for i in range(len(models)):
    axs[1].text(x[i] - width/2, gpu_memory_training[i] + 1500, f"{gpu_memory_training[i]}", ha='center', fontsize=font_size)
    axs[1].text(x[i] + width/2, gpu_memory_testing[i] + 1500, f"{gpu_memory_testing[i]}", ha='center', fontsize=font_size)

# (c) Training time for different models
training_datasets = models
axs[2].bar(training_datasets, training_time, bar_width, color=color1)
axs[2].set_ylabel("Training time (sec/100 batches)", fontsize=font_size)
axs[2].set_title("(c) Training time for different models", fontsize=font_size, x=0.5, y=-0.2)
axs[2].set_ylim(0, 80)
axs[2].set_xticklabels(models, fontsize=font_size)
axs[2].set_yticklabels(axs[2].get_yticklabels(), fontsize=font_size)
for i, v in enumerate(training_time):
    axs[2].text(i, v + 3, f"{v}", ha='center', fontsize=font_size)

# Adjust layout
plt.tight_layout()
plt.gcf().set_size_inches(35, 6)
plt.savefig('fig_plots.png', dpi=100)
plt.show()
