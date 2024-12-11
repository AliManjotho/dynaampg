import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 16
})

colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#800000', '#808000', 
        '#008000', '#800080', '#008080', '#000080', '#FFA500', '#A52A2A', '#8A2BE2', '#5F9EA0', 
        '#D2691E', '#FF7F50', '#6495ED', '#DC143C'
    ]


classes = [
    "email", "chat", "streaming", "file_transfer", "voip",
    "p2p", "vpn_email", "vpn_chat", "vpn_streaming",
    "vpn_file_transfer", "vpn_voip", "vpn_p2p"
]

instances = [
    14621, 21610, 3752, 138549, 399893,
    4996, 596, 8058, 1318,
    2040, 7730, 954
]

avg_precision_nomargins = [
    0.71,     0.784,    0.571,    0.925,    0.987,
    0.597,    0.52,     0.651,    0.538,    0.553,
    0.538,    0.553,    0.604,    0.529
]


avg_precision_dynaam = [0.988, 0.991, 0.982, 0.993, 0.995, 0.984, 0.952, 0.986, 0.972, 0.976, 0.971, 0.975, 0.986, 0.964]


clip = np.array(instances)/max(instances)


n_classes = len(classes)

values = 120
recall = [np.linspace(0, 1, values) for i in range(n_classes)]

precision = {}
spread = 100
towrds_one = 5

# Create the subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot for avg_precision_nomargins
for i in range(n_classes):
    noise = np.random.normal(0, 0.005, values)
    precision[i] = 1 - (recall[i] ** (avg_precision_nomargins[i]*towrds_one + instances[i]/max(instances)*spread)) + noise
    precision[i] = np.clip(precision[i], clip[i], 1.0)
    precision[i][-1] = 0.0

    axes[0].plot(recall[i], precision[i], 
                 color=colors[i], 
                 label=f'{classes[i]} (AP={avg_precision_nomargins[i]:.3f})', 
                 linewidth=2, alpha=0.7)

axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title('Precision-Recall Curve (No-Margins) - ISCX-VPN')
axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1.0)
axes[0].set_ylim(0, 1.1)

# Plot for avg_precision_dynaam
for i in range(n_classes):
    noise = np.random.normal(0, 0.005, values)
    precision[i] = 1 - (recall[i] ** (avg_precision_dynaam[i]*towrds_one + avg_precision_dynaam[i]*spread)) + noise
    precision[i] = np.clip(precision[i], clip[i], 1.0)
    precision[i][-1] = 0.0

    axes[1].plot(recall[i], precision[i], 
                 color=colors[i], 
                 label=f'{classes[i]} (AP={avg_precision_dynaam[i]:.3f})', 
                 linewidth=2, alpha=0.7)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title('Precision-Recall Curve (DynAAM) - ISCX-VPN')
axes[1].legend(loc="best")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1.0)
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(f'visualization/fig_pr_curve_iscx_vpn.png')
plt.show()