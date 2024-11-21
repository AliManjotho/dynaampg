import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the default font to Times New Roman
plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 12
})

def plot_class_distribution():
    # Updated ISCX2016 distribution and labels
    iscx2016_distribution = [
        14621, 21610, 3752, 138549, 399893, 4996, 
        596, 8058, 1318, 2040, 7730, 954
    ]
    iscx2016_labels = [
        'email', 'chat', 'stream', 'ft', 'voip', 'p2p',
        'vpn_email', 'vpn_chat', 'vpn_stream', 'vpn_ft', 'vpn_voip', 'vpn_p2p'
    ]

    # Updated VNAT-PN distribution and labels
    vnat_vpn_distribution = [32826, 27182, 3518, 3052, 712, 18, 16, 10]
    vnat_vpn_labels = [
        'ft', 'p2p', 'stream', 'voip', 'vpn_voip', 
        'vpn_ft', 'vpn_p2p', 'vpn_stream'
    ]

    # Updated Tor distribution and labels
    tor_distribution = [2645, 497, 485, 1026, 1529, 1663, 4524, 2139]
    tor_labels = [
        'browse', 'email', 'chat', 'audio', 
        'video', 'ft', 'voip', 'p2p' 
    ]

    # Create a DataFrame for each dataset
    iscx2016_df = pd.DataFrame({'Labels': iscx2016_labels, 'Count': iscx2016_distribution})
    vnat_vpn_df = pd.DataFrame({'Labels': vnat_vpn_labels, 'Count': vnat_vpn_distribution})
    tor_df = pd.DataFrame({'Labels': tor_labels, 'Count': tor_distribution})

    # Create a figure with 3 subplots in one column
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Function to add value labels on bars
    def add_value_labels(ax, fontsize=10):
        for container in ax.containers:
            ax.bar_label(container, fontsize=fontsize, padding=3, rotation=0)

    # Plot bar chart for ISCX2016 using seaborn
    sns.barplot(x='Labels', y='Count', data=iscx2016_df, ax=axes[0], 
                hue='Labels', legend=False, palette='tab20')
    add_value_labels(axes[0])
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylim(0, max(iscx2016_distribution) * 1.15)
    axes[0].set_title('ISCX-VPN Dataset Distribution', pad=10)
    axes[0].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[0].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for VNAT-PN using seaborn
    sns.barplot(x='Labels', y='Count', data=vnat_vpn_df, ax=axes[1], 
                hue='Labels', legend=False, palette='tab10')
    add_value_labels(axes[1])
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(0, max(vnat_vpn_distribution) * 1.15)
    axes[1].set_title('VNAT Dataset Distribution', pad=10)
    axes[1].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[1].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for Tor using seaborn
    sns.barplot(x='Labels', y='Count', data=tor_df, ax=axes[2], 
                hue='Labels', legend=False, palette='tab10')
    add_value_labels(axes[2])
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].set_ylim(0, max(tor_distribution) * 1.15)
    axes[2].set_title('ISCX-Tor Dataset Distribution', pad=10)
    axes[2].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[2].set_ylabel('Number of Samples')  # Set consistent y-label
    
    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_dataset_distribution_barplot.png')
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()