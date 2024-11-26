import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the default font to Times New Roman
plt.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 12
})

def plot_class_distribution():
    # Updated ISCX-VPN distribution and labels
    iscx_vpn_distribution = [
        14621, 21610, 3752, 138549, 399893, 4996, 
        596, 8058, 1318, 2040, 7730, 954
    ]
    iscx_vpn_labels = [
        'email', 'chat', 'stream', 'ft', 'voip', 'p2p',
        'vpn_email', 'vpn_chat', 'vpn_stream', 'vpn_ft', 'vpn_voip', 'vpn_p2p'
    ]

    # Updated VNAT distribution and labels
    vnat_distribution = [32826, 27182, 3518, 3052, 712, 18, 16, 10]
    vnat_labels = [
        'ft', 'p2p', 'stream', 'voip', 'vpn_voip', 
        'vpn_ft', 'vpn_p2p', 'vpn_stream'
    ]

    # Updated Tor distribution and labels
    iscx_tor_distribution = [55660, 700, 1008, 2016, 3774, 3104, 2902, 45838, 68, 12, 42, 46, 294, 12, 28, 40]    
    iscx_tor_labels = [
        'browse', 'email','chat','audio','video','ft','voip','p2p','tor_browse','tor_email',
        'tor_chat','tor_audio','tor_video','tor_ft','tor_voip','tor_p2p'
    ]


    # Create a DataFrame for each dataset
    iscx_vpn_df = pd.DataFrame({'Labels': iscx_vpn_labels, 'Count': iscx_vpn_distribution})
    vnat_df = pd.DataFrame({'Labels': vnat_labels, 'Count': vnat_distribution})
    iscx_tor_df = pd.DataFrame({'Labels': iscx_tor_labels, 'Count': iscx_tor_distribution})

    # Create a figure with 3 subplots in one column
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    # Function to add value labels on bars
    def add_value_labels(ax, fontsize=10):
        for container in ax.containers:
            ax.bar_label(container, fontsize=fontsize, padding=3, rotation=0)

    # Plot bar chart for ISCX2016 using seaborn
    sns.barplot(x='Labels', y='Count', data=iscx_vpn_df, ax=axes[0], 
                hue='Labels', legend=False, palette='tab20')
    add_value_labels(axes[0])
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylim(0, max(iscx_vpn_distribution) * 1.15)
    axes[0].set_title('ISCX-VPN Dataset Distribution', pad=10)
    axes[0].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[0].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for VNAT-PN using seaborn
    sns.barplot(x='Labels', y='Count', data=vnat_df, ax=axes[1], 
                hue='Labels', legend=False, palette='tab10')
    add_value_labels(axes[1])
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_ylim(0, max(vnat_distribution) * 1.15)
    axes[1].set_title('VNAT Dataset Distribution', pad=10)
    axes[1].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[1].set_ylabel('Number of Samples')  # Set consistent y-label
    
    # Plot bar chart for Tor using seaborn
    sns.barplot(x='Labels', y='Count', data=iscx_tor_df, ax=axes[2], 
                hue='Labels', legend=False, palette='tab10')
    add_value_labels(axes[2])
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].set_ylim(0, max(iscx_tor_distribution) * 1.15)
    axes[2].set_title('ISCX-Tor Dataset Distribution', pad=10)
    axes[2].set_xlabel('Traffic Classes')  # Set consistent x-label
    axes[2].set_ylabel('Number of Samples')  # Set consistent y-label
    
    plt.tight_layout(pad=2.0)
    plt.savefig('visualization/fig_dataset_distribution_barplot.png')
    plt.show()

# Call the function to plot the distributions
plot_class_distribution()