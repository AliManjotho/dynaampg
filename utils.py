from pathlib import Path
from enum import Enum
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np

class DATASET(Enum):
    ISCX_VPN=0
    VNAT=1
    ISCX_TOR=2
    NETWORK_TRAFFIC=3
    REALTIME=4

iscx_vpn_map = {
            'email': ['email'],
            'chat': ['aim_chat', 'AIMchat', 'facebook_chat', 'facebookchat', 'hangout_chat', 'hangouts_chat', 'icq_chat', 'ICQchat', 'gmailchat', 'gmail_chat', 'skype_chat'],
            'streaming': ['netflix', 'spotify', 'vimeo', 'youtube', 'youtubeHTML5'],
            'file_transfer': ['ftps_down', 'ftps_up','sftp_up', 'sftpUp', 'sftp_down', 'sftpDown', 'sftp', 'skype_file', 'scpUp', 'scpDown', 'scp'],
            'voip': ['voipbuster', 'facebook_audio', 'hangout_audio', 'hangouts_audio', 'skype_audio'],
            'p2p': ['skype_video', 'facebook_video', 'hangout_video', 'hangouts_video'],
            'vpn_email': ['vpn_email'],
            'vpn_chat': ['vpn_aim_chat', 'vpn_facebook_chat', 'vpn_hangouts_chat', 'vpn_icq_chat', 'vpn_skype_chat'],
            'vpn_streaming': ['vpn_netflix', 'vpn_spotify', 'vpn_vimeo', 'vpn_youtube'],
            'vpn_file_transfer': ['vpn_ftps', 'vpn_sftp', 'vpn_skype_files'],
            'vpn_voip': ['vpn_facebook_audio', 'vpn_skype_audio', 'vpn_voipbuster'],
            'vpn_p2p': ['vpn_bittorrent']   
}

vnat_map = {
            'streaming': ['nonvpn_netflix', 'nonvpn_youtube', 'nonvpn_vimeo'],
            'voip': ['nonvpn_voip', 'nonvpn_skype'],
            'file_transfer': ['nonvpn_rsync', 'nonvpn_sftp', 'nonvpn_scp'],
            'p2p': ['nonvpn_ssh', 'nonvpn_rdp'],
            'vpn_streaming': ['vpn_netflix', 'vpn_youtube', 'vpn_vimeo'],
            'vpn_voip': ['vpn_voip', 'vpn_skype'],
            'vpn_file_transfer': ['vpn_rsync', 'vpn_sftp', 'vpn_scp'],
            'vpn_p2p': ['vpn_ssh', 'vpn_rdp']
}



iscx_tor_map = {
            'browsing': ['NONTOR_browsing', 'NONTOR_SSL_Browsing'],
            'email': ['NONTOR_Email', 'NONTOR_POP', 'NONTOR_Workstation_Thunderbird'],
            'chat': ['NONTOR_aimchat', 'NONTOR_AIM_Chat', 'NONTOR_facebookchat', 'NONTOR_facebook_chat', 'NONTOR_hangoutschat', 'NONTOR_hangout_chat', 'NONTOR_icq', 'NONTOR_ICQ', 'NONTOR_skypechat', 'NONTOR_skype_chat', 'NONTOR_skype_transfer'],
            'audio_stream': ['NONTOR_spotify'],
            'video_stream': ['NONTOR_Vimeo','NONTOR_Youtube'],        
            'file_transfer': ['NONTOR_FTP'],
            'voip': ['NONTOR_facebook_Audio', 'NONTOR_Facebook_Voice_Workstation', 'NONTOR_Hangouts_voice_Workstation', 'NONTOR_Hangout_Audio', 'NONTOR_Skype_Audio', 'NONTOR_Skype_Voice_Workstation'],
            'p2p': ['NONTOR_p2p', 'NONTOR_SFTP', 'NONTOR_ssl'],
            'tor_browsing': ['BROWSING', 'torGoogle', 'torTwitter'],
            'tor_email': ['MAIL'],
            'tor_chat': ['CHAT', 'torFacebook'],
            'tor_audio_stream': ['AUDIO', 'tor_spotify'],
            'tor_video_stream': ['VIDEO', 'torVimeo', 'torYoutube'],         
            'tor_file_transfer': ['FILE-TRANSFER'],
            'tor_voip': ['VOIP'],
            'tor_p2p': ['P2P', 'tor_p2p']
}

network_traffic_map = {
            'browsing': ['browsing'],
            'file_transfer': ['file_transfer'],
            'p2p': ['p2p'],
            'streaming': ['video']
}

realtime_map = {
            'streaming': ['audio', 'video'],
            'chat': ['chat'],
            'voip': ['voip'],
            'game': ['game']
}

def iscx_vpn_get_unique_labels(): 
    return list(iscx_vpn_map.keys())

def vnat_get_unique_labels(): 
    return list(vnat_map.keys())

def iscx_tor_get_unique_labels(): 
    return list(iscx_tor_map.keys())

def network_traffic_get_unique_labels(): 
    return list(network_traffic_map.keys())

def realtime_get_unique_labels(): 
    return list(realtime_map.keys())

def iscx_vpn_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in iscx_vpn_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls


def vnat_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in vnat_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls


def iscx_tor_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in iscx_tor_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls

def network_traffic_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in network_traffic_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls

def realtime_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in realtime_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls

def iscx_vpn_get_one_hot(cls):
    clss = iscx_vpn_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot


def vnat_get_one_hot(cls):
    clss = vnat_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot


def iscx_tor_get_one_hot(cls):
    clss = iscx_tor_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot

def network_traffic_get_one_hot(cls):
    clss = network_traffic_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot

def realtime_get_one_hot(cls):
    clss = realtime_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot

def filenumber_to_id(file_num, length=8):
    file_num_str = str(file_num)
    file_num_str_len = len(file_num_str)
    return '0' * (length - file_num_str_len) + file_num_str

def num_packets_to_edge_indices(num_packets):
    return [list(range(0,num_packets-1)), list(range(1,num_packets))]
    



        








def count_classes(dataset_path, dataset):

    class_names = {}
    if dataset == DATASET.ISCX_VPN:
        class_names = {c: 0 for c in list(iscx_vpn_map.keys())}
    elif dataset == DATASET.VNAT:
        class_names = {c: 0 for c in list(vnat_map.keys())} 
    elif dataset == DATASET.ISCX_TOR:
        class_names = {c: 0 for c in list(iscx_tor_map.keys())}
    elif dataset == DATASET.NETWORK_TRAFFIC:
        class_names = {c: 0 for c in list(network_traffic_map.keys())}
    elif dataset == DATASET.REALTIME:
        class_names = {c: 0 for c in list(realtime_map.keys())}

    # Get list of all PCAP session file paths
    files = list(Path(dataset_path).rglob('*.pcap'))

    for file in enumerate(files, start=1):
        class_label = ''
        if dataset == DATASET.ISCX_VPN:
            class_label = iscx_vpn_get_class_label(Path(file.__str__()).name)
        elif dataset == DATASET.VNAT:
            class_label = vnat_get_class_label(Path(file.__str__()).name)
        elif dataset == DATASET.ISCX_TOR:
            class_label = iscx_tor_get_class_label(Path(file.__str__()).name)
        elif dataset == DATASET.NETWORK_TRAFFIC:
            class_label = network_traffic_get_class_label(Path(file.__str__()).name)
        elif dataset == DATASET.REALTIME:
            class_label = realtime_get_class_label(Path(file.__str__()).name)

        class_names[class_label] = class_names[class_label]  + 1
    
    for key, value in class_names.items():
        print(key, value)
    print('')



    

def reduce_dimentions(logits, method='PCA', n_components=2): 
    n_samples = len(logits)
    perplexity = min(30, n_samples // 4)
    reduced_features = []

    if method == 'PCA':
        pca = PCA(n_components)
        reduced_features = pca.fit_transform(logits)
    elif method == 't-SNE':
        tsne = TSNE(n_components, random_state=42, perplexity=perplexity)
        reduced_features = tsne.fit_transform(logits)

    return torch.tensor(reduced_features)





def get_onehot_by_index(class_index, n_classes):
    one_hot = [0 for _ in range(n_classes)]
    one_hot[class_index] = 1

    return torch.tensor(one_hot, dtype=torch.float32)

def get_onehot_by_label(label, class_labels):
    n_classes = len(class_labels)
    one_hot = [0 for _ in range(n_classes)]
    one_hot[class_labels.index(label)] = 1

    return torch.tensor(one_hot, dtype=torch.float32)

if __name__ == '__main__':
    
    
    count_classes(r'C:\Datasets\Realtime', DATASET.REALTIME)
