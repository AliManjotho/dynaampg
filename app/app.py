import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import shutil
import pyshark
import threading
from datetime import datetime



def capture_thread(interface='Wi-Fi'):
    # Create temp directory if it doesn't exist
    temp_pcaps_dir = os.path.join(TEMP_CAPTURE_DIR, 'pcaps')
    os.makedirs(temp_pcaps_dir, exist_ok=True)
    
    current_packets = []
    current_size = 0
    file_counter = 1

    # Start capturing packets
    capture = pyshark.LiveCapture(interface=interface)
    
    def save_packets():
        nonlocal current_packets, current_size, file_counter
        if current_packets:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'capture_{timestamp}_{file_counter}.pcap'
            filepath = os.path.join(temp_pcaps_dir, filename)
            
            # Save packets using pyshark's dump feature
            capture.dump_packets(filepath, current_packets)
            
            # Reset counters
            current_packets = []
            current_size = 0
            file_counter += 1

    try:
        for packet in capture.sniff_continuously():
            # Approximate packet size (actual size might vary)
            packet_size = len(str(packet))
            current_size += packet_size
            current_packets.append(packet)

            # If accumulated size exceeds 10KB (10240 bytes), save to new file
            if current_size >= 5:
                save_packets()
                
    except KeyboardInterrupt:
        save_packets()  # Save remaining packets
        capture.close()

def capture_live_traffic():
    # Start capture in a separate thread
    capture_worker = threading.Thread(target=capture_thread, args=('Wi-Fi',), daemon=True)
    capture_worker.start()
    return capture_worker

def clear_temp_dir():
    if os.path.exists(TEMP_CAPTURE_DIR):
        shutil.rmtree(TEMP_CAPTURE_DIR)

def split_sessions(dataset_path):
    temp_pcaps_dir = os.path.join(TEMP_CAPTURE_DIR, 'pcaps')
    temp_splitted_dir = os.path.join(TEMP_CAPTURE_DIR, 'splitted')

    cmd1 = SPLITCAP_PATH + " -r " +  temp_pcaps_dir + " -o " + temp_splitted_dir + " -recursive -s session"    
    cmd2 = "del /Q " + temp_pcaps_dir + "\\*.pcap"
    cmd3 = "move /Y " + temp_splitted_dir + "\\*.pcap " + temp_pcaps_dir

    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    
if __name__ == "__main__":
    capture_live_traffic()
