import pyshark
import os
import threading
class LiveTrafficCapture:
    def __init__(self, interface, output_dir):
        self.interface = interface
        self.output_dir = output_dir
        self.capture = pyshark.LiveCapture(interface=self.interface)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_capture(self):
        capture_thread = threading.Thread(target=self._capture_traffic)
        capture_thread.start()

    def _capture_traffic(self):
        for packet in self.capture.sniff_continuously():
            self._save_packet(packet)

    def _save_packet(self, packet):
        session_id = packet.ip.src + '-' + packet.ip.dst
        file_path = os.path.join(self.output_dir, f"{session_id}.pcap")
        with open(file_path, 'ab') as f:
            f.write(bytes(packet))

# Usage
output_dir = 'temp'
interface = 'Wi-Fi'
live_capture = LiveTrafficCapture(interface=interface, output_dir=output_dir)
live_capture.start_capture()
