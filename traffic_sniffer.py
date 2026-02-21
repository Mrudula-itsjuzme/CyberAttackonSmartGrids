import pyshark
import pandas as pd
import datetime
import time

# Set network interface (check using "ipconfig" in cmd)
INTERFACE = "Wi-Fi"  # Change this if needed

# Output CSV file
OUTPUT_FILE = "traffic_log.csv"

# Packet storage
packet_data = []

# Function to process packets
def packet_callback(pkt):
    try:
        timestamp = datetime.datetime.now()
        src_ip = pkt.ip.src if hasattr(pkt, 'ip') else "N/A"
        dst_ip = pkt.ip.dst if hasattr(pkt, 'ip') else "N/A"
        protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') else "N/A"
        length = pkt.length if hasattr(pkt, 'length') else "N/A"
        src_port = pkt[pkt.transport_layer].srcport if hasattr(pkt, 'transport_layer') else "N/A"
        dst_port = pkt[pkt.transport_layer].dstport if hasattr(pkt, 'transport_layer') else "N/A"

        # Append packet data
        packet_data.append([timestamp, src_ip, dst_ip, protocol, length, src_port, dst_port])

        # Save every 50 packets
        if len(packet_data) % 50 == 0:
            df = pd.DataFrame(packet_data, columns=["Timestamp", "Src_IP", "Dst_IP", "Protocol", "Length", "Src_Port", "Dst_Port"])
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"[+] Saved {len(packet_data)} packets to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")

# Run for exactly 2 minutes
print(f"[*] Capturing packets for 2 minutes on {INTERFACE}... Press Ctrl+C to stop.")
start_time = time.time()

capture = pyshark.LiveCapture(interface=INTERFACE, display_filter="ip")

try:
    for pkt in capture.sniff_continuously():
        packet_callback(pkt)
        if time.time() - start_time > 120:  # Stop after 2 minutes
            break

except KeyboardInterrupt:
    print("[!] Capture stopped manually.")

finally:
    # Save final data before exit
    if packet_data:
        df = pd.DataFrame(packet_data, columns=["Timestamp", "Src_IP", "Dst_IP", "Protocol", "Length", "Src_Port", "Dst_Port"])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"[+] Final save: {len(packet_data)} packets to {OUTPUT_FILE}")
    
    print("[*] Capture completed. Traffic saved to traffic_log.csv.")
