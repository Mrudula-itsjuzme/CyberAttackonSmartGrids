import os
import pandas as pd
from scapy.all import rdpcap, IP
import numpy as np
from tqdm import tqdm
import glob

# Configuration
BASE_DIR = r"G:\Sem1\Cyberattack_on_smartGrid\IEC 60870-5-104 SEG Intrusion Detection Dataset"
OUTPUT_CSV = "pcap_feature_summary.csv"

def extract_pcap_metrics(pcap_path):
    """Extract network metrics from PCAP file using Scapy"""
    try:
        print(f"📂 Processing: {os.path.basename(pcap_path)}")
        
        # Read packets with IP layer only
        packets = rdpcap(pcap_path)
        ip_packets = [pkt for pkt in packets if IP in pkt]
        
        if len(ip_packets) < 2:
            return None
        
        # Extract timestamps and packet sizes
        times = [float(pkt.time) for pkt in ip_packets]
        lengths = [len(pkt) for pkt in ip_packets]
        
        # Calculate metrics
        times.sort()
        iats = np.diff(times)  # Inter-arrival times
        duration = times[-1] - times[0]
        total_packets = len(ip_packets)
        total_bytes = sum(lengths)
        
        return {
            'File': os.path.basename(pcap_path),
            'Mean_IAT_ms': round(np.mean(iats) * 1000, 3),
            'Std_IAT_ms': round(np.std(iats) * 1000, 3),
            'Flow_Duration_s': round(duration, 3),
            'Total_Packets': total_packets,
            'Total_Bytes': total_bytes,
            'Throughput_BytesPerSec': round(total_bytes / duration, 3) if duration > 0 else 0,
            'PacketsPerSecond': round(total_packets / duration, 3) if duration > 0 else 0,
            'Path': pcap_path
        }
        
    except Exception as e:
        print(f"[ERROR] {pcap_path}: {e}")
        return None

def main():
    # Find all PCAP files
    pcap_pattern = os.path.join(BASE_DIR, "**", "*.pcap")
    pcap_files = glob.glob(pcap_pattern, recursive=True)
    
    print(f"🔍 Found {len(pcap_files)} PCAP files")
    
    # Process files one by one
    summary_data = []
    for pcap_file in tqdm(pcap_files, desc="Processing PCAPs"):
        result = extract_pcap_metrics(pcap_file)
        if result:
            summary_data.append(result)
    
    # Save results
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Summary saved to: {OUTPUT_CSV}")
        print(f"📊 Processed {len(summary_data)} files successfully")
    else:
        print("⚠️ No valid PCAPs processed.")

if __name__ == "__main__":
    main()