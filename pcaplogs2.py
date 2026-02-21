import os
import pandas as pd
from scapy.all import PcapReader, IP, TCP
import numpy as np
from tqdm import tqdm
import glob
from collections import defaultdict
import multiprocessing as mp
from functools import partial

# Configuration
BASE_DIR = r"G:\Sem1\Cyberattack_on_smartGrid\IEC 60870-5-104 SEG Intrusion Detection Dataset"
OUTPUT_CSV = "pcap_feature_summary.csv"

def extract_pcap_metrics(pcap_path):
    """Fast extraction of network metrics from PCAP file"""
    try:
        # Fast packet reading with PcapReader (streaming)
        times, lengths, flows, tcp_flags = [], [], defaultdict(list), []
        
        with PcapReader(pcap_path) as pcap:
            for i, pkt in enumerate(pcap):
                if IP not in pkt:
                    continue
                
                # Limit packets for speed (sample large files)
                if i > 50000:  # Process max 50k packets
                    break
                    
                t = float(pkt.time)
                times.append(t)
                lengths.append(len(pkt))
                
                # Flow tracking (simplified)
                flow_key = f"{pkt[IP].src}-{pkt[IP].dst}"
                flows[flow_key].append(t)
                
                # TCP flags for RTT estimation
                if TCP in pkt and len(tcp_flags) < 1000:  # Limit TCP analysis
                    tcp_flags.append((t, pkt[TCP].flags, flow_key))
        
        if len(times) < 2:
            return None
        
        # Fast numpy calculations
        times = np.array(times)
        lengths = np.array(lengths)
        times.sort()
        iats = np.diff(times)
        
        duration = times[-1] - times[0]
        total_packets = len(times)
        total_bytes = np.sum(lengths)
        
        # Basic metrics
        mean_iat = np.mean(iats) * 1000
        jitter = np.std(iats) * 1000
        min_iat = np.min(iats) * 1000
        max_iat = np.max(iats) * 1000
        
        # Fast log propagation analysis
        log_metrics = fast_log_analysis(flows, iats)
        
        # Fast latency analysis
        rtt_est = fast_rtt_estimation(tcp_flags) if tcp_flags else 0
        
        # Fast burst analysis
        burst_metrics = fast_burst_analysis(iats)
        
        return {
            'File': os.path.basename(pcap_path),
            'Flow_Duration_s': round(duration, 3),
            'Total_Packets': total_packets,
            'Total_Bytes': int(total_bytes),
            'PacketsPerSecond': round(total_packets / duration, 3) if duration > 0 else 0,
            'Throughput_BytesPerSec': round(total_bytes / duration, 3) if duration > 0 else 0,
            'Utilization_Mbps': round((total_bytes * 8) / (duration * 1000000), 3) if duration > 0 else 0,
            
            # Basic IAT metrics
            'Mean_IAT_ms': round(mean_iat, 3),
            'Min_IAT_ms': round(min_iat, 3),
            'Max_IAT_ms': round(max_iat, 3),
            'Jitter_ms': round(jitter, 3),
            
            # Log Propagation (simplified)
            'Log_Prop_Delay_ms': log_metrics['prop_delay'],
            'Flow_Count': log_metrics['flow_count'],
            'Event_Burst_Factor': log_metrics['burst_factor'],
            
            # Network Latency
            'RTT_Estimate_ms': round(rtt_est, 3),
            'One_Way_Delay_ms': round(rtt_est / 2, 3),
            'Queue_Delay_Variance': round(np.var(iats) * 1000000, 3),
            
            # Burst Analysis
            'Burst_Intensity': burst_metrics['intensity'],
            'Burst_Ratio': burst_metrics['ratio'],
            
            # Packet Size Analysis
            'Avg_Pkt_Size': round(np.mean(lengths), 2),
            'Min_Pkt_Size': int(np.min(lengths)),
            'Max_Pkt_Size': int(np.max(lengths)),
            'Pkt_Size_Std': round(np.std(lengths), 2)
        }
        
    except Exception as e:
        print(f"[ERROR] {pcap_path}: {e}")
        return None

def fast_log_analysis(flows, iats):
    """Simplified log propagation analysis"""
    flow_count = len(flows)
    
    if flow_count < 2:
        return {'prop_delay': 0, 'flow_count': flow_count, 'burst_factor': 0}
    
    # Quick propagation delay estimate
    prop_delay = np.percentile(iats, 90) * 1000  # 90th percentile as prop delay
    
    # Burst factor (simplified)
    burst_threshold = np.percentile(iats, 25)
    burst_factor = np.mean(iats < burst_threshold)
    
    return {
        'prop_delay': round(prop_delay, 3),
        'flow_count': flow_count,
        'burst_factor': round(burst_factor, 3)
    }

def fast_rtt_estimation(tcp_flags):
    """Fast RTT estimation from TCP flags"""
    if len(tcp_flags) < 4:
        return 0
    
    # Look for SYN-ACK patterns (simplified)
    syn_times = {}
    rtt_samples = []
    
    for t, flags, flow in tcp_flags[:500]:  # Limit analysis
        if flags & 0x02:  # SYN
            syn_times[flow] = t
        elif flags & 0x10 and flow in syn_times:  # ACK
            rtt = (t - syn_times[flow]) * 1000
            if 0 < rtt < 1000:
                rtt_samples.append(rtt)
            syn_times.pop(flow, None)  # Remove to avoid duplicates
    
    return np.median(rtt_samples) if rtt_samples else 0

def fast_burst_analysis(iats):
    """Fast burst pattern analysis"""
    if len(iats) < 10:
        return {'intensity': 0, 'ratio': 0}
    
    # Quick burst detection
    burst_threshold = 0.05  # 50ms
    burst_packets = np.sum(iats < burst_threshold)
    
    intensity = burst_packets / len(iats)
    ratio = np.mean(iats < np.median(iats))
    
    return {
        'intensity': round(intensity, 3),
        'ratio': round(ratio, 3)
    }

def process_single_file(pcap_file):
    """Process single PCAP file - for multiprocessing"""
    return extract_pcap_metrics(pcap_file)

def main():
    # Find all PCAP files
    pcap_pattern = os.path.join(BASE_DIR, "**", "*.pcap")
    pcap_files = glob.glob(pcap_pattern, recursive=True)
    
    print(f"🔍 Found {len(pcap_files)} PCAP files")
    
    # Use multiprocessing for speed
    cpu_count = min(mp.cpu_count(), 4)  # Limit to 4 cores max
    print(f"🚀 Using {cpu_count} CPU cores")
    
    summary_data = []
    
    if len(pcap_files) > 1 and cpu_count > 1:
        # Parallel processing
        with mp.Pool(cpu_count) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, pcap_files),
                total=len(pcap_files),
                desc="Processing PCAPs"
            ))
        summary_data = [r for r in results if r is not None]
    else:
        # Sequential processing with progress bar
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