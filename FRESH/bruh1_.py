import os
import pandas as pd
import dpkt
from multiprocessing import Pool

# Top-level function to process a single pcap file
def process_pcap_worker(file_path):
    print(f"Processing: {file_path}")
    packet_data = []

    try:
        with open(file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data

                    data = {
                        'timestamp': timestamp,
                        'src_ip': dpkt.utils.inet_to_str(ip.src) if hasattr(ip, 'src') else None,
                        'dst_ip': dpkt.utils.inet_to_str(ip.dst) if hasattr(ip, 'dst') else None,
                        'protocol': type(ip).__name__,  # Protocol type (e.g., TCP/UDP)
                        'length': len(buf),
                    }
                    packet_data.append(data)
                except Exception:
                    # Skip corrupted packets
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return packet_data

# Function to process all pcap files in parallel
def process_all_pcaps_parallel(input_dir, output_csv, num_workers=4):
    pcap_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.endswith('.pcap')
    ]

    print(f"Found {len(pcap_files)} pcap files to process.")

    # Process files in parallel
    with Pool(num_workers) as pool:
        all_data = pool.map(process_pcap_worker, pcap_files)

    # Flatten the list of lists into a single list
    flattened_data = [item for sublist in all_data for item in sublist]

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_csv, index=False)
    print(f"All pcap files combined into: {output_csv}")

if __name__ == "__main__":
    input_directory = r"F:\IEC 60870-5-104 SEG Intrusion Detection Dataset"  # Change this
    output_file = r'output_combined.csv'  # Name of output file

    # Run the function
    process_all_pcaps_parallel(input_directory, output_file, num_workers=6)  # Adjust workers based on your CPU cores
