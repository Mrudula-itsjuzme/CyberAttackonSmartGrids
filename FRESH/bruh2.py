import os
import pandas as pd
import dpkt
from multiprocessing import get_context

# Lists to track processed and failed files
processed_files = []
failed_files = []

# Function to check if a file is a valid PCAP format
def is_valid_pcap(file_path):
    try:
        with open(file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            # Try reading a packet to validate
            next(iter(pcap))
        return True
    except Exception:
        return False

# Worker function to process a single pcap file
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
                        'protocol': type(ip).__name__,
                        'length': len(buf),
                    }
                    packet_data.append(data)
                except Exception:
                    # Skip invalid packets
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        failed_files.append(file_path)
        return []

    processed_files.append(file_path)
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

    # Validate files before processing
    valid_pcap_files = [file for file in pcap_files if is_valid_pcap(file)]
    invalid_pcap_files = [file for file in pcap_files if file not in valid_pcap_files]

    # Log invalid files
    if invalid_pcap_files:
        failed_files.extend(invalid_pcap_files)
        print(f"Found {len(invalid_pcap_files)} invalid PCAP files. Skipping them.")

    # Use spawn context for multiprocessing
    with get_context("spawn").Pool(num_workers) as pool:
        all_data = pool.map(process_pcap_worker, valid_pcap_files)

    # Flatten the list of lists into a single list
    flattened_data = [item for sublist in all_data for item in sublist]

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_csv, index=False)
    print(f"All valid PCAP files combined into: {output_csv}")

    # Log failed files
    if failed_files:
        with open("failed_files.log", "w") as log_file:
            log_file.write("\n".join(failed_files))
        print(f"Logged failed files to: failed_files.log")

    # Log successfully processed files
    if processed_files:
        with open("processed_files.log", "w") as log_file:
            log_file.write("\n".join(processed_files))
        print(f"Logged processed files to: processed_files.log")

if __name__ == "__main__":
    # Set your directory and output file
    input_directory = r"F:\IEC 60870-5-104 SEG Intrusion Detection Dataset"  # Change this
    output_file = r'output_combined.csv'  # Name of output file

    # Run the function
    process_all_pcaps_parallel(input_directory, output_file, num_workers=8)
