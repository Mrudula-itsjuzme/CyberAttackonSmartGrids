import os
from scapy.all import rdpcap, wrpcap

# Directory containing all PCAP files (including subdirectories)
input_folder = r"F:\IEC 60870-5-104 SEG Intrusion Detection Dataset"  # Replace with your folder path
output_file = "combined_output.pcap"  # Output combined file name

def combine_pcaps_recursive(input_folder, output_file):
    # Collect all PCAP files recursively from the directory
    pcap_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files.append(os.path.join(root, file))
    
    if not pcap_files:
        print("No PCAP files found in the specified directory or its subdirectories!")
        return

    print(f"Found {len(pcap_files)} PCAP files. Combining...")
    combined_packets = []

    # Read packets from each PCAP file
    for pcap_file in pcap_files:
        try:
            print(f"Reading packets from {pcap_file}...")
            packets = rdpcap(pcap_file)
            combined_packets.extend(packets)
        except Exception as e:
            print(f"Skipping file {pcap_file} due to error: {e}")

    # Write all packets to the combined output file
    if combined_packets:
        wrpcap(output_file, combined_packets)
        print(f"Combined PCAP saved to {output_file}")
    else:
        print("No valid packets found to combine.")

# Run the function
combine_pcaps_recursive(input_folder, output_file)
