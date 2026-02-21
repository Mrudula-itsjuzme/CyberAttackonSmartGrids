import os

def repair_pcap(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        # Check for valid PCAP magic number at the start
        if not data.startswith(b'\xd4\xc3\xb2\xa1') and not data.startswith(b'\xa1\xb2\xc3\xd4'):
            print(f"{file_path} has an invalid header. Attempting to fix...")
            # Add a basic PCAP header
            pcap_header = b'\xd4\xc3\xb2\xa1\x02\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\x00\x00'
            data = pcap_header + data
        
        # Write the "fixed" file
        repaired_file = os.path.join("repaired_files", os.path.basename(file_path))
        with open(repaired_file, "wb") as f:
            f.write(data)
        print(f"Repaired {file_path} -> {repaired_file}")
    except Exception as e:
        print(f"Failed to repair {file_path}: {e}")

# Directory paths
quarantine_dir = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\quarantine_failed_files"
repaired_dir = "repaired_files"
os.makedirs(repaired_dir, exist_ok=True)

# Process files
for file in os.listdir(quarantine_dir):
    file_path = os.path.join(quarantine_dir, file)
    if file.endswith(".pcap"):
        repair_pcap(file_path)
