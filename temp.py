import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import socket
import threading
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class DataManager:
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.total_rows = self._get_total_rows()
        self.data_chunks = []
        self.current_data = pd.DataFrame()
    
    def _get_total_rows(self):
        try:
            return sum(1 for _ in open(self.file_path)) - 1  # Exclude header
        except:
            return 0
    
    def load_next_chunk(self):
        try:
            skiprows = self.current_chunk * self.chunk_size + 1 if self.current_chunk > 0 else 0
            chunk = pd.read_csv(self.file_path, skiprows=skiprows, nrows=self.chunk_size)
            if not chunk.empty:
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce')
                self.data_chunks.append(chunk)
                self.current_chunk += 1
                return chunk
        except Exception as e:
            print(f"Error loading chunk: {e}")
        return None
    
    def get_combined_data(self, max_chunks=5):
        if len(self.data_chunks) < max_chunks:
            self.load_next_chunk()
        return pd.concat(self.data_chunks[-max_chunks:], ignore_index=True) if self.data_chunks else pd.DataFrame()

class SecuritySystem:
    def __init__(self):
        self.honeypot = HoneypotSystem()
        self.ids = IntrusionDetectionSystem()
        self.firewall = FirewallSystem()
        self.anomaly_detector = AnomalyDetectionSystem()

class HoneypotSystem:
    def __init__(self, port=2404):
        self.port = port
        self.connection_attempts = []
        self.active = False
        self.server_socket = None

    def start(self):
        self.active = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(('127.0.0.1', self.port))  # Use localhost for safety
            self.server_socket.listen(5)
            threading.Thread(target=self._listen_connections, daemon=True).start()
        except Exception as e:
            print(f"Honeypot start error: {e}")

    def _listen_connections(self):
        while self.active:
            try:
                client, address = self.server_socket.accept()
                attack_info = {
                    'timestamp': datetime.now(),
                    'ip': address[0],
                    'port': address[1],
                    'protocol': 'IEC60870-5-104'
                }
                self.connection_attempts.append(attack_info)
                client.close()
            except:
                break

    def stop(self):
        self.active = False
        if self.server_socket:
            self.server_socket.close()

class IntrusionDetectionSystem:
    def __init__(self):
        self.rules = {'max_connections': 100, 'blacklisted_ips': set()}

    def analyze_packet(self, packet):
        return packet.get('ip') in self.rules['blacklisted_ips']

class FirewallSystem:
    def __init__(self):
        self.rules = {'allowed_ports': {2404}, 'blacklisted_ips': set()}

    def check_packet(self, packet):
        ip, port = packet.get('ip'), packet.get('port')
        return ip not in self.rules['blacklisted_ips'] and port in self.rules['allowed_ports']

    def add_rule(self, rule_type, value):
        if rule_type in self.rules:
            self.rules[rule_type].add(value)

    def remove_rule(self, rule_type, value):
        if rule_type in self.rules:
            self.rules[rule_type].discard(value)

class AnomalyDetectionSystem:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.scaler = StandardScaler()

    def train(self, data):
        if not data.empty:
            scaled_data = self.scaler.fit_transform(data.select_dtypes(include=[np.number]))
            self.model.fit(scaled_data)

    def detect_anomalies(self, data):
        if not data.empty:
            scaled_data = self.scaler.transform(data.select_dtypes(include=[np.number]))
            return self.model.predict(scaled_data)
        return []

class SmartGridSecurityGUI:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("Smart Grid Security System - Chunked Loading")
        self.root.state('zoomed')
        
        self.data_manager = DataManager(data_path)
        self.security_system = SecuritySystem()
        self.is_running = False
        
        # Initialize plotting
        self.fig = Figure(figsize=(12, 8))
        self.setup_plot_axes()
        self.setup_gui()
        
        # Start periodic data loading
        self.schedule_data_loading()

    def setup_plot_axes(self):
        self.attack_ax = self.fig.add_subplot(311)
        self.attack_ax.set_title('Attack Timeline (Recent Data)')
        
        self.performance_ax = self.fig.add_subplot(312)
        self.performance_ax.set_title('System Performance')
        
        self.threat_ax = self.fig.add_subplot(313)
        self.threat_ax.set_title('Threat Distribution')

    def setup_gui(self):
        # Main control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Next Chunk", command=self.load_next_chunk).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Start Honeypot", command=self.start_honeypot).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop Honeypot", command=self.stop_honeypot).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(side='right', padx=5, fill='x', expand=True)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side='right', padx=5)
        
        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Dashboard tab
        dashboard_frame = ttk.Frame(notebook)
        notebook.add(dashboard_frame, text='Dashboard')
        self.setup_dashboard(dashboard_frame)

        # Logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text='Security Logs')
        self.setup_logs_tab(logs_frame)

        # Firewall tab
        firewall_frame = ttk.Frame(notebook)
        notebook.add(firewall_frame, text='Firewall')
        self.setup_firewall_tab(firewall_frame)

    def setup_dashboard(self, parent):
        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas = canvas

    def setup_logs_tab(self, parent):
        # Honeypot logs
        honeypot_frame = ttk.LabelFrame(parent, text="Honeypot Connections")
        honeypot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.honeypot_log = tk.Text(honeypot_frame, height=8)
        scrollbar1 = ttk.Scrollbar(honeypot_frame, orient="vertical", command=self.honeypot_log.yview)
        self.honeypot_log.configure(yscrollcommand=scrollbar1.set)
        self.honeypot_log.pack(side="left", fill="both", expand=True)
        scrollbar1.pack(side="right", fill="y")
        
        # IDS logs
        ids_frame = ttk.LabelFrame(parent, text="IDS Detections")
        ids_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.ids_log = tk.Text(ids_frame, height=8)
        scrollbar2 = ttk.Scrollbar(ids_frame, orient="vertical", command=self.ids_log.yview)
        self.ids_log.configure(yscrollcommand=scrollbar2.set)
        self.ids_log.pack(side="left", fill="both", expand=True)
        scrollbar2.pack(side="right", fill="y")

    def setup_firewall_tab(self, parent):
        rule_frame = ttk.LabelFrame(parent, text="Firewall Rules")
        rule_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.rules_list = tk.Listbox(rule_frame)
        self.rules_list.pack(fill='both', expand=True)

        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Add Test Rule", command=self.add_firewall_rule).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Remove Rule", command=self.remove_firewall_rule).pack(side='left', padx=5)

    def schedule_data_loading(self):
        self.load_next_chunk()
        self.root.after(5000, self.schedule_data_loading)  # Load chunk every 5 seconds

    def load_next_chunk(self):
        chunk = self.data_manager.load_next_chunk()
        if chunk is not None:
            self.update_status(f"Loaded chunk {self.data_manager.current_chunk}")
            self.update_progress()
            self.update_displays()
        else:
            self.update_status("No more data to load")
        return chunk

    def update_progress(self):
        total_chunks = max(1, self.data_manager.total_rows // self.data_manager.chunk_size)
        progress = min(100, (self.data_manager.current_chunk / total_chunks) * 100)
        self.progress['value'] = progress

    def update_status(self, message):
        self.status_label.config(text=message)

    def update_displays(self):
        self.update_plots()
        self.populate_logs()

    def update_plots(self):
        current_data = self.data_manager.get_combined_data()
        if current_data.empty:
            return

        self.attack_ax.clear()
        self.performance_ax.clear()
        self.threat_ax.clear()

        try:
            # Attack timeline
            if 'Timestamp' in current_data.columns:
                timeline_data = current_data.groupby(current_data['Timestamp'].dt.date).size()
                if not timeline_data.empty:
                    self.attack_ax.plot(timeline_data.index, timeline_data.values, 'bo-', label="Attacks/Day")
                    self.attack_ax.legend()
                    self.attack_ax.tick_params(axis='x', rotation=45)

            # Attack types distribution
            if 'Attack_Type' in current_data.columns:
                attack_counts = current_data['Attack_Type'].value_counts().head(10)
                if not attack_counts.empty:
                    self.performance_ax.bar(range(len(attack_counts)), attack_counts.values, 
                                          color='red', alpha=0.7, label="Attack Types")
                    self.performance_ax.set_xticks(range(len(attack_counts)))
                    self.performance_ax.set_xticklabels(attack_counts.index, rotation=45, ha='right')
                    self.performance_ax.legend()

            # Source IP distribution
            if 'Source_IP' in current_data.columns:
                ip_counts = current_data['Source_IP'].value_counts().head(10)
                if not ip_counts.empty:
                    self.threat_ax.bar(range(len(ip_counts)), ip_counts.values, 
                                     color='orange', alpha=0.7, label="Top Source IPs")
                    self.threat_ax.set_xticks(range(len(ip_counts)))
                    self.threat_ax.set_xticklabels([ip[:15] + '...' if len(ip) > 15 else ip 
                                                  for ip in ip_counts.index], rotation=45, ha='right')
                    self.threat_ax.legend()

        except Exception as e:
            print(f"Plot update error: {e}")

        self.fig.tight_layout()
        self.canvas.draw()

    def populate_logs(self):
        current_data = self.data_manager.get_combined_data()
        if current_data.empty:
            return

        # Clear logs periodically to prevent memory issues
        if self.honeypot_log.index('end-1c').split('.')[0] == '100':
            self.honeypot_log.delete('1.0', '50.0')
        if self.ids_log.index('end-1c').split('.')[0] == '100':
            self.ids_log.delete('1.0', '50.0')

        # Add recent entries
        recent_data = current_data.tail(10)
        
        for _, row in recent_data.iterrows():
            if 'Source_IP' in row:
                log_entry = f"[{row.get('Timestamp', 'N/A')}] Connection: {row['Source_IP']} ({row.get('Protocol', 'Unknown')})\n"
                self.honeypot_log.insert('end', log_entry)
                
                if row.get('Malicious', 0) == 1:
                    ids_entry = f"[{row.get('Timestamp', 'N/A')}] ALERT: {row['Source_IP']} - {row.get('Attack_Type', 'Unknown')}\n"
                    self.ids_log.insert('end', ids_entry)

        self.honeypot_log.see('end')
        self.ids_log.see('end')

    def start_honeypot(self):
        self.security_system.honeypot.start()
        self.update_status("Honeypot started")

    def stop_honeypot(self):
        self.security_system.honeypot.stop()
        self.update_status("Honeypot stopped")

    def add_firewall_rule(self):
        rule = f"192.168.1.{np.random.randint(1, 255)}"  # Random test IP
        self.security_system.firewall.add_rule('blacklisted_ips', rule)
        self.rules_list.insert('end', f"Blacklist: {rule}")

    def remove_firewall_rule(self):
        selected = self.rules_list.curselection()
        if selected:
            rule = self.rules_list.get(selected[0]).split(": ")[1]
            self.security_system.firewall.remove_rule('blacklisted_ips', rule)
            self.rules_list.delete(selected[0])

    def generate_report(self):
        current_data = self.data_manager.get_combined_data()
        if not current_data.empty:
            report_path = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            current_data.to_csv(report_path, index=False)
            messagebox.showinfo("Report", f"Report generated: {report_path}")
        else:
            messagebox.showwarning("Report", "No data available for report")

def main():
    root = tk.Tk()
    data_path = "G:/Sem1/Cyberattack_on_smartGrid/intermediate_combined_data.csv"
    
    try:
        app = SmartGridSecurityGUI(root, data_path)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()