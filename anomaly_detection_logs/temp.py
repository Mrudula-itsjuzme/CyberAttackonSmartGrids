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

class SecuritySystem:
    def __init__(self):
        self.honeypot = HoneypotSystem()
        self.ids = IntrusionDetectionSystem()
        self.firewall = FirewallSystem()
        self.anomaly_detector = AnomalyDetectionSystem()

class HoneypotSystem:
    def __init__(self, port=2404):  # Default IEC 60870-5-104 port
        self.port = port
        self.connection_attempts = []
        self.active = False
        self.server_socket = None

    def start(self):
        self.active = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(5)
        threading.Thread(target=self._listen_connections, daemon=True).start()

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
            except Exception as e:
                print(f"Honeypot error: {e}")

    def stop(self):
        self.active = False
        if self.server_socket:
            self.server_socket.close()

class IntrusionDetectionSystem:
    def __init__(self):
        self.rules = {
            'max_connections': 100,
            'blacklisted_ips': set()
        }

    def analyze_packet(self, packet):
        # Placeholder for protocol-specific logic
        if packet['ip'] in self.rules['blacklisted_ips']:
            return True
        return False

class FirewallSystem:
    def __init__(self):
        self.rules = {
            'allowed_ports': {2404},
            'blacklisted_ips': set()
        }

    def check_packet(self, packet):
        ip = packet.get('ip')
        port = packet.get('port')
        if ip in self.rules['blacklisted_ips'] or port not in self.rules['allowed_ports']:
            return False
        return True

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
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)

    def detect_anomalies(self, data):
        scaled_data = self.scaler.transform(data)
        predictions = self.model.predict(scaled_data)
        return predictions

class SmartGridSecurityGUI:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("Smart Grid Security System")
        self.root.state('zoomed')

        self.data_path = data_path
        self.dataset = pd.read_csv(self.data_path)  # Load dataset
        self.dataset['Timestamp'] = pd.to_datetime(self.dataset['Timestamp'], errors='coerce')


        self.security_system = SecuritySystem()
        self.is_running = False

        # Initialize plotting
        self.fig = Figure(figsize=(10, 8))
        self.setup_plot_axes()

        # Set up the main GUI
        self.setup_gui()

    def setup_plot_axes(self):
        # Attack timeline
        self.attack_ax = self.fig.add_subplot(311)
        self.attack_ax.set_title('Attack Timeline')

        # System performance
        self.performance_ax = self.fig.add_subplot(312)
        self.performance_ax.set_title('System Performance')

        # Threat distribution
        self.threat_ax = self.fig.add_subplot(313)
        self.threat_ax.set_title('Threat Distribution')

    def setup_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Dashboard tab
        dashboard_frame = ttk.Frame(notebook)
        notebook.add(dashboard_frame, text='Dashboard')
        self.setup_dashboard(dashboard_frame)

        # Honeypot tab
        honeypot_frame = ttk.Frame(notebook)
        notebook.add(honeypot_frame, text='Honeypot')
        self.setup_honeypot_tab(honeypot_frame)

        # IDS tab
        ids_frame = ttk.Frame(notebook)
        notebook.add(ids_frame, text='IDS/IPS')
        self.setup_ids_tab(ids_frame)

        # Firewall tab
        firewall_frame = ttk.Frame(notebook)
        notebook.add(firewall_frame, text='Firewall')
        self.setup_firewall_tab(firewall_frame)

        # Analytics tab
        self.analytics_frame = ttk.Frame(notebook)
        notebook.add(self.analytics_frame, text='Analytics')
        self.setup_analytics_tab(self.analytics_frame)

    def setup_dashboard(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)

        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_honeypot_tab(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Connection Log")
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.honeypot_log = tk.Text(log_frame, height=10)
        self.honeypot_log.pack(fill='both', expand=True)

    def setup_ids_tab(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Detection Log")
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.ids_log = tk.Text(log_frame, height=10)
        self.ids_log.pack(fill='both', expand=True)

    def setup_firewall_tab(self, parent):
        rule_frame = ttk.LabelFrame(parent, text="Firewall Rules")
        rule_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.rules_list = tk.Listbox(rule_frame)
        self.rules_list.pack(fill='both', expand=True)

        control_frame = ttk.LabelFrame(parent, text="Controls")
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(control_frame, text="Add Rule", command=self.add_firewall_rule).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Remove Rule", command=self.remove_firewall_rule).pack(side='left', padx=5)

    def setup_analytics_tab(self, parent):
        controls = ttk.LabelFrame(parent, text="Analysis Controls")
        controls.pack(fill='x', padx=5, pady=5)

        ttk.Button(controls, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)

    def update_plots(self):
        self.attack_ax.clear()
        self.performance_ax.clear()
        self.threat_ax.clear()

        if not self.dataset.empty:
            timeline_data = self.dataset.groupby(self.dataset['Timestamp'].dt.date).size()
            self.attack_ax.plot(timeline_data.index, timeline_data.values, marker='o', label="Attack Count")
            self.attack_ax.legend()

            attack_counts = self.dataset['Attack_Type'].value_counts()
            self.performance_ax.bar(attack_counts.index, attack_counts.values, label="Attack Intensity")
            self.performance_ax.legend()

            threat_types = self.dataset['Attack_Type'].value_counts()
            self.threat_ax.bar(threat_types.index, threat_types.values, label="Threat Types")
            self.threat_ax.legend()

        self.fig.tight_layout()

    def populate_honeypot_logs(self):
        for _, row in self.dataset.iterrows():
            log_entry = f"[{row['Timestamp']}] Connection attempt from {row['Source_IP']} ({row.get('Protocol', 'Unknown')})\n"
            self.honeypot_log.insert('end', log_entry)
            self.honeypot_log.see('end')

    def populate_ids_logs(self):
        malicious_data = self.dataset[self.dataset['Malicious'] == 1]
        for _, row in malicious_data.iterrows():
            log_entry = f"[{row['Timestamp']}] IDS detected suspicious activity from {row['Source_IP']} ({row['Attack_Type']})\n"
            self.ids_log.insert('end', log_entry)
            self.ids_log.see('end')

    def add_firewall_rule(self):
        rule = "192.168.1.100"  # Example IP for testing
        self.security_system.firewall.add_rule('blacklisted_ips', rule)
        self.rules_list.insert('end', f"Blacklist: {rule}")

    def remove_firewall_rule(self):
        selected = self.rules_list.curselection()
        if selected:
            rule = self.rules_list.get(selected[0]).split(": ")[1]
            self.security_system.firewall.remove_rule('blacklisted_ips', rule)
            self.rules_list.delete(selected[0])

    def generate_report(self):
        report_path = "security_report.csv"
        self.dataset.to_csv(report_path, index=False)
        messagebox.showinfo("Report", f"Report generated successfully at {report_path}")

# Main Function
def main():
    root = tk.Tk()
    data_path = "F:/mrudula college/Sem1_project/Cyberattack_on_smartGrid/intermediate_combined_data.csv"
    app = SmartGridSecurityGUI(root, data_path)
    app.populate_honeypot_logs()
    app.populate_ids_logs()
    app.update_plots()
    root.mainloop()

if __name__ == "__main__":
    main()
