import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import threading
import queue
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score
import socket
import logging
import json

class SecuritySystem:
    def __init__(self):
        self.honeypot = HoneypotSystem()
        self.ids = IntrusionDetectionSystem()
        self.firewall = FirewallSystem()
        self.anomaly_detector = AnomalyDetectionSystem()
        self.attack_patterns = []
        self.blocked_ips = set()

class HoneypotSystem:
    def __init__(self, port=2404):  # IEC 60870-5-104 default port
        self.port = port
        self.connection_attempts = queue.Queue()
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
                self.connection_attempts.put(attack_info)
                client.close()
            except Exception as e:
                logging.error(f"Honeypot error: {str(e)}")
                
    def stop(self):
        self.active = False
        if self.server_socket:
            self.server_socket.close()

class IntrusionDetectionSystem:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.rules = {
            'max_connections': 100,
            'suspicious_commands': ['C_IC_NA_1', 'C_CS_NA_1'],  # IEC-104 specific commands
            'blacklisted_ips': set()
        }
        
    def analyze_packet(self, packet_data):
        # Check for IEC-104 specific attacks
        if self._check_protocol_violations(packet_data):
            return True
        return False
        
    def _check_protocol_violations(self, packet):
        # IEC-104 specific checks
        violations = []
        if 'apci' in packet:
            if packet['apci']['length'] > 253:  # Max APDU size
                violations.append('Invalid APDU size')
            if packet['apci']['control_field'] not in [0x00, 0x01, 0x03]:  # Valid control fields
                violations.append('Invalid control field')
        return violations

class FirewallSystem:
    def __init__(self):
        self.rules = {
            'allowed_ips': set(),
            'allowed_ports': {2404},  # IEC-104 default port
            'max_rate': 1000,  # packets per second
            'whitelist': set(),
            'blacklist': set()
        }
        
    def check_packet(self, packet_info):
        ip = packet_info.get('ip')
        port = packet_info.get('port')
        
        if ip in self.rules['blacklist']:
            return False
        if port not in self.rules['allowed_ports']:
            return False
        return True
        
    def add_rule(self, rule_type, value):
        if rule_type in self.rules:
            if isinstance(self.rules[rule_type], set):
                self.rules[rule_type].add(value)
            else:
                self.rules[rule_type] = value

class AnomalyDetectionSystem:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.scaler = StandardScaler()
        self.normal_patterns = {}
        
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
        # Control panel
        control_frame = ttk.LabelFrame(parent, text="System Controls")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Security System", 
                                     command=self.start_system)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Security System",
                                    command=self.stop_system, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Status panel
        status_frame = ttk.LabelFrame(parent, text="System Status")
        status_frame.pack(fill='x', padx=5, pady=5)
        
        self.honeypot_status = ttk.Label(status_frame, text="Honeypot: Inactive")
        self.honeypot_status.pack(side='left', padx=20)
        
        self.ids_status = ttk.Label(status_frame, text="IDS/IPS: Inactive")
        self.ids_status.pack(side='left', padx=20)
        
        self.firewall_status = ttk.Label(status_frame, text="Firewall: Inactive")
        self.firewall_status.pack(side='left', padx=20)
        
        # Dashboard visualization area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_honeypot_tab(self, parent):
        # Honeypot controls
        controls = ttk.LabelFrame(parent, text="Honeypot Controls")
        controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(controls, text="Port:").pack(side='left', padx=5)
        self.port_entry = ttk.Entry(controls, width=10)
        self.port_entry.insert(0, "2404")
        self.port_entry.pack(side='left', padx=5)
        
        # Connection log
        log_frame = ttk.LabelFrame(parent, text="Connection Log")
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.honeypot_log = tk.Text(log_frame, height=10)
        self.honeypot_log.pack(fill='both', expand=True)

    def setup_ids_tab(self, parent):
        # IDS controls
        controls = ttk.LabelFrame(parent, text="IDS Controls")
        controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls, text="Update Rules", 
                  command=self.update_ids_rules).pack(side='left', padx=5)
        
        # Detection log
        log_frame = ttk.LabelFrame(parent, text="Detection Log")
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.ids_log = tk.Text(log_frame, height=10)
        self.ids_log.pack(fill='both', expand=True)

    def setup_firewall_tab(self, parent):
        # Firewall controls
        controls = ttk.LabelFrame(parent, text="Firewall Rules")
        controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls, text="Add Rule", 
                  command=self.add_firewall_rule).pack(side='left', padx=5)
        ttk.Button(controls, text="Remove Rule", 
                  command=self.remove_firewall_rule).pack(side='left', padx=5)
        
        # Rule list
        rules_frame = ttk.LabelFrame(parent, text="Active Rules")
        rules_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.rules_list = tk.Listbox(rules_frame)
        self.rules_list.pack(fill='both', expand=True)

    def setup_analytics_tab(self, parent):
        # Analytics controls
        controls = ttk.LabelFrame(parent, text="Analysis Controls")
        controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        
        # Graphs area
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_plots(self):
        # Clear previous plots
        self.attack_ax.clear()
        self.performance_ax.clear()
        self.threat_ax.clear()

        # Add your plotting logic here
        # Example:
        self.attack_ax.plot(range(10), np.random.rand(10), label='Example Data')
        self.attack_ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def start_system(self):
        self.is_running = True
        self.security_system.honeypot.start()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Update status labels
        self.honeypot_status.config(text="Honeypot: Active")
        self.ids_status.config(text="IDS/IPS: Active")
        self.firewall_status.config(text="Firewall: Active")

        # Start monitoring thread
        threading.Thread(target=self.monitor_system, daemon=True).start()

    def stop_system(self):
        self.is_running = False
        self.security_system.honeypot.stop()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        # Update status labels
        self.honeypot_status.config(text="Honeypot: Inactive")
        self.ids_status.config(text="IDS/IPS: Inactive")
        self.firewall_status.config(text="Firewall: Inactive")

    def monitor_system(self):
        while self.is_running:
            try:
                # Check honeypot
                while not self.security_system.honeypot.connection_attempts.empty():
                    attack = self.security_system.honeypot.connection_attempts.get_nowait()
                    self.log_attack(attack)

                # Update visualizations
                self.root.after(0, self.update_plots)

                # Sleep briefly
                threading.Event().wait(1.0)

            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")

    def log_attack(self, attack):
        log_entry = (f"[{attack['timestamp']}] Connection attempt from "
                     f"{attack['ip']}:{attack['port']} ({attack['protocol']})\n")
        self.honeypot_log.insert('end', log_entry)
        self.honeypot_log.see('end')

    def update_ids_rules(self):
        # Add your IDS rule update logic
        messagebox.showinfo("IDS Rules", "IDS rules updated successfully!")

    def add_firewall_rule(self):
        # Add your firewall rule logic
        messagebox.showinfo("Firewall", "Firewall rule added successfully!")

    def remove_firewall_rule(self):
        # Add your firewall rule removal logic
        messagebox.showinfo("Firewall", "Firewall rule removed successfully!")

    def generate_report(self):
        # Example: Generate a CSV report
        report_path = "security_report.csv"
        with open(report_path, 'w') as f:
            f.write("Timestamp,IP,Port,Protocol\n")
            while not self.security_system.honeypot.connection_attempts.empty():
                attack = self.security_system.honeypot.connection_attempts.get_nowait()
                f.write(f"{attack['timestamp']},{attack['ip']},{attack['port']},{attack['protocol']}\n")

        messagebox.showinfo("Report", f"Report generated at {report_path}")


def main():
    root = tk.Tk()
    data_path = "F:/mrudula college/Sem1_project/Cyberattack_on_smartGrid/intermediate_combined_data.csv"
    app = SmartGridSecurityGUI(root, data_path)
    root.mainloop()


if __name__ == "__main__":
    main()