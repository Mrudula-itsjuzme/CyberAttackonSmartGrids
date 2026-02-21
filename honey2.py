import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
from datetime import datetime
import threading
import queue
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np

class SimulatedAttack:
    def __init__(self):
        self.attack_types = ['SQL Injection', 'Brute Force', 'DDoS', 'XSS', 'Port Scan']
        self.countries = ['USA', 'China', 'Russia', 'Brazil', 'India']
        self.ips = [
            '192.168.1.', '10.0.0.', '172.16.0.', '8.8.8.',
            '1.1.1.', '9.9.9.', '208.67.222.', '208.67.220.'
        ]

    def generate_attack(self):
        attack_type = random.choice(self.attack_types)
        country = random.choice(self.countries)
        ip = random.choice(self.ips) + str(random.randint(1, 255))
        severity = random.randint(1, 100)
        return {
            'timestamp': datetime.now(),
            'type': attack_type,
            'source_ip': ip,
            'country': country,
            'severity': severity
        }

class EnhancedHoneypotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Honeypot Training System")
        self.root.state('zoomed')  # Maximize window
        
        # Initialize simulation components
        self.simulator = SimulatedAttack()
        self.attack_queue = queue.Queue()
        self.is_running = False
        self.attack_history = []
        self.score = 0
        self.accuracy = 0
        
        self.setup_gui()
        self.setup_plots()
        
    def setup_gui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Alert.TFrame', background='#FFE4E1')
        style.configure('Success.TFrame', background='#E0FFE0')
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Simulation", 
                                     command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Simulation", 
                                    command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Stats Panel
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.score_label = ttk.Label(stats_frame, text="Defense Score: 0", 
                                   font=('Arial', 12, 'bold'))
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.accuracy_label = ttk.Label(stats_frame, text="Detection Accuracy: 0%", 
                                      font=('Arial', 12, 'bold'))
        self.accuracy_label.pack(side=tk.LEFT, padx=20)
        
        self.attack_count_label = ttk.Label(stats_frame, text="Total Attacks: 0", 
                                          font=('Arial', 12, 'bold'))
        self.attack_count_label.pack(side=tk.LEFT, padx=20)
        
        # Create split view for graph and alerts
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Graph frame
        self.graph_frame = ttk.Frame(paned_window)
        paned_window.add(self.graph_frame, weight=1)
        
        # Alert frame
        alert_container = ttk.Frame(paned_window)
        paned_window.add(alert_container, weight=1)
        
        # Alert header
        ttk.Label(alert_container, text="Live Attack Feed", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Alert list with scrollbar
        self.alert_frame = ttk.Frame(alert_container)
        self.alert_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame for alerts
        canvas = tk.Canvas(self.alert_frame)
        scrollbar = ttk.Scrollbar(self.alert_frame, orient="vertical", 
                                command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_plots(self):
        # Create figure for plots
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.attack_ax = self.fig.add_subplot(211)
        self.severity_ax = self.fig.add_subplot(212)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot data
        self.timestamps = []
        self.severities = []
        
        # Configure plots
        self.attack_ax.set_title('Attack Timeline')
        self.severity_ax.set_title('Attack Severity Distribution')
        self.fig.tight_layout()

    def update_plots(self):
        # Clear previous plots
        self.attack_ax.clear()
        self.severity_ax.clear()
        
        # Update attack timeline
        if self.timestamps:
            self.attack_ax.plot(self.timestamps, self.severities, 'b-')
            self.attack_ax.set_title('Attack Timeline')
            self.attack_ax.set_xlabel('Time')
            self.attack_ax.set_ylabel('Severity')
            
        # Update severity distribution
        if self.severities:
            sns.histplot(self.severities, ax=self.severity_ax, kde=True)
            self.severity_ax.set_title('Attack Severity Distribution')
            self.severity_ax.set_xlabel('Severity')
            self.severity_ax.set_ylabel('Count')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def add_alert(self, attack):
        # Create alert frame with color based on severity
        color = '#FFE4E1' if attack['severity'] > 70 else '#E0FFE0'
        alert = ttk.Frame(self.scrollable_frame, style='Alert.TFrame')
        alert.configure(style='Alert.TFrame')
        alert.pack(fill=tk.X, padx=5, pady=2)
        
        # Add alert content
        ttk.Label(alert, text=f"Type: {attack['type']}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(alert, text=f"Source: {attack['source_ip']} ({attack['country']})"
                 ).pack(anchor='w')
        ttk.Label(alert, text=f"Severity: {attack['severity']}%").pack(anchor='w')
        ttk.Label(alert, text=f"Time: {attack['timestamp'].strftime('%H:%M:%S')}"
                 ).pack(anchor='w')
        
        # Keep only last 10 alerts
        if len(self.scrollable_frame.winfo_children()) > 10:
            self.scrollable_frame.winfo_children()[0].destroy()

    def simulate_attack(self):
        while self.is_running:
            attack = self.simulator.generate_attack()
            self.attack_queue.put(attack)
            self.attack_history.append(attack)
            
            # Update plots data
            self.timestamps.append(attack['timestamp'])
            self.severities.append(attack['severity'])
            
            # Keep only last 50 data points
            if len(self.timestamps) > 50:
                self.timestamps.pop(0)
                self.severities.pop(0)
            
            # Update statistics
            self.score += attack['severity'] // 10
            self.accuracy = random.uniform(80, 100)  # Simulated accuracy
            
            # Update GUI
            self.root.after(0, self.update_gui, attack)
            
            # Simulate detection time
            time.sleep(random.uniform(1, 3))

    def update_gui(self, attack):
        # Update labels
        self.score_label.config(text=f"Defense Score: {self.score}")
        self.accuracy_label.config(text=f"Detection Accuracy: {self.accuracy:.1f}%")
        self.attack_count_label.config(text=f"Total Attacks: {len(self.attack_history)}")
        
        # Add new alert
        self.add_alert(attack)
        
        # Update plots
        self.update_plots()

    def start_simulation(self):
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.simulate_attack, daemon=True).start()

    def stop_simulation(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = EnhancedHoneypotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()