import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class AdvancedHoneypotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Honeypot Training System")
        self.root.state('zoomed')
        
        # Initialize components
        self.attack_queue = queue.Queue()
        self.is_running = False
        self.attack_history = []
        self.score = 0
        self.accuracy = 0
        self.scaler = StandardScaler()
        
        self.setup_gui()
        self.setup_plots()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Analysis", 
                                     command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Analysis", 
                                    command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Stats Panel
        stats_frame = ttk.LabelFrame(main_frame, text="Detection Statistics", padding="5")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.score_label = ttk.Label(stats_frame, text="Detection Score: 0", 
                                   font=('Arial', 12, 'bold'))
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.accuracy_label = ttk.Label(stats_frame, text="Model Accuracy: 0%", 
                                      font=('Arial', 12, 'bold'))
        self.accuracy_label.pack(side=tk.LEFT, padx=20)
        
        self.attack_count_label = ttk.Label(stats_frame, text="Attacks Analyzed: 0", 
                                          font=('Arial', 12, 'bold'))
        self.attack_count_label.pack(side=tk.LEFT, padx=20)
        
        # Create split view
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Graph frame
        self.graph_frame = ttk.Frame(paned_window)
        paned_window.add(self.graph_frame, weight=1)
        
        # Alert frame
        alert_container = ttk.Frame(paned_window)
        paned_window.add(alert_container, weight=1)
        
        # Alert header
        ttk.Label(alert_container, text="Detection Results", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Scrollable alert frame
        self.alert_frame = ttk.Frame(alert_container)
        self.alert_frame.pack(fill=tk.BOTH, expand=True)
        
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
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.attack_ax = self.fig.add_subplot(211)
        self.metrics_ax = self.fig.add_subplot(212)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.timestamps = []
        self.detection_scores = []
        self.accuracy_history = []
        
        self.attack_ax.set_title('Attack Detection Timeline')
        self.metrics_ax.set_title('Model Performance Metrics')
        self.fig.tight_layout()

    def update_plots(self):
        self.attack_ax.clear()
        self.metrics_ax.clear()
        
        if self.timestamps:
            self.attack_ax.plot(self.timestamps, self.detection_scores, 'b-')
            self.attack_ax.set_title('Attack Detection Timeline')
            self.attack_ax.set_xlabel('Time')
            self.attack_ax.set_ylabel('Detection Score')
            
        if self.accuracy_history:
            self.metrics_ax.plot(range(len(self.accuracy_history)), 
                               self.accuracy_history, 'g-', label='Accuracy')
            self.metrics_ax.set_title('Model Performance Over Time')
            self.metrics_ax.set_xlabel('Analysis Run')
            self.metrics_ax.set_ylabel('Accuracy Score')
            self.metrics_ax.legend()
        
        self.fig.tight_layout()
        self.canvas.draw()

    def add_detection_result(self, result):
        alert = ttk.Frame(self.scrollable_frame)
        alert.pack(fill=tk.X, padx=5, pady=2)
        
        style = 'Success.TFrame' if result['detected'] else 'Alert.TFrame'
        alert.configure(style=style)
        
        ttk.Label(alert, text=f"Attack Type: {result['type']}", 
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(alert, text=f"Confidence: {result['confidence']:.2f}%").pack(anchor='w')
        ttk.Label(alert, text=f"Time: {result['timestamp'].strftime('%H:%M:%S')}"
                 ).pack(anchor='w')
        
        if len(self.scrollable_frame.winfo_children()) > 10:
            self.scrollable_frame.winfo_children()[0].destroy()

    def analyze_data(self):
        while self.is_running:
            try:
                # Simulate detection results with your ADT
                result = self.generate_detection_result()
                self.attack_history.append(result)
                
                # Update visualization data
                self.timestamps.append(result['timestamp'])
                self.detection_scores.append(result['confidence'])
                self.accuracy_history.append(result['model_accuracy'])
                
                # Keep recent data points
                if len(self.timestamps) > 50:
                    self.timestamps.pop(0)
                    self.detection_scores.pop(0)
                
                # Update score and accuracy
                self.score += result['confidence'] / 10
                self.accuracy = np.mean(self.accuracy_history)
                
                # Update GUI
                self.root.after(0, self.update_gui, result)
                
                # Simulate analysis time
                threading.Event().wait(2.0)
                
            except Exception as e:
                print(f"Analysis error: {str(e)}")

    def generate_detection_result(self):
        # This would be replaced with your actual ADT detection logic
        return {
            'timestamp': datetime.now(),
            'type': np.random.choice(['SQL Injection', 'XSS', 'DDoS', 'Port Scan']),
            'detected': np.random.choice([True, False], p=[0.8, 0.2]),
            'confidence': np.random.uniform(60, 100),
            'model_accuracy': np.random.uniform(0.75, 0.95)
        }

    def update_gui(self, result):
        self.score_label.config(text=f"Detection Score: {self.score:.0f}")
        self.accuracy_label.config(text=f"Model Accuracy: {self.accuracy:.1f}%")
        self.attack_count_label.config(text=f"Attacks Analyzed: {len(self.attack_history)}")
        
        self.add_detection_result(result)
        self.update_plots()

    def start_analysis(self):
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.analyze_data, daemon=True).start()

    def stop_analysis(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = AdvancedHoneypotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()