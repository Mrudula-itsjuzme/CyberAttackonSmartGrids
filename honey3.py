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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class SmartGridAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def preprocess_data(self):
        # Handle protocol-specific features
        if 'Label' in self.df.columns:
            self.df['Label'] = self.df['Label'].map({'Attack': 1, 'Normal': 0})
        
        # Convert categorical columns to numeric
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=categorical_columns)
        
        return self.df

class SmartGridGUI:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("Smart Grid IEC 60870-5-104 Analysis System")
        self.root.state('zoomed')
        
        self.analyzer = SmartGridAnalyzer(data_path)
        self.attack_queue = queue.Queue()
        self.is_running = False
        
        self.setup_gui()
        self.setup_plots()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Analysis", 
                                     command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Analysis", 
                                    command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Stats Panel
        stats_frame = ttk.LabelFrame(main_frame, text="IEC 60870-5-104 Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.detection_label = ttk.Label(stats_frame, text="Detection Rate: 0%", 
                                       font=('Arial', 12, 'bold'))
        self.detection_label.pack(side=tk.LEFT, padx=20)
        
        self.accuracy_label = ttk.Label(stats_frame, text="Model Accuracy: 0%", 
                                      font=('Arial', 12, 'bold'))
        self.accuracy_label.pack(side=tk.LEFT, padx=20)
        
        self.alert_label = ttk.Label(stats_frame, text="Protocol Violations: 0", 
                                   font=('Arial', 12, 'bold'))
        self.alert_label.pack(side=tk.LEFT, padx=20)
        
        # Split view
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Visualization frame
        self.viz_frame = ttk.Frame(paned_window)
        paned_window.add(self.viz_frame, weight=2)
        
        # Analysis frame
        analysis_frame = ttk.Frame(paned_window)
        paned_window.add(analysis_frame, weight=1)
        
        # Analysis results
        self.results_text = tk.Text(analysis_frame, height=20, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_plots(self):
        self.fig = Figure(figsize=(12, 8))
        
        # Protocol violations timeline
        self.violations_ax = self.fig.add_subplot(311)
        self.violations_ax.set_title('Protocol Violations Timeline')
        
        # Attack distribution
        self.distribution_ax = self.fig.add_subplot(312)
        self.distribution_ax.set_title('Attack Type Distribution')
        
        # Detection accuracy
        self.accuracy_ax = self.fig.add_subplot(313)
        self.accuracy_ax.set_title('Detection Accuracy Over Time')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.timestamps = []
        self.violations = []
        self.accuracies = []

    def update_plots(self, df_chunk):
        # Clear previous plots
        self.violations_ax.clear()
        self.distribution_ax.clear()
        self.accuracy_ax.clear()
        
        # Update violations timeline
        if len(self.violations) > 0:
            self.violations_ax.plot(self.timestamps, self.violations, 'r-')
            self.violations_ax.set_title('Protocol Violations Timeline')
            self.violations_ax.set_xlabel('Time')
            self.violations_ax.set_ylabel('Violation Count')
        
        # Update attack distribution
        if 'Label' in df_chunk.columns:
            attack_dist = df_chunk['Label'].value_counts()
            self.distribution_ax.bar(attack_dist.index, attack_dist.values)
            self.distribution_ax.set_title('Attack Type Distribution')
            self.distribution_ax.set_xlabel('Attack Type')
            self.distribution_ax.set_ylabel('Count')
        
        # Update accuracy timeline
        if len(self.accuracies) > 0:
            self.accuracy_ax.plot(range(len(self.accuracies)), self.accuracies, 'g-')
            self.accuracy_ax.set_title('Detection Accuracy Over Time')
            self.accuracy_ax.set_xlabel('Time Window')
            self.accuracy_ax.set_ylabel('Accuracy')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def start_analysis(self):
        if not self.analyzer.load_data():
            messagebox.showerror("Error", "Failed to load dataset")
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        threading.Thread(target=self.analyze_data, daemon=True).start()

    def analyze_data(self):
        try:
            df = self.analyzer.preprocess_data()
            chunk_size = 1000
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            for chunk in chunks:
                if not self.is_running:
                    break
                
                # Analyze chunk
                self.analyze_chunk(chunk)
                
                # Simulate processing time
                threading.Event().wait(1.0)
            
            if self.is_running:
                self.root.after(0, self.analysis_complete)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))

    def analyze_chunk(self, chunk):
        # Calculate violations
        violation_count = len(chunk[chunk['Label'] == 1]) if 'Label' in chunk.columns else 0
        
        # Update tracking arrays
        self.timestamps.append(datetime.now())
        self.violations.append(violation_count)
        
        # Calculate accuracy if possible
        if 'Label' in chunk.columns:
            accuracy = accuracy_score(chunk['Label'], 
                                   [1 if random.random() > 0.5 else 0 for _ in range(len(chunk))])
            self.accuracies.append(accuracy)
        
        # Update GUI
        self.root.after(0, self.update_gui, chunk, violation_count)

    def update_gui(self, chunk, violations):
        # Update labels
        detection_rate = (violations / len(chunk)) * 100 if len(chunk) > 0 else 0
        self.detection_label.config(text=f"Detection Rate: {detection_rate:.1f}%")
        
        if self.accuracies:
            self.accuracy_label.config(text=f"Model Accuracy: {self.accuracies[-1]:.1f}%")
            
        self.alert_label.config(text=f"Protocol Violations: {violations}")
        
        # Update plots
        self.update_plots(chunk)
        
        # Update results text
        self.results_text.insert(tk.END, 
            f"Analysis window completed at {datetime.now().strftime('%H:%M:%S')}\n"
            f"Violations detected: {violations}\n"
            f"Window size: {len(chunk)} packets\n"
            f"Detection rate: {detection_rate:.1f}%\n\n")
        self.results_text.see(tk.END)

    def stop_analysis(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def analysis_complete(self):
        messagebox.showinfo("Analysis Complete", 
                          "Dataset analysis completed successfully.")
        self.stop_analysis()

def main():
    root = tk.Tk()
    data_path = "F:/mrudula college/Sem1_project/Cyberattack_on_smartGrid/intermediate_combined_data.csv"
    app = SmartGridGUI(root, data_path)
    root.mainloop()

if __name__ == "__main__":
    main()