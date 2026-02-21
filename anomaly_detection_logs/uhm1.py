import tkinter as tk
from tkinter import messagebox
import pandas as pd
import random
import os
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import threading
import numpy as np

class SmartGridCyberGame:
    def __init__(self, root, dataset_path):
        self.root = root
        self.root.title("Smart Grid Cybersecurity Simulator")
        self.root.geometry("900x700")
        self.dataset_path = dataset_path
        self.data = None
        self.attack_log = []
        self.output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_logging()

        # Initialize game state
        self.level = 1
        self.score = 0
        self.honeypot_triggered = False
        self.feature_names = []

        # Create UI elements
        self.create_widgets()

    def setup_logging(self):
        log_file = os.path.join(self.output_dir, "game.log")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])

    def create_widgets(self):
        # Logs Display
        self.log_box = tk.Text(self.root, height=15, width=80, bg="black", fg="white", font=("Courier New", 10))
        self.log_box.pack(pady=10)

        # Buttons
        self.load_button = tk.Button(self.root, text="Load Dataset", command=self.load_data, bg="lightblue", font=("Helvetica", 12))
        self.load_button.pack(pady=5)

        self.simulate_attack_button = tk.Button(self.root, text="Simulate Attack", command=self.simulate_attack, bg="orange", font=("Helvetica", 12))
        self.simulate_attack_button.pack(pady=5)

        self.run_ids_button = tk.Button(self.root, text="Run IDS", command=self.run_ids, bg="green", font=("Helvetica", 12))
        self.run_ids_button.pack(pady=5)

        self.deploy_honeypot_button = tk.Button(self.root, text="Deploy Honeypot", command=self.deploy_honeypot, bg="purple", font=("Helvetica", 12))
        self.deploy_honeypot_button.pack(pady=5)

        self.visualize_data_button = tk.Button(self.root, text="Visualize Data", command=self.visualize_data, bg="lightgreen", font=("Helvetica", 12))
        self.visualize_data_button.pack(pady=5)

        self.quit_button = tk.Button(self.root, text="Exit Game", command=self.root.quit, bg="lightgray", font=("Helvetica", 12))
        self.quit_button.pack(pady=20)

        # Labels
        self.level_label = tk.Label(self.root, text=f"Level: {self.level}", font=("Helvetica", 16), fg="blue")
        self.level_label.pack(pady=5)

        self.score_label = tk.Label(self.root, text=f"Score: {self.score}", font=("Helvetica", 16), fg="green")
        self.score_label.pack(pady=5)

    def load_data(self):
        try:
            self.data = pd.read_csv(self.dataset_path)
            self.feature_names = self.data.columns.tolist()
            self.log_box.insert(tk.END, "Dataset loaded successfully!\n")
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", "Failed to load dataset.")
            logging.error(f"Failed to load dataset: {e}")

    def simulate_attack(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded.")
            return

        attack_type = random.choice(["DDoS", "Phishing", "MITM"])
        sampled_row = self.data.sample(1).iloc[0]
        features = ', '.join([f"{col}: {sampled_row[col]}" for col in random.sample(self.feature_names, min(3, len(self.feature_names)))])

        attack_details = f"[ATTACK] {attack_type} attack detected! Features: {features}\n"
        self.attack_log.append(attack_details)
        self.log_box.insert(tk.END, attack_details)
        logging.info(attack_details)

        self.level += 1
        self.score += 10
        self.update_game_state()

    def run_ids(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded.")
            return

        self.log_box.insert(tk.END, "Running Intrusion Detection System (IDS)...\n")
        self.root.update()

        # Basic IDS using Isolation Forest
        clf = IsolationForest(random_state=42, contamination=0.1)
        try:
            X = self.data.select_dtypes(include=[np.number]).dropna()
            labels = clf.fit_predict(X)
            self.data['Anomaly'] = labels
            
            # Confusion Matrix (assuming ground truth available as 'Label')
            if 'Label' in self.data.columns:
                cm = confusion_matrix(self.data['Label'], self.data['Anomaly'])
                plt.matshow(cm, cmap='coolwarm')
                plt.title("Confusion Matrix")
                plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
                plt.show()
                logging.info("Confusion matrix saved.")

            self.log_box.insert(tk.END, "IDS completed. Anomaly column added.\n")
        except Exception as e:
            logging.error(f"IDS failed: {e}")

    def deploy_honeypot(self):
        if self.honeypot_triggered:
            self.log_box.insert(tk.END, "Honeypot already deployed!\n")
            return

        self.log_box.insert(tk.END, "Deploying honeypot...\n")
        self.root.update()

        self.honeypot_triggered = True
        self.log_box.insert(tk.END, "Honeypot triggered! Attack attempts being logged.\n")
        logging.info("Honeypot deployed and triggered.")
        
    def visualize_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded.")
            return

        sampled_feature = random.choice(self.feature_names)
        self.data[sampled_feature].value_counts().plot(kind='bar', title=f"Feature: {sampled_feature}")
        plt.savefig(os.path.join(self.output_dir, f"{sampled_feature}_visualization.png"))
        plt.show()
        logging.info(f"Visualization for {sampled_feature} saved.")

    def update_game_state(self):
        self.level_label.config(text=f"Level: {self.level}")
        self.score_label.config(text=f"Score: {self.score}")
        self.root.update()

# Main program
if __name__ == "__main__":
    root = tk.Tk()
    dataset_path = r"D:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"  # Update this with your dataset path
    game = SmartGridCyberGame(root, dataset_path)
    root.mainloop()
