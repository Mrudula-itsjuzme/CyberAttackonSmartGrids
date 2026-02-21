import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import random
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SmartGridPipelineGame:
    def __init__(self, root, dataset_path):
        self.root = root
        self.root.title("Smart Grid Cybersecurity: The Game")
        self.root.geometry("900x700")
        self.dataset_path = dataset_path
        self.data = None
        self.output_dir = f"smart_grid_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.setup_logging()

        # Initialize game state
        self.attack_types = ["DDoS", "MITM", "Phishing"]
        self.level = 1
        self.score = 0

        # Create UI elements
        self.create_widgets()

    def setup_logging(self):
        log_file = os.path.join(self.output_dir, "game.log")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    def load_data(self):
        try:
            self.progress_bar.start(10)
            self.data = pd.read_csv(self.dataset_path)
            self.feature_names = self.data.columns.tolist()
            logging.info(f"Dataset loaded successfully from: {self.dataset_path}")
            self.log_box.insert(tk.END, "Dataset loaded! Let's get started.\n")
            self.update_summary()
            self.root.update()
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            messagebox.showerror("Error", "Failed to load dataset.")
        finally:
            self.progress_bar.stop()

    def update_summary(self):
        if self.data is not None:
            summary_text = f"Dataset Summary:\nRows: {self.data.shape[0]}\nColumns: {self.data.shape[1]}\n"
            self.log_box.insert(tk.END, summary_text)

    def create_widgets(self):
        self.log_box = tk.Text(self.root, height=10, width=100, bg="black", fg="white", font=("Courier New", 10))
        self.log_box.pack(pady=10)

        self.load_button = tk.Button(self.root, text="Load Dataset", command=self.load_data, bg="lightblue", font=("Helvetica", 12))
        self.load_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="indeterminate")
        self.progress_bar.pack(pady=5)

        self.attack_button = tk.Button(self.root, text="Simulate Attack", command=self.simulate_attack, bg="lightgreen", font=("Helvetica", 12))
        self.attack_button.pack(pady=5)

        self.defense_button = tk.Button(self.root, text="Defend Grid!", command=self.defend_grid, bg="lightcoral", font=("Helvetica", 12))
        self.defense_button.pack(pady=5)

        self.level_label = tk.Label(self.root, text=f"Level: {self.level}", font=("Helvetica", 16), fg="blue")
        self.level_label.pack(pady=5)

        self.score_label = tk.Label(self.root, text=f"Score: {self.score}", font=("Helvetica", 16), fg="green")
        self.score_label.pack(pady=5)

        self.visualize_button = tk.Button(self.root, text="Visualize Data", command=self.visualize_data, bg="orange", font=("Helvetica", 12))
        self.visualize_button.pack(pady=5)

        self.quit_button = tk.Button(self.root, text="Exit Game", command=self.root.quit, bg="lightgray", font=("Helvetica", 12))
        self.quit_button.pack(pady=20)

    def simulate_attack(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return

        attack_type = random.choice(self.attack_types)
        attack_data = self.data.sample(1).iloc[0]
        attack_details = f"Detected anomaly in traffic pattern, features: {attack_data[self.feature_names[0]]}, {attack_data[self.feature_names[1]]}"

        self.log_box.insert(tk.END, f"\n[ATTACK] A {attack_type} attack is incoming!\n")
        self.log_box.insert(tk.END, f"Attack details: {attack_details}\n")
        self.root.update()

        self.show_defense_options(attack_type)

    def show_defense_options(self, attack_type):
        self.defense_choice_var = tk.StringVar()

        self.defense_radio1 = tk.Radiobutton(self.root, text="DDoS Protection", variable=self.defense_choice_var, value="DDoS", font=("Helvetica", 12))
        self.defense_radio2 = tk.Radiobutton(self.root, text="MITM Encryption", variable=self.defense_choice_var, value="MITM", font=("Helvetica", 12))
        self.defense_radio3 = tk.Radiobutton(self.root, text="Phishing Detection", variable=self.defense_choice_var, value="Phishing", font=("Helvetica", 12))

        self.defense_radio1.pack(pady=2)
        self.defense_radio2.pack(pady=2)
        self.defense_radio3.pack(pady=2)

        self.confirm_defense_button = tk.Button(self.root, text="Confirm Defense", command=lambda: self.evaluate_defense_choice(self.defense_choice_var.get(), attack_type), bg="yellow", font=("Helvetica", 12))
        self.confirm_defense_button.pack(pady=5)

        self.attack_button.config(state=tk.DISABLED)
        self.defense_button.config(state=tk.DISABLED)

    def evaluate_defense_choice(self, defense_choice, attack_type):
        success = False
        if defense_choice == 'DDoS' and attack_type == 'DDoS':
            success = True
        elif defense_choice == 'MITM' and attack_type == 'MITM':
            success = True
        elif defense_choice == 'Phishing' and attack_type == 'Phishing':
            success = True

        self.defense_success_or_failure(success)

        self.defense_radio1.pack_forget()
        self.defense_radio2.pack_forget()
        self.defense_radio3.pack_forget()
        self.confirm_defense_button.pack_forget()
        self.attack_button.config(state=tk.NORMAL)
        self.defense_button.config(state=tk.NORMAL)

    def defense_success_or_failure(self, success):
        if success:
            self.score += 10
            self.level += 1
            self.log_box.insert(tk.END, "\n[SUCCESS] You successfully defended against the attack!\n")
        else:
            self.score -= 5
            self.level -= 1
            self.log_box.insert(tk.END, "\n[FAILURE] The defense failed. The attack was successful.\n")

        self.update_game_state()

    def update_game_state(self):
        self.level_label.config(text=f"Level: {self.level}")
        self.score_label.config(text=f"Score: {self.score}")
        self.root.update()

    def visualize_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return

        plt.figure(figsize=(10, 6))
        for col in self.feature_names[:5]:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                plt.plot(self.data[col], label=col)

        plt.legend()
        plt.title("Feature Trends")
        plt.xlabel("Index")
        plt.ylabel("Values")

        fig = plt.gcf()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def defend_grid(self):
        self.log_box.insert(tk.END, "\n[INFO] Defending the grid...\n")
        self.root.update()

        defense_success = random.choice([True, False])

        if defense_success:
            self.score += 10
            self.log_box.insert(tk.END, "[SUCCESS] The defense was successful!\n")
        else:
            self.score -= 5
            self.log_box.insert(tk.END, "[FAILURE] The defense failed!\n")
        
        self.update_game_state()

# Create the main Tkinter window
root = tk.Tk()

# Path to your dataset
dataset_path = r"D:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv"

# Create the game UI
game_ui = SmartGridPipelineGame(root, dataset_path)

# Run the Tkinter main loop
root.mainloop()
