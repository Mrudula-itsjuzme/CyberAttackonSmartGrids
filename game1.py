import tkinter as tk
from tkinter import messagebox, ttk, font
import pandas as pd
import random
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import math

class WelcomeScreen:
    def __init__(self, root, start_game_callback):
        self.root = root
        self.start_game_callback = start_game_callback
        
        # Create gradient background
        self.canvas = tk.Canvas(root, width=900, height=700, bg='#1a1a1a')
        self.canvas.pack(fill="both", expand=True)
        
        # Title with cybersecurity-themed font
        title_font = font.Font(family="Helvetica", size=36, weight="bold")
        self.canvas.create_text(450, 200, text="Smart Grid\nCybersecurity", 
                              fill="#00ff00", font=title_font, justify="center")
        
        # Animated subtitle
        self.subtitle_pos = 0
        self.subtitle_text = "Defend the Grid. Save the Power."
        self.animate_subtitle()
        
        # Styled buttons
        self.create_styled_button("Start Game", 450, 400, self.start_game_callback)
        self.create_styled_button("Help", 450, 470, self.show_help)
        self.create_styled_button("Exit", 450, 540, root.quit)

    def create_styled_button(self, text, x, y, command):
        button = tk.Button(self.root, text=text, command=command,
                          bg='#2d2d2d', fg='#00ff00',
                          font=("Helvetica", 14),
                          relief="flat",
                          width=20,
                          activebackground='#404040')
        self.canvas.create_window(x, y, window=button)
        
        # Hover effect
        button.bind('<Enter>', lambda e: button.config(bg='#404040'))
        button.bind('<Leave>', lambda e: button.config(bg='#2d2d2d'))

    def animate_subtitle(self):
        if self.subtitle_pos < len(self.subtitle_text):
            self.canvas.delete("subtitle")
            self.canvas.create_text(450, 300, 
                                  text=self.subtitle_text[:self.subtitle_pos+1],
                                  fill="#00ff00", 
                                  font=("Helvetica", 18),
                                  tags="subtitle")
            self.subtitle_pos += 1
            self.root.after(50, self.animate_subtitle)

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Game Help")
        help_window.geometry("600x400")
        
        help_text = """
        🎮 Game Controls:
        
        • Load Dataset: Initialize the game with cybersecurity data
        • Simulate Attack: Generate random cyber threats
        • Defend Grid: Deploy defensive measures
        • Visualize Data: See attack patterns and statistics
        
        🏆 Scoring:
        • Successful defense: +10 points
        • Failed defense: -5 points
        
        💡 Tips:
        • Watch for attack patterns
        • Choose appropriate defenses
        • Monitor grid stability
        """
        
        help_label = tk.Label(help_window, text=help_text, 
                            justify="left", padx=20, pady=20,
                            font=("Courier New", 12))
        help_label.pack(expand=True, fill="both")

class AnimatedAttack:
    def __init__(self, canvas, attack_type):
        self.canvas = canvas
        self.attack_type = attack_type
        self.particles = []
        
        # Create attack particles
        for _ in range(20):
            x = random.randint(0, 800)
            y = -20
            self.particles.append({
                'x': x, 'y': y,
                'dx': random.uniform(-2, 2),
                'dy': random.uniform(3, 7)
            })
        
        self.animate()

    def animate(self):
        self.canvas.delete("attack_particle")
        
        still_active = False
        for p in self.particles:
            if p['y'] < 700:
                still_active = True
                p['x'] += p['dx']
                p['y'] += p['dy']
                
                color = "#ff0000" if self.attack_type == "DDoS" else "#ff00ff" if self.attack_type == "MITM" else "#ffff00"
                self.canvas.create_oval(p['x'], p['y'], p['x']+5, p['y']+5,
                                     fill=color, tags="attack_particle")
        
        if still_active:
            self.canvas.after(50, self.animate)

class SmartGridPipelineGame:
    def __init__(self, root, dataset_path):
        self.root = root
        self.root.title("Smart Grid Cybersecurity: The Game")
        self.root.geometry("900x700")
        self.dataset_path = dataset_path
        
        # Show welcome screen first
        self.welcome_screen = WelcomeScreen(root, self.start_game)

    def start_game(self):
        # Clear welcome screen
        for widget in self.root.winfo_children():
            widget.destroy()
            
        self.setup_game()

    def setup_game(self):
        # Create main game canvas
        self.canvas = tk.Canvas(self.root, width=900, height=700, bg='#1a1a1a')
        self.canvas.pack(fill="both", expand=True)
        
        # Initialize game state
        self.data = None
        self.attack_types = ["DDoS", "MITM", "Phishing"]
        self.level = 1
        self.score = 0
        self.defense_active = False
        
        # Create grid layout
        self.create_grid_layout()
        
        # Setup logging
        self.setup_logging()

    def create_grid_layout(self):
        # Status panel (Top)
        status_frame = tk.Frame(self.root, bg='#2d2d2d')
        status_frame.place(x=0, y=0, width=900, height=100)
        
        self.level_label = self.create_status_label(status_frame, f"Level: {self.level}", 0)
        self.score_label = self.create_status_label(status_frame, f"Score: {self.score}", 1)
        self.status_label = self.create_status_label(status_frame, "Status: Ready", 2)
        
        # Main game area (Middle)
        game_frame = tk.Frame(self.root, bg='#1a1a1a')
        game_frame.place(x=0, y=100, width=900, height=400)
        
        self.game_canvas = tk.Canvas(game_frame, bg='#1a1a1a', 
                                   width=900, height=400,
                                   highlightthickness=0)
        self.game_canvas.pack(fill="both", expand=True)
        
        # Draw grid infrastructure
        self.draw_grid_infrastructure()
        
        # Control panel (Bottom)
        control_frame = tk.Frame(self.root, bg='#2d2d2d')
        control_frame.place(x=0, y=500, width=900, height=200)
        
        # Create styled buttons
        buttons = [
            ("Load Dataset", self.load_data, '#4CAF50'),
            ("Simulate Attack", self.simulate_attack, '#f44336'),
            ("Defend Grid!", self.defend_grid, '#2196F3'),
            ("Visualize Data", self.visualize_data, '#FF9800'),
            ("Help", self.show_help, '#9C27B0'),
            ("Exit Game", self.root.quit, '#607D8B')
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(control_frame, text=text, command=command,
                          bg=color, fg='white',
                          font=("Helvetica", 12, "bold"),
                          relief="flat",
                          width=15,
                          height=2)
            btn.grid(row=i//3, column=i%3, padx=10, pady=5)
            
            # Hover effect
            btn.bind('<Enter>', lambda e, b=btn, c=color: b.config(bg=self.lighten_color(c)))
            btn.bind('<Leave>', lambda e, b=btn, c=color: b.config(bg=c))

    def create_status_label(self, parent, text, column):
        label = tk.Label(parent, text=text,
                        bg='#2d2d2d', fg='#00ff00',
                        font=("Helvetica", 16, "bold"))
        label.grid(row=0, column=column, padx=20, pady=10)
        return label

    def draw_grid_infrastructure(self):
        # Draw power lines
        for i in range(5):
            y = 100 + i * 50
            self.game_canvas.create_line(50, y, 850, y,
                                       fill="#444", width=2,
                                       dash=(5, 5))
        
        # Draw power stations
        for i in range(3):
            x = 200 + i * 250
            self.draw_power_station(x, 200)

    def draw_power_station(self, x, y):
        size = 40
        self.game_canvas.create_rectangle(x-size/2, y-size/2,
                                        x+size/2, y+size/2,
                                        fill="#666", outline="#888")
        self.game_canvas.create_text(x, y, text="⚡",
                                   font=("Helvetica", 20),
                                   fill="#ffff00")

    def lighten_color(self, color):
        # Convert color to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        # Lighten
        factor = 1.2
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"

    def simulate_attack(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return
        
        if self.defense_active:
            return
            
        attack_type = random.choice(self.attack_types)
        
        # Update status
        self.status_label.config(text=f"Status: {attack_type} Attack Incoming!")
        
        # Create attack animation
        self.current_attack = AnimatedAttack(self.game_canvas, attack_type)
        
        # Show defense options after delay
        self.root.after(2000, lambda: self.show_defense_options(attack_type))

    def show_defense_options(self, attack_type):
        defense_window = tk.Toplevel(self.root)
        defense_window.title("Choose Defense")
        defense_window.geometry("400x300")
        
        tk.Label(defense_window, 
                text="Select Defense Strategy:",
                font=("Helvetica", 14, "bold"),
                pady=10).pack()
        
        defense_var = tk.StringVar()
        
        defenses = [
            ("DDoS Protection\n(Firewall + Load Balancer)", "DDoS"),
            ("MITM Prevention\n(Encryption + Authentication)", "MITM"),
            ("Phishing Defense\n(AI Detection + Filtering)", "Phishing")
        ]
        
        for text, value in defenses:
            tk.Radiobutton(defense_window,
                          text=text,
                          variable=defense_var,
                          value=value,
                          font=("Helvetica", 12),
                          pady=10).pack()
        
        tk.Button(defense_window,
                 text="Deploy Defense",
                 command=lambda: self.evaluate_defense(defense_var.get(), 
                                                     attack_type,
                                                     defense_window),
                 bg="#4CAF50",
                 fg="white",
                 font=("Helvetica", 12, "bold"),
                 pady=10).pack(pady=20)

    def evaluate_defense(self, defense_choice, attack_type, defense_window):
        defense_window.destroy()
        
        success = defense_choice == attack_type
        self.animate_defense(success)
        
        # Update score and level
        if success:
            self.score += 10
            self.level += 1
            self.status_label.config(text="Status: Defense Successful!")
        else:
            self.score -= 5
            self.status_label.config(text="Status: Defense Failed!")
        
        self.level_label.config(text=f"Level: {self.level}")
        self.score_label.config(text=f"Score: {self.score}")

    def animate_defense(self, success):
        self.defense_active = True
        
        # Create defense shield animation
        shield_color = "#00ff00" if success else "#ff0000"
        shield = self.game_canvas.create_arc(100, 100, 800, 400,
                                           start=0, extent=180,
                                           fill=shield_color,
                                           stipple="gray50")
        
        def fade_shield(alpha):
            if alpha > 0:
                self.game_canvas.itemconfig(shield, 
                                          stipple=f"gray{int(alpha/255*100)}")
                self.root.after(50, lambda: fade_shield(alpha-10))
            else:
                self.game_canvas.delete(shield)
                self.defense_active = False
        
        self.root.after(1000, lambda: fade_shield(255))

    def setup_logging(self):
        self.output_dir = f"smart_grid_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        log_file = os.path.join(self.output_dir, "game.log")
        logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(message)s",
                          handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    def load_data(self):
        try:
            self.data = pd.read_csv(self.dataset_path)
            logging.info(f"Dataset loaded successfully from: {self.dataset_path}")
            self.status_label.config(text="Status: Dataset Loaded!")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            messagebox.showerror("Error", "Failed to load dataset.")

    def defend_grid(self):
        if not self.defense_active:
            self.status_label.config(text="Status: Grid Defense Activated")
            self.animate_defense(True)

    def visualize_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return
            
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Data Visualization")
        viz_window.geometry("800x600")
        
        # Create notebook for multiple visualizations
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill="both", expand=True)
        
        # Attack patterns tab
        attack_frame = ttk.Frame(notebook)
        notebook.add(attack_frame, text="Attack Patterns")
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for col in self.data.columns[:5]:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                ax1.plot(self.data[col], label=col)
        ax1.set_title("Attack Pattern Analysis")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        
        canvas1 = FigureCanvasTkAgg(fig1, master=attack_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # Distribution tab
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Attack Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        numeric_cols = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]
        if numeric_cols:
            self.data[numeric_cols[0]].hist(ax=ax2, bins=30)
        ax2.set_title("Attack Intensity Distribution")
        
        canvas2 = FigureCanvasTkAgg(fig2, master=dist_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Game Help & Tutorial")
        help_window.geometry("600x700")
        
        # Create tabs for different help topics
        notebook = ttk.Notebook(help_window)
        notebook.pack(fill="both", expand=True)
        
        # Basic Controls
        controls_frame = ttk.Frame(notebook)
        notebook.add(controls_frame, text="Controls")
        
        controls_text = """
        🎮 Basic Controls
        
        • Load Dataset: Initialize game with cybersecurity data
        • Simulate Attack: Generate random cyber threats
        • Defend Grid: Deploy defensive measures
        • Visualize Data: View attack patterns
        
        ⚡ Power Grid Elements
        • Power Lines: Horizontal lines showing grid connectivity
        • Power Stations: Building icons showing major nodes
        • Defense Shield: Appears during defense activation
        """
        
        tk.Label(controls_frame, text=controls_text,
                justify="left", padx=20, pady=20,
                font=("Courier New", 12)).pack()
        
        # Scoring System
        scoring_frame = ttk.Frame(notebook)
        notebook.add(scoring_frame, text="Scoring")
        
        scoring_text = """
        🏆 Scoring System
        
        Success Rewards:
        • Successful defense: +10 points
        • Level completion: +5 points
        • Perfect defense streak: +15 bonus
        
        Penalties:
        • Failed defense: -5 points
        • Missed attack: -3 points
        • Grid damage: -2 points per attack
        
        📈 Level Progression:
        • Attacks become more frequent
        • Multiple simultaneous threats
        • New attack patterns emerge
        • Faster response required
        """
        
        tk.Label(scoring_frame, text=scoring_text,
                justify="left", padx=20, pady=20,
                font=("Courier New", 12)).pack()
        
        # Strategy Guide
        strategy_frame = ttk.Frame(notebook)
        notebook.add(strategy_frame, text="Strategy")
        
        strategy_text = """
        💡 Strategy Tips
        
        1. Attack Recognition:
        • DDoS: Multiple rapid connections
        • MITM: Unusual traffic patterns
        • Phishing: Suspicious data requests
        
        2. Defense Tactics:
        • Monitor power flow patterns
        • Deploy preemptive defenses
        • Maintain resource balance
        
        3. Advanced Techniques:
        • Chain multiple defenses
        • Predict attack patterns
        • Optimize grid stability
        """
        
        tk.Label(strategy_frame, text=strategy_text,
                justify="left", padx=20, pady=20,
                font=("Courier New", 12)).pack()

def main():
    root = tk.Tk()
    root.title("Smart Grid Cybersecurity: The Game")
    
    # Set dark theme
    style = ttk.Style()
    style.theme_use('alt')
    style.configure("TNotebook", background='#1a1a1a')
    style.configure("TFrame", background='#1a1a1a')
    
    dataset_path = r"G:\Sem 1\Cyberattack_on_smartGrid\intermediate_combined_data.csv" # Update with actual path
    game = SmartGridPipelineGame(root, dataset_path)
    
    root.mainloop()

if __name__ == "__main__":
    main()