import tkinter as tk
from tkinter import messagebox, ttk, font
import pandas as pd
import random
import os
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import math
import json

class GameState:
    def __init__(self):
        self.achievements = set()
        self.high_score = 0
        self.tutorial_completed = False
        self.current_level = 1
        self.current_score = 0
        self.learning_progress = {}
        
    def save_game(self):
        save_data = {
            'achievements': list(self.achievements),
            'high_score': self.high_score,
            'tutorial_completed': self.tutorial_completed,
            'current_level': self.current_level,
            'current_score': self.current_score,
            'learning_progress': self.learning_progress
        }
        with open('game_save.json', 'w') as f:
            json.dump(save_data, f)
            
    def load_game(self):
        try:
            with open('game_save.json', 'r') as f:
                save_data = json.load(f)
                self.achievements = set(save_data['achievements'])
                self.high_score = save_data['high_score']
                self.tutorial_completed = save_data['tutorial_completed']
                self.current_level = save_data['current_level']
                self.current_score = save_data['current_score']
                self.learning_progress = save_data.get('learning_progress', {})
                return True
        except FileNotFoundError:
            return False

class Tutorial:
    def __init__(self, parent):
        self.parent = parent
        self.current_step = 0
        self.steps = [
            "Welcome to Smart Grid Defense! Let's learn the basics.",
            "First, load your dataset using the 'Load Dataset' button.",
            "Watch for incoming attacks - they'll appear as colored particles.",
            "Choose the correct defense strategy to protect your grid.",
            "Monitor your score and level progress at the top.",
            "Use the visualization tools to analyze attack patterns.",
            "Complete challenges to unlock achievements and progress."
        ]
        
    def start(self):
        self.show_step()
        
    def show_step(self):
        if self.current_step < len(self.steps):
            tutorial_window = tk.Toplevel(self.parent)
            tutorial_window.title(f"Tutorial Step {self.current_step + 1}")
            
            frame = tk.Frame(tutorial_window, bg='#2d2d2d', padx=20, pady=20)
            frame.pack(fill="both", expand=True)
            
            tk.Label(frame, 
                    text=self.steps[self.current_step],
                    padx=20, pady=20,
                    font=("Helvetica", 12),
                    bg='#2d2d2d',
                    fg='#00ff00').pack()
                    
            btn = tk.Button(frame,
                          text="Next" if self.current_step < len(self.steps)-1 else "Finish",
                          command=lambda: self.next_step(tutorial_window),
                          bg='#4CAF50',
                          fg='white',
                          font=("Helvetica", 12, "bold"))
            btn.pack(pady=10)
                     
    def next_step(self, window):
        window.destroy()
        self.current_step += 1
        if self.current_step < len(self.steps):
            self.show_step()

class WelcomeScreen:
    def __init__(self, root, start_game_callback):
        self.root = root
        self.start_game_callback = start_game_callback
        self.game_state = GameState()
        
        self.canvas = tk.Canvas(root, width=900, height=700, bg='#1a1a1a')
        self.canvas.pack(fill="both", expand=True)
        
        title_font = font.Font(family="Helvetica", size=36, weight="bold")
        self.canvas.create_text(450, 200, text="Smart Grid\nCybersecurity", 
                              fill="#00ff00", font=title_font, justify="center")
        
        self.subtitle_pos = 0
        self.subtitle_text = "Defend the Grid. Save the Power."
        self.animate_subtitle()
        
        if self.game_state.load_game():
            self.create_styled_button("Continue Game", 450, 350, self.continue_game)
            
        self.create_styled_button("New Game", 450, 420, self.start_game_callback)
        self.create_styled_button("Tutorial", 450, 490, self.start_tutorial)
        self.create_styled_button("Learning Mode", 450, 560, self.show_learning_mode)
        self.create_styled_button("Exit", 450, 630, root.quit)

    def continue_game(self):
        self.start_game_callback(load_saved=True)
        
    def start_tutorial(self):
        Tutorial(self.root).start()
        
    def show_learning_mode(self):
        LearningMode(self.root, self.game_state).show_topic('ddos')

    def create_styled_button(self, text, x, y, command):
        button = tk.Button(self.root, text=text, command=command,
                          bg='#2d2d2d', fg='#00ff00',
                          font=("Helvetica", 14),
                          relief="flat",
                          width=20,
                          activebackground='#404040')
        self.canvas.create_window(x, y, window=button)
        
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

class GameState:
    def __init__(self):
        self.achievements = set()
        self.high_score = 0
        self.tutorial_completed = False
        self.current_level = 1
        self.current_score = 0
        self.learning_progress = {}
        
    def save_game(self):
        save_data = {
            'achievements': list(self.achievements),
            'high_score': self.high_score,
            'tutorial_completed': self.tutorial_completed,
            'current_level': self.current_level,
            'current_score': self.current_score,
            'learning_progress': self.learning_progress
        }
        with open('game_save.json', 'w') as f:
            json.dump(save_data, f)
            
    def load_game(self):
        try:
            with open('game_save.json', 'r') as f:
                save_data = json.load(f)
                self.achievements = set(save_data['achievements'])
                self.high_score = save_data['high_score']
                self.tutorial_completed = save_data['tutorial_completed']
                self.current_level = save_data['current_level']
                self.current_score = save_data['current_score']
                self.learning_progress = save_data.get('learning_progress', {})
                return True
        except FileNotFoundError:
            return False

class Tutorial:
    def __init__(self, parent):
        self.parent = parent
        self.current_step = 0
        self.steps = [
            "Welcome to Cyber Defense! Let's learn the basics.",
            "Watch for incoming attacks - they'll appear as red particles.",
            "Click to defend against attacks.",
            "Monitor your score and grid health.",
            "Complete challenges to unlock achievements."
        ]
        
    def start(self):
        self.show_step()
        
    def show_step(self):
        if self.current_step < len(self.steps):
            tutorial_window = tk.Toplevel(self.parent)
            tutorial_window.title(f"Tutorial Step {self.current_step + 1}")
            
            frame = tk.Frame(tutorial_window, bg='#2d2d2d', padx=20, pady=20)
            frame.pack(fill="both", expand=True)
            
            tk.Label(frame, 
                    text=self.steps[self.current_step],
                    padx=20, pady=20,
                    font=("Helvetica", 12),
                    bg='#2d2d2d',
                    fg='#00ff00').pack()
                    
            btn = tk.Button(frame,
                          text="Next" if self.current_step < len(self.steps)-1 else "Finish",
                          command=lambda: self.next_step(tutorial_window),
                          bg='#4CAF50',
                          fg='white',
                          font=("Helvetica", 12, "bold"))
            btn.pack(pady=10)
                     
    def next_step(self, window):
        window.destroy()
        self.current_step += 1
        if self.current_step < len(self.steps):
            self.show_step()

class WelcomeScreen:
    def __init__(self, root, start_game_callback):
        self.root = root
        self.start_game_callback = start_game_callback
        self.game_state = GameState()
        
        self.canvas = tk.Canvas(root, width=900, height=600, bg='#1a1a1a')
        self.canvas.pack(fill="both", expand=True)
        
        title_font = font.Font(family="Helvetica", size=36, weight="bold")
        self.canvas.create_text(450, 200, text="Cyber Defense", 
                              fill="#00ff00", font=title_font, justify="center")
        
        self.subtitle_pos = 0
        self.subtitle_text = "Defend the Grid. Stop the Attacks."
        self.animate_subtitle()
        
        if self.game_state.load_game():
            self.create_styled_button("Continue Game", 450, 350, self.continue_game)
            
        self.create_styled_button("New Game", 450, 420, self.start_game_callback)
        self.create_styled_button("Tutorial", 450, 490, self.start_tutorial)
        self.create_styled_button("Learning Mode", 450, 560, lambda: self.show_learning_mode('ddos'))
        
    def continue_game(self):
        self.start_game_callback(load_saved=True)
        
    def start_tutorial(self):
        Tutorial(self.root).start()
        
    def show_learning_mode(self, topic):
        learning_mode = LearningMode(self.root, self)
        learning_mode.show_topic(topic)

    def create_styled_button(self, text, x, y, command):
        button = tk.Button(self.root, text=text, command=command,
                          bg='#2d2d2d', fg='#00ff00',
                          font=("Helvetica", 14),
                          relief="flat",
                          width=20,
                          activebackground='#404040')
        self.canvas.create_window(x, y, window=button)
        
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

class AnimatedAttack:
    def __init__(self, canvas, x, y, target_x, target_y):
        self.canvas = canvas
        self.particles = []
        self.create_particles(x, y, target_x, target_y)
        
    def create_particles(self, x, y, target_x, target_y):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'x': x,
                'y': y,
                'dx': (target_x - x) / 100 + random.uniform(-0.5, 0.5),
                'dy': (target_y - y) / 100 + random.uniform(-0.5, 0.5),
                'life': 1.0
            })
            
    def animate(self):
        self.canvas.delete("attack_particle")
        
        still_active = False
        for p in self.particles:
            if p['life'] > 0:
                still_active = True
                p['x'] += p['dx']
                p['y'] += p['dy']
                p['life'] -= 0.02
                
                size = 5 * p['life']
                alpha = int(p['life'] * 255)
                
                self.canvas.create_oval(
                    p['x'] - size, p['y'] - size,
                    p['x'] + size, p['y'] + size,
                    fill="#ff0000", tags="attack_particle"
                )
        
        if still_active:
            self.canvas.after(20, self.animate)

# Update CyberDefenseGame class to include new features
class CyberDefenseGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyber Defense Game")
        self.root.geometry("900x600")
        
        self.game_state = GameState()
        self.welcome_screen = WelcomeScreen(self.root, self.start_game)
        
        self.setup_logging()
        
    def start_game(self, load_saved=False):
        if load_saved:
            self.game_state.load_game()
            
        for widget in self.root.winfo_children():
            widget.destroy()
            
        self.setup_ui()
        self.setup_game_state()
        
        if not self.game_state.tutorial_completed:
            Tutorial(self.root).start()
            self.game_state.tutorial_completed = True
            
    def setup_logging(self):
        self.output_dir = f"cyber_defense_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        log_file = os.path.join(self.output_dir, "game.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def visualize_data(self):
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Attack Analysis")
        viz_window.geometry("800x600")
        
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill="both", expand=True)
        
        # Create tabs for different visualizations
        self.create_attack_patterns_tab(notebook)
        self.create_distribution_tab(notebook)
        
    def create_attack_patterns_tab(self, notebook):
        attack_frame = ttk.Frame(notebook)
        notebook.add(attack_frame, text="Attack Patterns")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(self.active_attacks)), [1] * len(self.active_attacks), 'r.')
        ax.set_title("Attack Timeline")
        ax.set_xlabel("Time")
        ax.set_ylabel("Attacks")
        
        canvas = FigureCanvasTkAgg(fig, master=attack_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def create_distribution_tab(self, notebook):
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Success Rate")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Successful Defense', 'Failed Defense']
        sizes = [self.score, max(0, len(self.active_attacks) - self.score)]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.set_title("Defense Success Rate")
        
        canvas = FigureCanvasTkAgg(fig, master=dist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            'button_click': pygame.mixer.Sound('sounds/click.wav'),
            'attack': pygame.mixer.Sound('sounds/attack.wav'),
            'defense': pygame.mixer.Sound('sounds/defense.wav'),
            'achievement': pygame.mixer.Sound('sounds/achievement.wav'),
            'game_over': pygame.mixer.Sound('sounds/game_over.wav')
        }
        
    def play(self, sound_name):
        try:
            self.sounds[sound_name].play()
        except KeyError:
            pass

class DefenseAnimation:
    def __init__(self, canvas, success=True):
        self.canvas = canvas
        self.success = success
        self.particles = []
        self.create_particles()
        
    def create_particles(self):
        color = "#00ff00" if self.success else "#ff0000"
        center_x, center_y = 450, 250
        
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'color': color,
                'size': random.uniform(3, 8),
                'life': 1.0
            })
            
    def animate(self):
        self.canvas.delete("defense_particle")
        
        still_active = False
        for p in self.particles:
            if p['life'] > 0:
                still_active = True
                p['x'] += p['dx']
                p['y'] += p['dy']
                p['life'] -= 0.02
                
                size = p['size'] * p['life']
                alpha = int(p['life'] * 255)
                color = p['color']
                
                self.canvas.create_oval(
                    p['x'] - size, p['y'] - size,
                    p['x'] + size, p['y'] + size,
                    fill=color, tags="defense_particle"
                )
        
        if still_active:
            self.canvas.after(20, self.animate)

class GridHealth:
    def __init__(self, canvas):
        self.canvas = canvas
        self.max_health = 100
        self.current_health = 100
        self.draw_health_bar()
        
    def draw_health_bar(self):
        self.canvas.delete("health_bar")
        width = (self.current_health / self.max_health) * 200
        
        # Background
        self.canvas.create_rectangle(20, 20, 220, 40,
                                   fill="#333333",
                                   tags="health_bar")
        
        # Health level
        color = "#00ff00" if self.current_health > 70 else "#ffff00" if self.current_health > 30 else "#ff0000"
        self.canvas.create_rectangle(20, 20, 20 + width, 40,
                                   fill=color,
                                   tags="health_bar")
        
        # Text
        self.canvas.create_text(120, 30,
                              text=f"Grid Health: {int(self.current_health)}%",
                              fill="white",
                              tags="health_bar")
                              
    def damage(self, amount):
        self.current_health = max(0, self.current_health - amount)
        self.draw_health_bar()
        return self.current_health > 0
        
    def heal(self, amount):
        self.current_health = min(self.max_health, self.current_health + amount)
        self.draw_health_bar()
class DifficultyManager:
    def __init__(self):
        self.difficulties = {
            'easy': {'attack_speed': 0.7, 'damage': 5, 'points': 1.0},
            'medium': {'attack_speed': 1.0, 'damage': 10, 'points': 1.5},
            'hard': {'attack_speed': 1.3, 'damage': 15, 'points': 2.0}
        }
        self.current = 'medium'
        
    def get_settings(self):
        return self.difficulties[self.current]
        
    def set_difficulty(self, level):
        if level in self.difficulties:
            self.current = level

class LearningMode:
    def __init__(self, parent, game_state):
        self.parent = parent
        self.game_state = game_state
        self.topics = {
            'ddos': {
                'title': 'DoS Attacks',
                'content': '''Distributed Denial of Service (DoS) attacks are sophisticated 
                cyber threats that overwhelm systems by flooding them with traffic from multiple sources.
                Key characteristics include:
                - Multiple attacking machines
                - Traffic amplification techniques
                - Various attack vectors (UDP, TCP, HTTP floods)
                - Botnet utilization''',
                'quiz': [
                    {
                'q': 'What characterizes a DDoS attack?',
                'options': ['Multiple sources', 'Single source', 'Data encryption', 'System upgrade'],
                'correct': 0,
                'explanation': 'DDoS attacks use multiple compromised systems to flood the target.'
            },
            {
                'q': 'Which defense works best against DDoS?',
                'options': ['Traffic filtering', 'Password change', 'System reboot', 'Data backup'],
                'correct': 0,
                'explanation': 'Traffic filtering helps identify and block malicious traffic patterns.'
            },
            {
                'q': 'What is a botnet in DDoS attacks?',
                'options': [
                    'Network of compromised computers',
                    'Security software',
                    'Network cable',
                    'Backup server'
                ],
                'correct': 0,
                'explanation': 'Botnets are networks of compromised devices controlled by attackers.'
            },
            {
                'q': 'How do DDoS attacks affect systems?',
                'options': [
                    'Exhaust resources',
                    'Encrypt files',
                    'Install software',
                    'Update systems'
                ],
                'correct': 0,
                'explanation': 'DDoS attacks overwhelm system resources, making them unavailable.'
            },
            {
                'q': 'Which is a sign of DDoS attack?',
                'options': [
                    'Unusual traffic spikes',
                    'Slow internet',
                    'Blue screen',
                    'System updates'
                ],
                'correct': 0,
                'explanation': 'Sudden, unusual traffic spikes often indicate DDoS attacks.'
            }
                ]
            },
            'mitm': {
                'title': 'Man in the Middle Attacks',
                'content': '''MITM attacks involve intercepting communications between two parties.
                Critical aspects include:
                - Traffic interception techniques
                - SSL/TLS stripping
                - ARP spoofing
                - Session hijacking methods''',
                'quiz': [
                    {
                'q': 'What is the primary goal of MITM attacks?',
                'options': [
                    'Intercept communications',
                    'Delete data',
                    'Crash systems',
                    'Send spam'
                ],
                'correct': 0,
                'explanation': 'MITM attacks aim to intercept and monitor communications.'
            },
            {
                'q': 'Which protocol helps prevent MITM attacks?',
                'options': [
                    'HTTPS',
                    'HTTP',
                    'FTP',
                    'SMTP'
                ],
                'correct': 0,
                'explanation': 'HTTPS provides encryption and authentication to prevent MITM attacks.'
            },
            {
                'q': 'What is SSL stripping?',
                'options': [
                    'Downgrading HTTPS to HTTP',
                    'Adding encryption',
                    'Blocking traffic',
                    'System cleanup'
                ],
                'correct': 0,
                'explanation': 'SSL stripping forces secure connections to downgrade to insecure ones.'
            },
            {
                'q': 'How does ARP spoofing work?',
                'options': [
                    'Falsifies MAC addresses',
                    'Blocks ports',
                    'Deletes files',
                    'Updates software'
                ],
                'correct': 0,
                'explanation': 'ARP spoofing involves sending falsified MAC addresses to redirect traffic.'
            },
            {
                'q': 'Which is NOT a MITM defense?',
                'options': [
                    'Disabling encryption',
                    'Certificate validation',
                    'VPN usage',
                    'Public key verification'
                ],
                'correct': 0,
                'explanation': 'Disabling encryption makes systems vulnerable to MITM attacks.'
            }
                ]
            }
        }
        
    def show_topic(self, topic_key):
        topic = self.topics[topic_key]
        window = tk.Toplevel(self.parent)
        window.title(f"Learning Mode - {topic['title']}")
        window.geometry("600x400")
        
        frame = tk.Frame(window, bg='#2d2d2d')
        frame.pack(fill="both", expand=True)
        
        tk.Label(frame, text=topic['content'], 
                wraplength=500, pady=20, bg='#2d2d2d', fg='#00ff00').pack()
                
        tk.Button(frame, text="Take Quiz",
                 command=lambda: self.show_quiz(topic_key),
                 bg='#4CAF50', fg='white').pack()

    def show_quiz(self, topic_key):
        quiz = self.topics[topic_key]['quiz']
        quiz_window = tk.Toplevel(self.parent)
        quiz_window.title(f"Quiz: {self.topics[topic_key]['title']}")
        quiz_window.geometry("600x400")
        
        frame = tk.Frame(quiz_window, bg='#2d2d2d', padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        
        current_question = 0
        score = 0
        answer_var = tk.IntVar()
        
        def next_question():
            nonlocal current_question, score
            if answer_var.get() == quiz[current_question]['correct']:
                score += 1
            
            current_question += 1
            if current_question < len(quiz):
                show_question()
            else:
                show_results()
        
        def show_question():
            for widget in frame.winfo_children():
                widget.destroy()
                
            question = quiz[current_question]
            tk.Label(frame, 
                    text=question['q'],
                    bg='#2d2d2d',
                    fg='#00ff00',
                    font=("Helvetica", 14, "bold"),
                    wraplength=500).pack(pady=20)
                    
            for i, option in enumerate(question['options']):
                tk.Radiobutton(frame,
                             text=option,
                             variable=answer_var,
                             value=i,
                             bg='#2d2d2d',
                             fg='#ffffff',
                             selectcolor='#4CAF50',
                             font=("Helvetica", 12)).pack(pady=10)
                             
            tk.Button(frame,
                     text="Next",
                     command=next_question,
                     bg='#4CAF50',
                     fg='white',
                     font=("Helvetica", 12, "bold")).pack(pady=20)
        
        def show_results():
            for widget in frame.winfo_children():
                widget.destroy()
                
            result_text = f"Quiz Complete!\nScore: {score}/{len(quiz)}"
            tk.Label(frame,
                    text=result_text,
                    bg='#2d2d2d',
                    fg='#00ff00',
                    font=("Helvetica", 16, "bold")).pack(pady=20)
            
            self.game_state.learning_progress[topic_key] = score / len(quiz)
            self.game_state.save_game()
            
            if score == len(quiz):
                self.game_state.achievements.add('learning_master')
        
        show_question()

class AchievementSystem:
    def __init__(self, parent):
        self.parent = parent
        self.achievements = {
            'quick_defense': {'name': 'Lightning Reflexes', 'description': 'Defend within 2 seconds', 'points': 50},
            'perfect_score': {'name': 'Perfect Defense', 'description': 'Complete level without damage', 'points': 100},
            'learning_master': {'name': 'Knowledge Guardian', 'description': 'Complete all quizzes', 'points': 150},
            'survivor': {'name': 'Grid Survivor', 'description': 'Survive 10 waves', 'points': 200},
            'mastermind': {'name': 'Security Mastermind', 'description': 'Reach level 10', 'points': 300}
        }
        
    def unlock(self, achievement_id, callback=None):
        if achievement_id in self.achievements:
            achievement = self.achievements[achievement_id]
            self.show_popup(achievement)
            if callback:
                callback(achievement['points'])
    
    def show_popup(self, achievement):
        window = tk.Toplevel(self.parent)
        window.title('Achievement Unlocked!')
        window.geometry('400x200')
        
        frame = tk.Frame(window, bg='#2d2d2d', padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        # Trophy emoji animation
        trophy = tk.Label(frame, text='🏆', font=('Helvetica', 48), bg='#2d2d2d')
        trophy.pack()
        
        def bounce():
            for i in range(5):
                trophy.place(y=20 + abs(math.sin(time.time() * 5)) * 10)
                window.after(50)
        
        bounce()
        
        tk.Label(frame,
                text=f"{achievement['name']}\n{achievement['description']}\n+{achievement['points']} points",
                font=('Helvetica', 14, 'bold'),
                bg='#2d2d2d',
                fg='#00ff00').pack(pady=20)
        
        self.parent.after(3000, window.destroy)

class AnimatedAttack:
    def __init__(self, canvas, attack_type, difficulty=1.0):
        self.canvas = canvas
        self.attack_type = attack_type
        self.particles = []
        
        num_particles = int(20 * difficulty)
        speed_factor = difficulty
        
        for _ in range(num_particles):
            x = random.randint(0, 800)
            y = -20
            self.particles.append({
                'x': x, 'y': y,
                'dx': random.uniform(-2, 2) * speed_factor,
                'dy': random.uniform(3, 7) * speed_factor
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
        self.game_state = GameState()
        
        self.welcome_screen = WelcomeScreen(root, self.start_game)

    def start_game(self, load_saved=False):
        for widget in self.root.winfo_children():
            widget.destroy()
            
        if load_saved:
            self.game_state.load_game()
            
        self.setup_game()
        
        if not self.game_state.tutorial_completed:
            Tutorial(self.root).start()
            self.game_state.tutorial_completed = True

    def setup_game(self):
        self.canvas = tk.Canvas(self.root, width=900, height=700, bg='#1a1a1a')
        self.canvas.pack(fill="both", expand=True)
        
        self.data = None
        self.attack_types = ["DDoS", "MITM", "Phishing"]
        self.level = self.game_state.current_level
        self.score = self.game_state.current_score
        self.defense_active = False
        self.last_attack_time = None
        
        self.create_grid_layout()
        self.setup_logging()
        
        self.achievements = {
            'first_defense': 'First Successful Defense',
            'perfect_level': 'Perfect Level Defense',
            'high_scorer': 'High Score Champion',
            'quick_defender': 'Lightning Fast Defense',
            'learning_master': 'Completed All Learning Modules'
        }
        
        self.schedule_next_attack()

    def create_grid_layout(self):
        status_frame = tk.Frame(self.root, bg='#2d2d2d')
        status_frame.place(x=0, y=0, width=900, height=100)
        
        self.level_label = self.create_status_label(status_frame, f"Level: {self.level}", 0)
        self.score_label = self.create_status_label(status_frame, f"Score: {self.score}", 1)
        self.status_label = self.create_status_label(status_frame, "Status: Ready", 2)
        
        game_frame = tk.Frame(self.root, bg='#1a1a1a')
        game_frame.place(x=0, y=100, width=900, height=400)
        
        self.game_canvas = tk.Canvas(game_frame, bg='#1a1a1a', 
                                   width=900, height=400,
                                   highlightthickness=0)
        self.game_canvas.pack(fill="both", expand=True)
        
        self.draw_grid_infrastructure()
        
        control_frame = tk.Frame(self.root, bg='#2d2d2d')
        control_frame.place(x=0, y=500, width=900, height=200)
        
        buttons = [
            ("Load Dataset", self.load_data, '#4CAF50'),
            ("Simulate Attack", self.simulate_attack, '#f44336'),
            ("Defend Grid!", self.defend_grid, '#2196F3'),
            ("Visualize Data", self.visualize_data, '#FF9800'),
            ("Learning Mode", self.show_learning_mode, '#9C27B0'),
            ("Save Game", self.save_game, '#607D8B')
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(control_frame, text=text, command=command,
                          bg=color, fg='white',
                          font=("Helvetica", 12, "bold"),
                          relief="flat",
                          width=15,
                          height=2)
            btn.grid(row=i//3, column=i%3, padx=10, pady=5)
            
            btn.bind('<Enter>', lambda e, b=btn, c=color: b.config(bg=self.lighten_color(c)))
            btn.bind('<Leave>', lambda e, b=btn, c=color: b.config(bg=c))

    def defend_grid(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return
            
        defense_window = tk.Toplevel(self.root)
        defense_window.title("Grid Defense System")
        defense_window.geometry("500x400")
        
        defense_options = {
            "Firewall Configuration": {
                "description": "Configure firewall rules and packet filtering",
                "effectiveness": 0.8
            },
            "Load Balancing": {
                "description": "Distribute traffic across multiple servers",
                "effectiveness": 0.7
            },
            "Encryption Protocol": {
                "description": "Implement end-to-end encryption",
                "effectiveness": 0.9
            },
            "Authentication System": {
                "description": "Enhanced user verification protocols",
                "effectiveness": 0.85
            }
        }
        
        for name, details in defense_options.items():
            frame = tk.Frame(defense_window, relief=tk.RAISED, borderwidth=1)
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            tk.Label(frame, text=name, font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
            tk.Label(frame, text=details["description"]).pack(anchor=tk.W)
            tk.Button(frame, 
                     text="Deploy",
                     command=lambda n=name, e=details["effectiveness"]: 
                     self.execute_defense(n, e, defense_window)).pack(anchor=tk.E)

    def execute_defense(self, defense_name, effectiveness, window):
        success = random.random() < effectiveness
        window.destroy()
        
        if success:
            self.score += int(effectiveness * 20)
            self.status_label.config(text=f"Defense Successful: {defense_name}")
            if self.score > self.game_state.high_score:
                self.unlock_achievement('high_scorer')
        else:
            self.score -= 10
            self.status_label.config(text=f"Defense Failed: {defense_name}")
            
        self.score_label.config(text=f"Score: {self.score}")
        self.game_state.current_score = self.score
        self.game_state.save_game()

    def create_status_label(self, parent, text, column):
        label = tk.Label(parent, text=text,
                        bg='#2d2d2d', fg='#00ff00',
                        font=("Helvetica", 16, "bold"))
        label.grid(row=0, column=column, padx=20, pady=10)
        return label

    def draw_grid_infrastructure(self):
        for i in range(5):
            y = 100 + i * 50
            self.game_canvas.create_line(50, y, 850, y,
                                       fill="#444", width=2,
                                       dash=(5, 5))
        
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
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        factor = 1.2
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"

    def schedule_next_attack(self):
        delay = max(5000 - (self.level * 500), 2000)
        self.root.after(delay, self.auto_attack)

    def auto_attack(self):
        if not self.defense_active:
            self.simulate_attack()
        self.schedule_next_attack()

    def simulate_attack(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return
            
        if self.defense_active:
            return
            
        attack_type = random.choice(self.attack_types)
        self.last_attack_time = datetime.now()
        
        self.status_label.config(text=f"Status: {attack_type} Attack Incoming!")
        
        difficulty = 1.0 + (self.level - 1) * 0.2
        self.current_attack = AnimatedAttack(self.game_canvas, attack_type, difficulty)

    def unlock_achievement(self, achievement_key):
        if achievement_key not in self.game_state.achievements:
            self.game_state.achievements.add(achievement_key)
            self.show_achievement(self.achievements[achievement_key])
            self.game_state.save_game()

    def show_achievement(self, achievement_text):
        achievement_window = tk.Toplevel(self.root)
        achievement_window.title("Achievement Unlocked!")
        
        frame = tk.Frame(achievement_window, bg='#2d2d2d', padx=20, pady=20)
        frame.pack(fill="both", expand=True)
        
        tk.Label(frame,
                text=f"🏆 Achievement Unlocked!\n\n{achievement_text}",
                font=("Helvetica", 14, "bold"),
                bg='#2d2d2d',
                fg='#00ff00').pack()
                
        self.root.after(3000, achievement_window.destroy)

    def show_learning_mode(self):
        LearningMode(self.root, self.game_state).show_topic('ddos')

    def save_game(self):
        self.game_state.save_game()
        messagebox.showinfo("Success", "Game progress saved!")

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

    def visualize_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Dataset not loaded!")
            return
            
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Data Visualization")
        viz_window.geometry("800x600")
        
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill="both", expand=True)
        
        self.create_attack_patterns_tab(notebook)
        self.create_distribution_tab(notebook)

    def create_attack_patterns_tab(self, notebook):
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

    def create_distribution_tab(self, notebook):
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

class CyberDefenseGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyber Defense Game")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")
        
        self.sound_manager = SoundManager()
        self.difficulty_manager = DifficultyManager()
        
        self.score = 0
        self.level = 1
        self.wave_count = 0
        self.active_attacks = []
        self.game_running = False
        
        self.setup_ui()
        self.setup_game_state()
        
    def setup_ui(self):
        # Main game canvas
        self.canvas = tk.Canvas(self.root, width=900, height=500, bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Control panel
        control_frame = tk.Frame(self.root, bg="#1e1e1e")
        control_frame.pack(fill="x", padx=10)
        
        # Score display
        self.score_label = tk.Label(control_frame, text="Score: 0", font=("Arial", 14), bg="#1e1e1e", fg="white")
        self.score_label.pack(side="left", padx=10)
        
        # Level display
        self.level_label = tk.Label(control_frame, text="Level: 1", font=("Arial", 14), bg="#1e1e1e", fg="white")
        self.level_label.pack(side="left", padx=10)
        
        # Difficulty selector
        difficulty_frame = tk.Frame(control_frame, bg="#1e1e1e")
        difficulty_frame.pack(side="right", padx=10)
        
        tk.Label(difficulty_frame, text="Difficulty:", bg="#1e1e1e", fg="white").pack(side="left")
        self.difficulty_var = tk.StringVar(value="medium")
        for diff in ["easy", "medium", "hard"]:
            rb = tk.Radiobutton(difficulty_frame, text=diff.capitalize(), variable=self.difficulty_var,
                              value=diff, command=self.change_difficulty, bg="#1e1e1e", fg="white",
                              selectcolor="black", activebackground="#1e1e1e")
            rb.pack(side="left")
        
        # Start button
        self.start_button = tk.Button(control_frame, text="Start Game", command=self.toggle_game,
                                    bg="#4CAF50", fg="white", font=("Arial", 12))
        self.start_button.pack(side="right", padx=10)
        
    def setup_game_state(self):
        self.grid_health = GridHealth(self.canvas)
        self.achievement_system = AchievementSystem(self.root)
        self.learning_mode = LearningMode(self.root, self)
        
        # Bind defense mechanism
        self.canvas.bind("<Button-1>", self.defend)
        
    def toggle_game(self):
        if not self.game_running:
            self.start_game()
        else:
            self.end_game()
            
    def start_game(self):
        self.game_running = True
        self.start_button.configure(text="End Game", bg="#f44336")
        self.spawn_attack()
        
    def end_game(self):
        self.game_running = False
        self.start_button.configure(text="Start Game", bg="#4CAF50")
        self.show_game_over()
        
        # Clear attacks
        for attack in self.active_attacks:
            self.canvas.delete(attack['id'])
        self.active_attacks.clear()
        
    def spawn_attack(self):
        if not self.game_running:
            return
            
        settings = self.difficulty_manager.get_settings()
        
        # Create attack visual
        size = 20
        x = random.randint(size, 900 - size)
        attack = {
            'id': self.canvas.create_oval(x-size, -size, x+size, size, fill="red"),
            'y': -size,
            'speed': settings['attack_speed'],
            'damage': settings['damage'],
            'spawn_time': time.time()
        }
        self.active_attacks.append(attack)
        
        # Schedule next attack
        delay = random.randint(1000, 3000)
        self.root.after(delay, self.spawn_attack)
        self.update_attacks()
        
    def update_attacks(self):
        if not self.game_running:
            return
            
        for attack in self.active_attacks[:]:
            # Move attack down
            self.canvas.move(attack['id'], 0, attack['speed'])
            attack['y'] += attack['speed']
            
            # Check if attack reached bottom
            if attack['y'] > 500:
                self.handle_successful_attack(attack)
                
        if self.game_running:
            self.root.after(16, self.update_attacks)
            
    def handle_successful_attack(self, attack):
        self.sound_manager.play('attack')
        self.canvas.delete(attack['id'])
        self.active_attacks.remove(attack)
        
        if not self.grid_health.damage(attack['damage']):
            self.end_game()
            
    def defend(self, event):
        if not self.game_running:
            return
            
        self.sound_manager.play('button_click')
        hit = False
        
        for attack in self.active_attacks[:]:
            coords = self.canvas.coords(attack['id'])
            if coords[0] <= event.x <= coords[2] and coords[1] <= event.y <= coords[3]:
                hit = True
                self.handle_successful_defense(attack)
                break
                
        if hit:
            DefenseAnimation(self.canvas, True)
        else:
            DefenseAnimation(self.canvas, False)
            
    def handle_successful_defense(self, attack):
        self.sound_manager.play('defense')
        self.canvas.delete(attack['id'])
        self.active_attacks.remove(attack)
        
        # Calculate score based on reaction time
        reaction_time = time.time() - attack['spawn_time']
        score_multiplier = max(1, 3 - reaction_time)
        points = int(50 * score_multiplier * self.difficulty_manager.get_settings()['points'])
        
        self.add_score(points)
        
        # Check for quick defense achievement
        if reaction_time < 2.0:
            self.achievement_system.unlock('quick_defense', self.add_score)
            
        # Level progression
        self.wave_count += 1
        if self.wave_count >= 10:
            self.wave_count = 0
            self.level_up()
            
    def add_score(self, points):
        self.score += points
        self.score_label.configure(text=f"Score: {self.score}")
        
    def level_up(self):
        self.level += 1
        self.level_label.configure(text=f"Level: {self.level}")
        
        if self.level == 10:
            self.achievement_system.unlock('mastermind', self.add_score)
            
        # Increase difficulty
        settings = self.difficulty_manager.get_settings()
        settings['attack_speed'] *= 1.1
        settings['damage'] *= 1.1
        
    def change_difficulty(self):
        self.difficulty_manager.set_difficulty(self.difficulty_var.get())
        
    def show_game_over(self):
        self.sound_manager.play('game_over')
        
        window = tk.Toplevel(self.root)
        window.title('Game Over')
        window.geometry('300x200')
        
        frame = tk.Frame(window, bg='#2d2d2d', padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        tk.Label(frame,
                text=f"Game Over!\nFinal Score: {self.score}\nLevel Reached: {self.level}",
                font=('Arial', 14, 'bold'),
                bg='#2d2d2d',
                fg='white').pack(pady=20)
                
        tk.Button(frame,
                 text="Play Again",
                 command=lambda: [window.destroy(), self.reset_game()],
                 bg='#4CAF50',
                 fg='white').pack()
                 
    def reset_game(self):
        self.score = 0
        self.level = 1
        self.wave_count = 0
        self.score_label.configure(text="Score: 0")
        self.level_label.configure(text="Level: 1")
        self.grid_health.current_health = self.grid_health.max_health
        self.grid_health.draw_health_bar()
        
    def show_learning_content(self, topic):
        if topic in self.learning_mode.topics:
            content = self.learning_mode.topics[topic]
            
            window = tk.Toplevel(self.root)
            window.title(content['title'])
            window.geometry('600x400')
            
            frame = tk.Frame(window, bg='#2d2d2d', padx=20, pady=20)
            frame.pack(fill='both', expand=True)
            
            tk.Label(frame,
                    text=content['title'],
                    font=('Arial', 16, 'bold'),
                    bg='#2d2d2d',
                    fg='white').pack()
                    
            tk.Label(frame,
                    text=content['content'],
                    font=('Arial', 12),
                    bg='#2d2d2d',
                    fg='white',
                    justify='left',
                    wraplength=500).pack(pady=20)
                    
            tk.Button(frame,
                     text="Take Quiz",
                     command=lambda: self.start_quiz(topic, window),
                     bg='#4CAF50',
                     fg='white').pack()
                     
    def start_quiz(self, topic, parent_window):
        parent_window.destroy()
        
        quiz = self.learning_mode.topics[topic]['quiz']
        current_question = 0
        correct_answers = 0
        
        quiz_window = tk.Toplevel(self.root)
        quiz_window.title(f"{self.learning_mode.topics[topic]['title']} Quiz")
        quiz_window.geometry('600x400')
        
        frame = tk.Frame(quiz_window, bg='#2d2d2d', padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        question_label = tk.Label(frame,
                                text="",
                                font=('Arial', 14),
                                bg='#2d2d2d',
                                fg='white',
                                wraplength=500)
        question_label.pack(pady=20)
        
        answer_var = tk.IntVar()
        answer_buttons = []
        
        def check_answer():
            nonlocal current_question, correct_answers
            
            if answer_var.get() == quiz[current_question]['correct']:
                correct_answers += 1
                
            current_question += 1
            
            if current_question < len(quiz):
                show_question()
            else:
                show_results()
                
        def show_question():
            question = quiz[current_question]
            question_label.configure(text=question['q'])
            
            for i, option in enumerate(question['options']):
                answer_buttons[i].configure(text=option)
                
        def show_results():
            for widget in frame.winfo_children():
                widget.destroy()
                
            score = (correct_answers / len(quiz)) * 100
            
            tk.Label(frame,
                    text=f"Quiz Complete!\nScore: {score:.1f}%\nCorrect: {correct_answers}/{len(quiz)}",
                    font=('Arial', 14, 'bold'),
                    bg='#2d2d2d',
                    fg='white').pack(pady=20)
                    
            if score == 100:
                self.achievement_system.unlock('learning_master', self.add_score)
                
            tk.Button(frame,
                     text="Close",
                     command=quiz_window.destroy,
                     bg='#4CAF50',
                     fg='white').pack()
                     
        # Create answer buttons
        for i in range(4):
            btn = tk.Radiobutton(frame,
                               text="",
                               variable=answer_var,
                               value=i,
                               bg='#2d2d2d',
                               fg='white',
                               selectcolor='black',
                               font=('Arial', 12))
            btn.pack(pady=5)
            answer_buttons.append(btn)
            
        tk.Button(frame,
                 text="Submit",
                 command=check_answer,
                 bg='#4CAF50',
                 fg='white').pack(pady=20)
                 
        show_question()

def main():
    root = tk.Tk()
    root.title("Smart Grid Cybersecurity: The Game")
    
    style = ttk.Style()
    style.theme_use('alt')
    style.configure("TNotebook", background='#1a1a1a')
    style.configure("TFrame", background='#1a1a1a')
    
    dataset_path = "intermediate_combined_data.csv"
    game = SmartGridPipelineGame(root, dataset_path)
    
    root.mainloop()

if __name__ == "__main__":
    main()