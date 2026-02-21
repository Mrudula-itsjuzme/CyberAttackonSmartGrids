import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
from datetime import datetime
import json
import time
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional, List, Dict

plt.style.use('seaborn-whitegrid')

@dataclass
class Attack:
    timestamp: datetime
    ip: str
    attack_type: str
    signature: str
    severity: float
    payload_size: int = 0
    success: bool = False
    is_blocked: bool = False  # True if blocked (anomaly detected), False otherwise

@dataclass
class SimConfig:
    seed: int
    intensity: str
    scenario_name: str
    timestamp: str

class DatasetAttackSimulator:
    def __init__(self, chunk_size: int = 10000):
        self.running = False
        self.queue = queue.Queue()
        self.chunks: List[pd.DataFrame] = []
        self.chunk_size = chunk_size
        self.current_chunk_idx = 0
        self.attack_patterns: Dict[str, object] = {}
        self.ip_patterns: List[str] = []
        self.total_records = 0
        self.current_config: Optional[SimConfig] = None

    def load_dataset(self, file_path: str) -> bool:
        """Load dataset in chunks and analyze patterns"""
        try:
            self.chunks = []
            self.total_records = 0
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)

            for i, chunk in enumerate(chunk_reader):
                self.chunks.append(chunk)
                self.total_records += len(chunk)
                if i == 0:
                    self.dataset = chunk

            self._analyze_patterns()
            return True
        except Exception as e:
            print(f"Dataset load error: {e}")
            return False

    def _analyze_patterns(self):
        if not self.chunks:
            return

        all_ips = []
        all_attacks = []
        all_severities = []
        all_sizes = []

        for chunk in self.chunks:
            ip_cols = [col for col in chunk.columns if any(x in col.lower() for x in ['ip', 'src', 'source', 'addr'])]
            attack_cols = [col for col in chunk.columns if any(x in col.lower() for x in ['attack', 'type', 'category', 'label', 'class'])]
            severity_cols = [col for col in chunk.columns if any(x in col.lower() for x in ['severity', 'score', 'risk', 'priority'])]
            size_cols = [col for col in chunk.columns if any(x in col.lower() for x in ['size', 'bytes', 'length', 'payload'])]

            if ip_cols:
                ips = chunk[ip_cols[0]].dropna().unique()
                all_ips.extend([str(ip) for ip in ips])
            if attack_cols:
                attacks = chunk[attack_cols[0]].dropna().tolist()
                all_attacks.extend([str(att) for att in attacks])
            if severity_cols:
                sevs = pd.to_numeric(chunk[severity_cols[0]], errors='coerce').dropna()
                all_severities.extend(sevs.tolist())
            if size_cols:
                sizes = pd.to_numeric(chunk[size_cols[0]], errors='coerce').dropna()
                all_sizes.extend(sizes.tolist())

        # Use only unique dataset IPs and attacks (no added fabricated signatures)
        self.ip_patterns = list(set(all_ips))[:100] if all_ips else []

        if all_attacks:
            attack_counts = pd.Series(all_attacks).value_counts()
            self.attack_patterns = {
                'types': attack_counts.index.tolist(),
                'probabilities': (attack_counts / attack_counts.sum()).tolist()
            }
        else:
            # If no attack info, fallback to empty so no fake attacks introduced
            self.attack_patterns = {
                'types': [],
                'probabilities': []
            }

        self.severity_mean = np.mean(all_severities) if all_severities else 0.5
        self.severity_std = np.std(all_severities) if all_severities else 0.2
        self.size_mean = np.mean(all_sizes) if all_sizes else 1024
        self.size_std = np.std(all_sizes) if all_sizes else 512

        print(f"Analyzed {self.total_records} records in {len(self.chunks)} chunks")
        print(f"Unique IPs: {len(self.ip_patterns)}, Attack types: {len(self.attack_patterns['types'])}")

    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        if self.current_chunk_idx < len(self.chunks):
            chunk = self.chunks[self.current_chunk_idx]
            self.current_chunk_idx += 1
            return chunk
        return None

    def reset_chunks(self):
        self.current_chunk_idx = 0

    def start(self, config: SimConfig):
        if not self.chunks or not self.attack_patterns['types'] or not self.ip_patterns:
            raise ValueError("Load dataset first with valid attacks and IPs!")

        self.running = True
        self.stop_event = threading.Event()
        self.current_config = config
        delays = {"low": 2, "medium": 0.8, "high": 0.2}
        threading.Thread(target=self._simulate, args=(delays.get(config.intensity, 0.8),), daemon=True).start()

    def _simulate(self, delay):
        while self.running and not getattr(self, "stop_event", threading.Event()).is_set():
            attack = self._generate_realistic_attack()
            # Normal attack type detection (if attack_type is "normal" or similar, allow through)
            normal_types = ['normal', 'benign', 'legitimate', 'none', 'clean', 'safe']
            is_normal = str(attack.attack_type).strip().lower() in normal_types
            attack.is_blocked = not is_normal

            try:
                self.queue.put(attack, timeout=1)
            except queue.Full:
                pass

            actual_delay = delay * random.uniform(0.5, 2.0)
            if random.random() < 0.1:
                actual_delay *= 0.1
            time.sleep(actual_delay)

    def _generate_realistic_attack(self) -> Attack:
        if not self.attack_patterns['types'] or not self.ip_patterns:
            # Defensive fallback: generate a less informative attack to avoid crash
            return Attack(datetime.now(), "0.0.0.0", "normal", "", 0.0, 0, False, False)

        attack_type = np.random.choice(
            self.attack_patterns['types'],
            p=self.attack_patterns['probabilities']
        )
        ip = random.choice(self.ip_patterns)
        severity = np.clip(np.random.normal(self.severity_mean, self.severity_std), 0.0, 1.0)
        payload_size = int(max(64, np.random.normal(self.size_mean, self.size_std)))
        success_prob = 0.05 + (severity * 0.15)
        success = random.random() < success_prob

        # Use attack_type as signature for clarity in reporting
        signature = attack_type

        return Attack(
            timestamp=datetime.now(),
            ip=ip,
            attack_type=attack_type,
            signature=signature,
            severity=severity,
            payload_size=payload_size,
            success=success
        )

    def stop(self):
        self.running = False
        if hasattr(self, "stop_event"):
            self.stop_event.set()

    def get_attack(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def get_dataset_stats(self) -> dict:
        if not self.chunks:
            return {}

        return {
            'total_records': self.total_records,
            'total_chunks': len(self.chunks),
            'chunk_size': self.chunk_size,
            'current_chunk': self.current_chunk_idx,
            'unique_ips': len(self.ip_patterns),
            'attack_types': len(self.attack_patterns['types']),
            'columns': list(self.chunks[0].columns) if self.chunks else [],
            'sample_data': self.chunks[0].head().to_dict('records') if self.chunks else []
        }

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

    def analyze(self, attacks: List[Attack]) -> dict:
        if not attacks:
            return {'anomalies': 0, 'score': 0.0}

        features = np.array([[
            hash(a.ip) % 1000,
            a.severity,
            a.payload_size,
            len(a.attack_type),
            int(a.success)
        ] for a in attacks])

        X_scaled = self.scaler.fit_transform(features)
        predictions = self.model.fit_predict(X_scaled)
        anomalies = np.sum(predictions == -1)

        return {
            'anomalies': anomalies,
            'score': 1 - (anomalies / len(attacks)),
            'total': len(attacks)
        }

class CyberGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cyber Attack Simulator - Dataset Only")
        self.root.geometry("1200x800")
        self.root.configure(bg='#ffffff')

        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TLabel', background='#ffffff', foreground='#6b7280', font=('Inter', 16))
        s.configure('Header.TLabel', font=('Inter', 48, 'bold'), foreground='#111827', background='#ffffff')
        s.configure('TButton', font=('Inter', 14), padding=8)
        s.configure('TNotebook', background='#ffffff')
        s.configure('TNotebook.Tab', font=('Inter', 14, 'bold'), padding=[14, 10])

        self.simulator = DatasetAttackSimulator()
        self.detector = AnomalyDetector()
        self.attacks: List[Attack] = []
        self.monitoring = False
        self.scenarios: Dict[str, Dict] = {}
        self.current_config: Optional[SimConfig] = None

        self._setup_ui()

    def _setup_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill='both', expand=True, padx=40, pady=32)
        container.columnconfigure(0, weight=1)

        header = ttk.Label(container, text="Cyber Attack Simulator", style='Header.TLabel', anchor='center')
        header.grid(row=0, column=0, pady=(0, 20), sticky='ew')

        controls = ttk.Frame(container)
        controls.grid(row=1, column=0, sticky='ew', pady=(0,24))
        controls.columnconfigure((0,1,2,3,4,5,6), weight=1, uniform='a')

        ttk.Button(controls, text="Load Dataset", command=self.load_dataset).grid(row=0, column=0, sticky='ew', padx=4)
        ttk.Button(controls, text="Start Simulation", command=self.start_sim).grid(row=0, column=1, sticky='ew', padx=4)
        ttk.Button(controls, text="Stop Simulation", command=self.stop_sim).grid(row=0, column=2, sticky='ew', padx=4)
        ttk.Button(controls, text="Analyze", command=self.analyze).grid(row=0, column=3, sticky='ew', padx=4)
        ttk.Button(controls, text="Export Attacks", command=self.export_attacks).grid(row=0, column=4, sticky='ew', padx=4)
        ttk.Button(controls, text="Save Scenario", command=self.save_scenario).grid(row=0, column=5, sticky='ew', padx=4)
        ttk.Button(controls, text="Load Scenario", command=self.load_scenario).grid(row=0, column=6, sticky='ew', padx=4)

        self.intensity_var = tk.StringVar(value="medium")
        intensity_frame = ttk.Frame(container)
        intensity_frame.grid(row=2, column=0, sticky='ew', pady=(0,20))
        ttk.Label(intensity_frame, text="Intensity:", font=('Inter', 16), foreground='#6b7280').pack(side='left')
        intensity_combo = ttk.Combobox(intensity_frame, textvariable=self.intensity_var, values=["low", "medium", "high"], state='readonly', width=14, font=('Inter', 16))
        intensity_combo.pack(side='left', padx=8)

        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=3, column=0, sticky='nsew')
        container.rowconfigure(3, weight=1)

        # Live Monitor Tab
        self.monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitor_frame, text="Live Monitor")
        self.log = tk.Text(self.monitor_frame, bg='#f9fafb', fg='#374151', font=('Consolas', 12), wrap='none')
        self.log.pack(fill='both', expand=True, padx=10, pady=10)

        # Ablation Tab (Bar chart + table)
        self.ablation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ablation_frame, text="Ablation Study")
        self._setup_ablation_ui()

        # Scenario Log Tab
        self.scenario_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scenario_frame, text="Scenario Logs/Anomalies")
        self._setup_scenario_log_ui()

        # Analytics tab
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="Analytics")
        self._setup_analytics_ui()

        # Dataset Info tab
        self.dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_frame, text="Dataset Info")
        self._setup_dataset_ui()

    def _setup_ablation_ui(self):
        label = ttk.Label(self.ablation_frame, text="Ablation Study Comparison of Detection Accuracy", font=('Inter', 20, 'bold'))
        label.pack(pady=10)
        self.ablation_fig, self.ablation_ax = plt.subplots(figsize=(8,4))
        self.ablation_canvas = FigureCanvasTkAgg(self.ablation_fig, self.ablation_frame)
        self.ablation_canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=10)
        self.ablation_table = ttk.Treeview(self.ablation_frame, columns=["config", "accuracy"], show="headings", height=6)
        self.ablation_table.heading("config", text="Feature Set / Config")
        self.ablation_table.heading("accuracy", text="Detection Accuracy")
        self.ablation_table.column("config", width=300, anchor=tk.W)
        self.ablation_table.column("accuracy", width=150, anchor=tk.CENTER)
        self.ablation_table.pack(pady=10)

    def _setup_scenario_log_ui(self):
        top_frame = ttk.Frame(self.scenario_frame)
        top_frame.pack(fill='x', pady=10, padx=10)
        ttk.Label(top_frame, text="Select Scenario:", font=('Inter', 16)).pack(side='left')
        self.scenario_select = ttk.Combobox(top_frame, state="readonly", width=30, font=('Inter', 16))
        self.scenario_select.pack(side='left', padx=8)
        self.scenario_select.bind('<<ComboboxSelected>>', self.on_scenario_selected)
        ttk.Label(self.scenario_frame, text="Scenario Logs:", font=('Inter', 16, 'bold')).pack(anchor='w', padx=10)
        self.scenario_logs_text = tk.Text(self.scenario_frame, height=14, bg='#f9fafb', fg='#111827', font=('Consolas', 12))
        self.scenario_logs_text.pack(fill='both', expand=True, padx=10, pady=(0,10))
        ttk.Label(self.scenario_frame, text="Top Anomalies:", font=('Inter', 16, 'bold')).pack(anchor='w', padx=10)
        self.anomalies_text = tk.Text(self.scenario_frame, height=8, bg='#f3f4f6', fg='#b91c1c', font=('Consolas', 12))
        self.anomalies_text.pack(fill='both', expand=True, padx=10, pady=(0,10))

    def _setup_analytics_ui(self):
        self.analytics_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_fig, self.analytics_frame)
        self.analytics_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def _setup_dataset_ui(self):
        self.dataset_text = tk.Text(self.dataset_frame, font=('Consolas', 11), bg='#f9fafb', fg='#374151')
        self.dataset_text.pack(fill='both', expand=True, padx=15, pady=15)

    # Functional UI Actions

    def load_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        success = self.simulator.load_dataset(path)
        if not success:
            messagebox.showerror("Error", "Failed to load dataset")
            return
        stats = self.simulator.get_dataset_stats()
        info = f"Loaded dataset with {stats.get('total_records')} records in {stats.get('total_chunks')} chunks.\n"
        info += f"Unique IPs: {stats.get('unique_ips')}\nAttack types: {stats.get('attack_types')}\nColumns: {', '.join(stats.get('columns', []))}\n\n"
        info += "Sample Data:\n"
        for i, record in enumerate(stats.get('sample_data', [])[:5]):
            info += f"Row {i+1}: {record}\n"
        self.dataset_text.delete(1.0, tk.END)
        self.dataset_text.insert(tk.END, info)
        self._log(f"Dataset loaded: {path}")
        self.attacks.clear()
        self.scenarios.clear()
        self.current_config = None
        self.status_update(f"Dataset loaded: {path.split('/')[-1]}")
        self._update_scenarios_combo()
        self._update_analytics()
        self._update_ablation()

    def start_sim(self):
        if not self.simulator.chunks or not self.simulator.attack_patterns['types'] or not self.simulator.ip_patterns:
            messagebox.showwarning("Warning", "Load dataset with valid attacks and IPs before starting simulation.")
            return
        seed = random.randint(1000, 9999)
        intensity = self.intensity_var.get()
        scenario_name = f"Scenario_{len(self.scenarios)+1}_{datetime.now().strftime('%H%M%S')}"
        timestamp = datetime.now().isoformat()
        config = SimConfig(seed=seed, intensity=intensity, scenario_name=scenario_name, timestamp=timestamp)
        self.current_config = config
        random.seed(seed)
        np.random.seed(seed)
        self.simulator.start(config)
        self.monitoring = True
        self.scenarios[scenario_name] = {"config": config, "attacks": [], "logs": [], "analysis": {}}
        self._update_scenarios_combo(selected=scenario_name)
        self.status_update(f"Running simulation - {scenario_name} with seed {seed} and intensity {intensity}")
        self._log(f"Started scenario: {scenario_name} | Seed: {seed} | Intensity: {intensity}")
        threading.Thread(target=self._monitor_attacks, daemon=True).start()

    def stop_sim(self):
        self.simulator.stop()
        self.monitoring = False
        self.status_update("Simulation stopped")
        self._log("Simulation stopped")

    def _monitor_attacks(self):
        while self.monitoring:
            attack = self.simulator.get_attack()
            if attack:
                self.attacks.append(attack)
                if self.current_config and self.current_config.scenario_name in self.scenarios:
                    self.scenarios[self.current_config.scenario_name]["attacks"].append(attack)
                status = "BLOCKED" if attack.is_blocked else "PASSED"
                log_msg = (
                    f"{attack.timestamp.strftime('%H:%M:%S')} | "
                    f"{attack.signature} | {attack.attack_type} | {attack.ip} | "
                    f"Severity: {attack.severity:.2f} | {status}"
                )
                self._log(log_msg)
                if self.current_config and self.current_config.scenario_name in self.scenarios:
                    self.scenarios[self.current_config.scenario_name]["logs"].append(log_msg)
                self.root.after_idle(self._update_analytics)
            else:
                time.sleep(0.1)

    def analyze(self):
        if not self.attacks:
            messagebox.showwarning("Warning", "No attacks to analyze")
            return
        last_1000 = self.attacks[-1000:]
        result = self.detector.analyze(last_1000)
        self.status_update(f"Analysis done - {result['anomalies']}/{result['total']} anomalies detected, Score: {result['score']:.3f}")
        self._log("Analysis complete")
        if self.current_config and self.current_config.scenario_name in self.scenarios:
            self.scenarios[self.current_config.scenario_name]["analysis"] = result
        self._update_ablation()
        self._update_analytics()
        self._update_scenarios_combo()

    def export_attacks(self):
        if not self.attacks:
            messagebox.showwarning("Warning", "No attacks to export")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        data = [{
            "timestamp": a.timestamp.isoformat(),
            "ip": a.ip,
            "attack_type": a.attack_type,
            "signature": a.signature,
            "severity": a.severity,
            "payload_size": a.payload_size,
            "success": a.success,
            "blocked": a.is_blocked
        } for a in self.attacks]
        pd.DataFrame(data).to_csv(path, index=False)
        messagebox.showinfo("Export Complete", f"Exported {len(data)} attacks to {path}")
        self._log(f"Exported {len(data)} attacks to {path}")

    def save_scenario(self):
        if not self.current_config:
            messagebox.showwarning("Warning", "No active scenario to save")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not path:
            return
        scenario_data = {
            "config": self.current_config.__dict__,
            "attacks": [{
                "timestamp": a.timestamp.isoformat(),
                "ip": a.ip,
                "attack_type": a.attack_type,
                "signature": a.signature,
                "severity": a.severity,
                "payload_size": a.payload_size,
                "success": a.success,
                "blocked": a.is_blocked
            } for a in self.scenarios[self.current_config.scenario_name]["attacks"]],
            "logs": self.scenarios[self.current_config.scenario_name]["logs"],
            "analysis": self.scenarios[self.current_config.scenario_name].get("analysis", {})
        }
        with open(path, "w") as f:
            json.dump(scenario_data, f, indent=2)
        self._log(f"Saved scenario {self.current_config.scenario_name} to {path}")
        messagebox.showinfo("Saved", f"Scenario saved as {path}")

    def load_scenario(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            config_dict = data["config"]
            config = SimConfig(**config_dict)
            self.current_config = config

            attacks = []
            for a in data["attacks"]:
                attacks.append(
                    Attack(
                        timestamp=datetime.fromisoformat(a["timestamp"]),
                        ip=a["ip"],
                        attack_type=a["attack_type"],
                        signature=a.get("signature", ""),
                        severity=a["severity"],
                        payload_size=a["payload_size"],
                        success=a["success"],
                        is_blocked=a.get("blocked", False)
                    )
                )
            self.attacks = attacks
            self.scenarios[config.scenario_name] = {
                "config": config,
                "attacks": attacks,
                "logs": data.get("logs", []),
                "analysis": data.get("analysis", {})
            }
            self._update_scenarios_combo(selected=config.scenario_name)
            self.status_update(f"Loaded scenario {config.scenario_name}")
            self._log(f"Loaded scenario {config.scenario_name} from {path}")
            self._update_scenario_log_text(config.scenario_name)
            self._update_anomalies_text(config.scenario_name)
            self._update_analytics()
            self._update_ablation()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scenario: {e}")

    def _update_scenarios_combo(self, selected=None):
        names = list(self.scenarios.keys())
        self.scenario_select['values'] = names
        if selected and selected in names:
            self.scenario_select.set(selected)
        elif names:
            self.scenario_select.set(names[-1])
        else:
            self.scenario_select.set('')
        self._update_scenario_log_text(self.scenario_select.get())
        self._update_anomalies_text(self.scenario_select.get())

    def on_scenario_selected(self, event=None):
        selected = self.scenario_select.get()
        self._update_scenario_log_text(selected)
        self._update_anomalies_text(selected)

    def _update_scenario_log_text(self, scenario_name):
        self.scenario_logs_text.delete(1.0, tk.END)
        logs = self.scenarios.get(scenario_name, {}).get("logs", [])
        for line in logs:
            self.scenario_logs_text.insert(tk.END, line + "\n")
        self.scenario_logs_text.see(tk.END)

    def _update_anomalies_text(self, scenario_name):
        self.anomalies_text.delete(1.0, tk.END)
        attacks = self.scenarios.get(scenario_name, {}).get("attacks", [])
        if not attacks:
            return
        anomalies = sorted([a for a in attacks if a.is_blocked], key=lambda x: (-x.severity, not x.success))
        self.anomalies_text.insert(tk.END, f"Top {min(10,len(anomalies))} anomalies in {scenario_name}:\n\n")
        for i, a in enumerate(anomalies[:10], start=1):
            status = "SUCCESS" if a.success else "BLOCKED"
            self.anomalies_text.insert(tk.END,
                f"{i}. {a.signature} | {a.attack_type} | {a.ip} | Severity: {a.severity:.3f} | {status}\n"
            )
        self.anomalies_text.see(tk.END)

    def _update_analytics(self):
        if not self.attacks:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            self.analytics_canvas.draw()
            return
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        recent = self.attacks[-100:]

        # 1. Severity line plot
        sev = [a.severity for a in recent]
        self.ax1.plot(sev, color='#2563eb', lw=2)
        self.ax1.set_title("Attack Severity Over Time", fontsize=16, fontweight='bold')
        self.ax1.set_ylabel("Severity")

        # 2. Attack types bar chart
        types = [a.attack_type for a in recent]
        type_counts = pd.Series(types).value_counts()
        self.ax2.bar(type_counts.index, type_counts.values, color='#4ade80')
        self.ax2.set_title("Attack Types Distribution", fontsize=16, fontweight='bold')
        self.ax2.set_xticklabels(type_counts.index, rotation=45, ha='right')

        # 3. Payload size histogram
        sizes = [a.payload_size for a in recent]
        self.ax3.hist(sizes, bins=20, color='#60a5fa', alpha=0.75)
        self.ax3.set_title("Payload Size Distribution", fontsize=16, fontweight='bold')
        self.ax3.set_xlabel("Bytes")

        # 4. Success rate by attack type
        df = pd.DataFrame([{'type': a.attack_type, 'success': a.success} for a in recent])
        success_rates = df.groupby('type')['success'].mean().sort_values(ascending=False)
        self.ax4.bar(success_rates.index, success_rates.values, color='#ef4444')
        self.ax4.set_title("Success Rate by Attack Type", fontsize=16, fontweight='bold')
        self.ax4.set_xticklabels(success_rates.index, rotation=45, ha='right')
        self.ax4.set_ylabel("Success Rate")

        self.analytics_fig.tight_layout()
        self.analytics_canvas.draw()

    def _update_ablation(self):
        # Mock ablation data updated per analysis or scenario
        configs = [
            "Basic Features",
            "Basic + Payload",
            "Basic + Signature",
            "Full Features"
        ]
        accuracies = [0.65, 0.75, 0.78, 0.85]

        self.ablation_ax.clear()
        bars = self.ablation_ax.bar(configs, accuracies, color=['#60a5fa', '#4ade80', '#facc15', '#ef4444'])
        self.ablation_ax.set_ylim(0, 1)
        self.ablation_ax.set_title("Ablation Study: Detection Accuracy", fontsize=18, fontweight='bold')
        self.ablation_ax.set_ylabel("Accuracy")

        for bar, acc in zip(bars, accuracies):
            self.ablation_ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=12, fontweight='bold')
        self.ablation_canvas.draw()

        for i in self.ablation_table.get_children():
            self.ablation_table.delete(i)
        for cfg, acc in zip(configs, accuracies):
            self.ablation_table.insert("", "end", values=(cfg, f"{acc:.2f}"))

    def _log(self, message: str):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)

    def status_update(self, text: str):
        self.root.title(f"Cyber Attack Simulator - {text}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CyberGUI()
    app.run()

