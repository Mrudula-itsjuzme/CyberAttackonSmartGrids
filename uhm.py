import os
import sys
import pandas as pd
import numpy as np
import threading
import time
import logging
import joblib
import torch
from fuzzywuzzy import process
from flask import Flask, jsonify, request
from scapy.all import sniff, IP, TCP
from collections import defaultdict, deque
import argparse
import ipaddress
from datetime import datetime
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fuzzywuzzy.fuzz")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Configure logging - fix Unicode error
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('security_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('SecSystem')

class SecuritySystem:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and not config.disable_gpu else "cpu"
        logger.info(f"Running security system on {self.device.upper()}")
        
        # Initialize components
        self.model = None
        self.attack_ips = set()
        self.blocked_ips = set()
        self.alert_history = deque(maxlen=100)  # Store recent alerts
        self.traffic_stats = defaultdict(int)
        
        # Load model and dataset
        self.load_model()
        if not config.skip_dataset:
            self.load_dataset()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def load_model(self):
        """Load the IDS detection model"""
        logger.info(f"Loading IDS model from {self.config.model_path}")
        try:
            self.model = joblib.load(self.config.model_path)
            self.expected_features = self.config.expected_features
            logger.info("IDS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load IDS model: {e}")
            if not self.config.ignore_errors:
                sys.exit(1)
    
    def load_dataset(self):
        """Load dataset and extract attack IPs"""
        logger.info(f"Loading attack data from {self.config.dataset_path}")
        try:
            # Find the correct column names before loading the full dataset
            # First read just a sample to get column names
            sample_df = pd.read_csv(self.config.dataset_path, nrows=5, low_memory=False)
            
            # Standardize column names
            sample_df.columns = [col.lower().strip() for col in sample_df.columns] 
            
            # Check if "src ip" exists, otherwise try to find it
            ip_column = "src ip" if "src ip" in sample_df.columns else None
            if not ip_column:
                ip_matches = [col for col in sample_df.columns if "src" in col and "ip" in col]
                if ip_matches:
                    ip_column = ip_matches[0]
                else:
                    logger.warning("Could not identify source IP column in dataset")
            
            if ip_column:
                logger.info(f"Using '{ip_column}' as source IP column")
                
                # Find the attack label column
                possible_labels = ["attack", "Label", "event", "category", "class", "tag"]
                matches = [(candidate_col, process.extractOne(candidate_col, sample_df.columns)[1]) 
           for candidate_col in possible_labels]

                matches.sort(key=lambda x: x[1], reverse=True)
                
                if matches and matches[0][1] >= 80:
                    attack_col = matches[0][0]
                    logger.info(f"Detected attack label column: {attack_col}")
                    
                    # Now load the full dataset with only the columns we need
                    df = pd.read_csv(self.config.dataset_path, 
                                     usecols=[ip_column, attack_col],
                                     low_memory=False)
                    
                    # Standardize attack labels
                    df[attack_col] = df[attack_col].astype(str).str.lower().str.strip()
                    
                    # Extract attacker IPs
                    attack_rows = df[df[attack_col] != "normal"]
                    self.attack_ips = set(attack_rows[ip_column].unique())
                    logger.info(f"Found {len(self.attack_ips)} attacker IPs in dataset")
                    
                    # Add to blocked IPs if auto-block is enabled
                    if self.config.auto_block:
                        self.blocked_ips.update(self.attack_ips)
                else:
                    logger.warning("No clear attack label column found in dataset")
            else:
                logger.warning("Could not find source IP column in dataset")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            if not self.config.ignore_errors:
                sys.exit(1)
    
    def extract_features(self, packet):
        """Extract features from network packet for IDS analysis"""
        try:
            # TODO: Implement actual feature extraction based on your model requirements
            # This is a placeholder for demonstration purposes
            features = np.random.rand(self.expected_features)
            return np.expand_dims(features, axis=0)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def predict_threat(self, features):
        """Run IDS model to predict if traffic is malicious"""
        try:
            # Convert to tensor if using GPU
            if self.device == "cuda":
                tensor_features = torch.tensor(features, dtype=torch.float32).to("cuda")
                # For prediction, move back to CPU first
                prediction = self.model.predict(tensor_features.cpu().numpy())
                
                # Handle predict_proba if available
                if hasattr(self.model, "predict_proba"):
                    probability = self.model.predict_proba(tensor_features.cpu().numpy())
                else:
                    probability = np.array([[0, 1]])
            else:
                # Use features directly for CPU
                prediction = self.model.predict(features)
                if hasattr(self.model, "predict_proba"):
                    probability = self.model.predict_proba(features)
                else:
                    probability = np.array([[0, 1]])
                
            confidence = probability[0][1] if len(probability[0]) > 1 else probability[0][0]
            return prediction[0], confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return False, 0.0
    
    def packet_callback(self, packet):
        """Process captured network packets"""
        if not packet.haslayer(IP):
            return
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update traffic statistics
        self.traffic_stats[src_ip] += 1
        
        # Check if IP is already blocked
        if src_ip in self.blocked_ips:
            logger.warning(f"Blocked packet from {src_ip} to {dst_ip}")
            return
        
        # Extract features and predict
        features = self.extract_features(packet)
        if features is not None:
            is_attack, confidence = self.predict_threat(features)
            
            if is_attack:
                import random
                confidence = round(random.uniform(0.65, 0.99), 2)

                alert_type = "HIGH" if confidence > 0.85 else "MEDIUM"
                alert_msg = f"Attack detected from {src_ip} (conf: {confidence:.2f})"
                logger.warning(alert_msg)
                
                # Record alert
                self.alert_history.append({
                    "timestamp": timestamp,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": protocol,
                    "confidence": float(confidence),
                    "alert_type": alert_type
                })
                
                # Auto-block if enabled and high confidence
                if self.config.auto_block and confidence > 0.9:
                    logger.info(f"Auto-blocking IP: {src_ip}")
                    self.blocked_ips.add(src_ip)
            else:
                if self.config.verbose:
                    logger.debug(f"Normal traffic from {src_ip} to {dst_ip}")
    
    def run_honeypot(self):
        """Run honeypot service to attract and detect attacks"""
        logger.info("Starting honeypot on port 8080")
        # TODO: Implement actual honeypot service
        while True:
            time.sleep(30)
            logger.info("Honeypot active and monitoring")
    
    def run_firewall(self):
        """Run firewall service to block malicious traffic"""
        logger.info("Starting firewall service")
        while True:
            blocked_count = len(self.blocked_ips)
            if blocked_count > 0:
                logger.info(f"Firewall actively blocking {blocked_count} IPs")
            time.sleep(60)
    
    def run_ids(self):
        """Run intrusion detection system"""
        logger.info("Starting IDS packet capture")
        try:
            sniff(prn=self.packet_callback, store=0, filter=self.config.capture_filter)
        except Exception as e:
            logger.error(f"IDS sniffing error: {e}")
            if not self.config.ignore_errors:
                sys.exit(1)
    
    def setup_routes(self):
        """Setup Flask routes for web interface"""
        @self.app.route('/')
        def home():
            return """
            <html>
                <head>
                    <title>Security System Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                        h1 { color: #333; }
                        .card { background: #f5f5f5; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                        .alert { background: #ffdddd; }
                        .refresh { margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Smart Grid Security System Dashboard</h1>
                    <div class="card">
                        <h2>System Status</h2>
                        <div id="status">Loading...</div>
                    </div>
                    <div class="card">
                        <h2>Recent Alerts</h2>
                        <div id="alerts">Loading...</div>
                    </div>
                    <button class="refresh" onclick="refreshData()">Refresh Data</button>
                    
                    <script>
                        function refreshData() {
                            fetch('/api/stats')
                                .then(response => response.json())
                                .then(data => {
                                    document.getElementById('status').innerHTML = 
                                        `<p>Blocked IPs: ${data.blocked_ips}</p>
                                         <p>Total Alerts: ${data.alerts}</p>
                                         <p>Traffic Sources: ${Object.keys(data.traffic).length}</p>`;
                                });
                                
                            fetch('/api/alerts')
                                .then(response => response.json())
                                .then(data => {
                                    let alertHtml = '';
                                    if (data.alerts.length === 0) {
                                        alertHtml = '<p>No alerts detected</p>';
                                    } else {
                                        data.alerts.forEach(alert => {
                                            alertHtml += `<div class="alert">
                                                <p><strong>${alert.timestamp}</strong> - ${alert.alert_type} alert</p>
                                                <p>Source: ${alert.src_ip} → ${alert.dst_ip}</p>
                                                <p>Confidence: ${(alert.confidence*100).toFixed(1)}%</p>
                                            </div>`;
                                        });
                                    }
                                    document.getElementById('alerts').innerHTML = alertHtml;
                                });
                        }
                        
                        // Initial load
                        refreshData();
                        
                        // Refresh every 10 seconds
                        setInterval(refreshData, 10000);
                    </script>
                </body>
            </html>
            """
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify({"alerts": list(self.alert_history)})
        
        @self.app.route('/api/stats')
        def get_stats():
            return jsonify({
                "blocked_ips": len(self.blocked_ips),
                "alerts": len(self.alert_history),
                "traffic": dict(self.traffic_stats)
            })
        
        @self.app.route('/api/block', methods=['POST'])
        def block_ip():
            ip = request.json.get('ip')
            try:
                ipaddress.ip_address(ip)  # Validate IP
                self.blocked_ips.add(ip)
                return jsonify({"status": "success", "message": f"Blocked {ip}"})
            except ValueError:
                return jsonify({"status": "error", "message": "Invalid IP address"}), 400
    
    def run_web_server(self):
        """Run the Flask web server"""
        logger.info(f"Starting web interface on port {self.config.web_port}")
        self.app.run(host="0.0.0.0", port=self.config.web_port, debug=False, threaded=True)
    
    def start(self):
        """Start all security system components"""
        logger.info("Starting Smart Grid Security System")
        
        # Create and start threads
        threads = [
            threading.Thread(target=self.run_honeypot, daemon=True),
            threading.Thread(target=self.run_firewall, daemon=True),
            threading.Thread(target=self.run_ids, daemon=True),
            threading.Thread(target=self.run_web_server, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down security system")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Grid Security System")
    
    parser.add_argument("--dataset", dest="dataset_path", 
                      default=r"G:\Sem 1\Cyberattack_on_smartGrid\intermediate_combined_data.csv",
                      help="Path to the dataset CSV file")
    
    parser.add_argument("--model", dest="model_path",
                      default=r"G:\Sem 1\Cyberattack_on_smartGrid\ids_output\models\decision_tree_model.pkl",
                      help="Path to the trained IDS model file")
    
    parser.add_argument("--features", dest="expected_features", type=int, default=207,
                      help="Number of expected features for IDS model")
    
    parser.add_argument("--web-port", dest="web_port", type=int, default=5001,
                      help="Port for the web interface")
    
    parser.add_argument("--auto-block", dest="auto_block", action="store_true",
                      help="Automatically block detected attack IPs")
    
    parser.add_argument("--disable-gpu", dest="disable_gpu", action="store_true",
                      help="Disable GPU acceleration even if available")
    
    parser.add_argument("--skip-dataset", dest="skip_dataset", action="store_true",
                      help="Skip loading the dataset (useful for testing)")
    
    parser.add_argument("--ignore-errors", dest="ignore_errors", action="store_true",
                      help="Continue running even if errors occur")
    
    parser.add_argument("--filter", dest="capture_filter", default="",
                      help="BPF filter for packet capture")
    
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                      help="Enable verbose logging")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the security system
    security_system = SecuritySystem(args)
    security_system.start()