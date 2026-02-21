import socket
import struct
import threading
import time
from datetime import datetime
import logging
from collections import deque
import json

class IEC104Monitor:
    def __init__(self, host='0.0.0.0', port=2404):
        self.host = host
        self.port = port
        self.connections = {}
        self.packet_buffer = deque(maxlen=1000)  # Store last 1000 packets
        self.running = False
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='iec104_monitor.log'
        )
        
    def start_monitoring(self):
        """Start the monitoring server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        logging.info(f"Started monitoring on {self.host}:{self.port}")
        
        # Start packet processor thread
        self.processor_thread = threading.Thread(target=self.process_packets)
        self.processor_thread.start()
        
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                logging.info(f"New connection from {addr}")
                self.connections[addr] = {
                    'socket': client_sock,
                    'thread': threading.Thread(
                        target=self.handle_client,
                        args=(client_sock, addr)
                    )
                }
                self.connections[addr]['thread'].start()
            except Exception as e:
                logging.error(f"Error accepting connection: {e}")
                
    def handle_client(self, client_socket, addr):
        """Handle individual client connections"""
        while self.running:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                    
                packet = self.parse_packet(data)
                self.packet_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    'source': addr[0],
                    'data': packet
                })
                
            except Exception as e:
                logging.error(f"Error handling client {addr}: {e}")
                break
                
        logging.info(f"Connection closed from {addr}")
        client_socket.close()
        del self.connections[addr]
        
    def parse_packet(self, data):
        """Parse IEC 60870-5-104 packet"""
        try:
            # Basic packet structure parsing
            packet = {}
            
            # Start byte should be 0x68
            if data[0] != 0x68:
                raise ValueError("Invalid start byte")
                
            # Parse APCI (Application Protocol Control Information)
            packet['length'] = data[1]
            packet['control_field'] = list(data[2:6])
            
            # Parse ASDU if present
            if len(data) > 6:
                packet['asdu'] = self.parse_asdu(data[6:])
                
            return packet
            
        except Exception as e:
            logging.error(f"Error parsing packet: {e}")
            return None
            
    def parse_asdu(self, data):
        """Parse Application Service Data Unit"""
        asdu = {}
        try:
            asdu['type'] = data[0]  # Type identification
            asdu['vsq'] = data[1]   # Variable structure qualifier
            asdu['cot'] = data[2]   # Cause of transmission
            asdu['addr'] = struct.unpack('<H', data[3:5])[0]  # Common address
            
            # Parse information objects based on type
            asdu['objects'] = self.parse_information_objects(
                data[5:],
                asdu['type'],
                asdu['vsq']
            )
            
            return asdu
            
        except Exception as e:
            logging.error(f"Error parsing ASDU: {e}")
            return None
            
    def parse_information_objects(self, data, type_id, vsq):
        """Parse information objects based on type ID"""
        objects = []
        
        # Number of objects
        num_objects = vsq & 0x7F
        
        # Parse based on common type IDs
        try:
            if type_id in [1, 3, 5]:  # Single-point information
                for i in range(num_objects):
                    obj = {
                        'value': bool(data[i] & 0x01),
                        'quality': data[i] >> 1
                    }
                    objects.append(obj)
                    
            elif type_id in [9, 11, 13]:  # Measured value, normalized value
                for i in range(num_objects):
                    start_idx = i * 3
                    obj = {
                        'value': struct.unpack('<h', data[start_idx:start_idx+2])[0],
                        'quality': data[start_idx+2]
                    }
                    objects.append(obj)
                    
            # Add more type parsing as needed
                    
        except Exception as e:
            logging.error(f"Error parsing information objects: {e}")
            
        return objects
        
    def process_packets(self):
        """Process captured packets for analysis"""
        while self.running:
            if self.packet_buffer:
                packet = self.packet_buffer[-1]  # Get latest packet
                
                # Analyze for potential anomalies
                self.analyze_packet(packet)
                
            time.sleep(0.1)  # Prevent tight loop
                
    def analyze_packet(self, packet):
        """Analyze packet for suspicious patterns"""
        try:
            if packet['data'] and 'asdu' in packet['data']:
                asdu = packet['data']['asdu']
                
                # Check for suspicious type IDs
                if asdu['type'] in [45, 46, 47]:  # Command types
                    logging.warning(f"Command packet detected from {packet['source']}")
                    
                # Check for unusual cause of transmission
                if asdu['cot'] in [41, 42, 43, 44]:  # Unknown cause of transmission
                    logging.warning(f"Unusual COT detected from {packet['source']}")
                    
                # Add more analysis rules as needed
                    
        except Exception as e:
            logging.error(f"Error analyzing packet: {e}")
            
    def stop_monitoring(self):
        """Stop the monitoring server"""
        self.running = False
        
        # Close all client connections
        for conn in self.connections.values():
            conn['socket'].close()
            
        self.server_socket.close()
        logging.info("Monitoring stopped")
        
    def get_statistics(self):
        """Get current monitoring statistics"""
        stats = {
            'total_connections': len(self.connections),
            'active_sources': list(self.connections.keys()),
            'packet_buffer_size': len(self.packet_buffer),
            'last_packet_time': None
        }
        
        if self.packet_buffer:
            stats['last_packet_time'] = self.packet_buffer[-1]['timestamp']
            
        return stats

# Example usage
if __name__ == "__main__":
    monitor = IEC104Monitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()