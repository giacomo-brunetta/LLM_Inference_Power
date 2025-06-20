#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
import json
import os
import struct
from pathlib import Path
import glob
import re
from typing import Dict, Optional
import time
import os 
import importlib.util

class SystemMonitor:
    def __init__(self):
        self.num_cpus = self._get_cpu_count()
        self.rapl_domains = {}
        self._init_rapl_paths()  # Initialize paths ONCE
        self.last_energy_reading = {domain: 0 for domain in self.rapl_domains} # Initialize with zeros
        self.last_energy_time = None

        # Initialize GPU monitoring
        self.gpu_monitor = None
        try:
            # Get the full path to the .so file
            build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')
            so_path = None
            for file in os.listdir(build_dir):
                if file.startswith('gpu_power') and file.endswith('.so'):
                    so_path = os.path.join(build_dir, file)
                    break

            if not so_path:
                print("Could not find gpu_power .so file in build directory")
                exit(1)

            print(f"Found module at: {so_path}")

            # Load the module directly
            spec = importlib.util.spec_from_file_location("gpu_power", so_path)
            gpu_power = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpu_power)

            # Create and test the monitor
            self.gpu_monitor = gpu_power.GPUPowerMonitor()
            if not self.gpu_monitor.initialize():
                print("Warning: Failed to initialize GPU monitoring")
        except Exception as e:
            print(f"Warning: Could not initialize GPU monitoring: {e}")
        
    def _get_cpu_count(self):
        """Get the number of CPU cores."""
        try:
            with open('/proc/cpuinfo') as f:
                return len([line for line in f if line.startswith('processor')])
        except Exception as e:
            print(f"Error getting CPU count: {e}")
            return 0
            
    def _init_rapl_paths(self):
        """Initialize paths for RAPL readings."""
        base_path = "/sys/class/powercap"
        
        try:
            # First find the intel-rapl directories
            rapl_dirs = glob.glob(os.path.join(base_path, "intel-rapl:*"))
            print(f"Found RAPL directories: {rapl_dirs}")
            
            for rapl_dir in rapl_dirs:
                try:
                    # Read name file to identify the domain
                    with open(os.path.join(rapl_dir, "name")) as f:
                        name = f.read().strip()
                    
                    # Get the energy file path
                    energy_path = os.path.join(rapl_dir, "energy_uj")
                    
                    if os.path.exists(energy_path):
                        if "package" in name.lower():
                            package_num = re.search(r'intel-rapl:(\d+)', rapl_dir)
                            if package_num:
                                self.rapl_domains[f'CPU Socket {package_num.group(1)}'] = energy_path
                        
                        # Look for DRAM in subdirectories
                        dram_dirs = glob.glob(os.path.join(rapl_dir, "intel-rapl:*"))
                        for dram_dir in dram_dirs:
                            with open(os.path.join(dram_dir, "name")) as f:
                                subdomain_name = f.read().strip()
                            if "dram" in subdomain_name.lower():
                                dram_num = re.search(r'intel-rapl:(\d+)', rapl_dir)
                                if dram_num:
                                    self.rapl_domains[f'DRAM Channel {dram_num.group(1)}'] = os.path.join(dram_dir, "energy_uj")
                                    
                except Exception as e:
                    print(f"Error processing RAPL directory {rapl_dir}: {e}")
                    
            print("Found RAPL domains:", self.rapl_domains)
            
        except Exception as e:
            print(f"Error initializing RAPL paths: {e}")

    def read_msr(self, cpu, register):
        """Read MSR register value for given CPU."""
        try:
            with open(f"/dev/cpu/{cpu}/msr", "rb") as f:
                f.seek(register)
                return struct.unpack("Q", f.read(8))[0]
        except Exception as e:
            print(f"Error reading MSR: {e}")
            return None

    def get_uncore_freq(self, cpu=0):
        """Get current uncore frequency."""
        MSR_UNCORE_PERF_STATUS = 0x621
        try:
            status_value = self.read_msr(cpu, MSR_UNCORE_PERF_STATUS)
            if status_value is not None:
                current_ratio = status_value & 0xFF
                return current_ratio * 100
            return None
        except Exception as e:
            print(f"Error getting uncore freq: {e}")
            return None

    def get_core_freq(self, cpu):
        """Get current frequency for a specific CPU core."""
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq') as f:
                return int(f.read().strip()) / 1000  # Convert KHz to MHz
        except Exception as e:
            print(f"Error reading CPU{cpu} frequency: {e}")
            return None

    def get_rapl_power(self):
        """Read RAPL power measurements in Watts."""
        current_time = time.time()
        power_data = {}
        
        try:
            # Read current energy values
            current_readings = {}
            for domain, path in self.rapl_domains.items():
                try:
                    with open(path) as f:
                        energy = int(f.read().strip())  # In µJ
                        current_readings[domain] = energy
                except Exception as e:
                    print(f"Error reading RAPL data for {domain}: {e}")
            
            # Calculate power if we have previous readings
            if self.last_energy_reading and self.last_energy_time:
                time_diff = current_time - self.last_energy_time
                if time_diff > 0:  # Prevent division by zero
                    for domain, current_energy in current_readings.items():
                        if domain in self.last_energy_reading:
                            # Handle counter wraparound (counter is 32 bits)
                            energy_diff = current_energy - self.last_energy_reading[domain]
                            if energy_diff < 0:
                                energy_diff += 2**32
                            
                            # Convert µJ to Watts (µJ/s = µW, then convert to W)
                            power = energy_diff / time_diff / 1_000_000
                            power_data[domain] = round(power, 2)
            
            # Update stored values for next reading
            self.last_energy_reading = current_readings
            self.last_energy_time = current_time
            
        except Exception as e:
            print(f"Error calculating power: {e}")
            
        return power_data

    def get_all_metrics(self):
        metrics = {
            'uncore_socket0': self.get_uncore_freq(0) or 0,
            'uncore_socket1': self.get_uncore_freq(32) or 0,
            'core_freqs': {},
            'rapl_power': self.get_rapl_power(),
            'gpu_power': {}
        }
        
        # Get frequency for each core
        for cpu in range(self.num_cpus):
            freq = self.get_core_freq(cpu)
            if freq is not None:
                metrics['core_freqs'][f'Core {cpu}'] = freq

        # Add GPU power readings if available
        if self.gpu_monitor:
            try:
                gpu_readings = self.gpu_monitor.get_power_readings()
                for i, gpu in enumerate(gpu_readings):
                    metrics['gpu_power'][f'GPU {i}'] = {
                        'name': gpu.gpu_name,
                        'uuid': gpu.uuid,
                        'total': gpu.card_power,
                        'tile0': gpu.tile0_power,
                        'tile1': gpu.tile1_power
                    }
            except Exception as e:
                print(f"Warning: Failed to get GPU readings: {e}")
        
        return metrics

class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            try:
                with open('monitor.html', 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_error(404, "File not found")
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = monitor.get_all_metrics()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

def main():
    if os.geteuid() != 0:
        print("This script must be run as root!")
        exit(1)

    if not Path("/dev/cpu/0/msr").exists():
        print("Loading MSR module...")
        os.system("modprobe msr")

    global monitor  # Declare monitor as global
    monitor = SystemMonitor()  # Create the monitor instance HERE, only once

    server_address = ('', 8030)
    httpd = HTTPServer(server_address, MonitorHandler)
    print("\nServer started at http://localhost:8030")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    main()