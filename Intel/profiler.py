import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import importlib.util
import numpy as np

class Profiler:
    def __init__(self, interval=0.5, gpus=1, active_gpus=1):
        self.gpus = gpus
        self.active_gpus = active_gpus

        self.interval = multiprocessing.Value('d', interval)
        self.len = int(7200 / interval)

        self.inference_powers = []
        self.inference_powers_time = []
        self.inference_mem_used = []
        self.inference_gpu_utils = []

        self.powers = [multiprocessing.Array('d', self.len) for _ in range(gpus)]
        self.powers_time = [multiprocessing.Array('d', self.len) for _ in range(gpus)]
        self.mem_used = []
        self.gpu_utils = []

        self.process = None
        self.prevTime = multiprocessing.Value('d', time.time())
        self.halt = multiprocessing.Value('i', 1)
        self.count = multiprocessing.Value('i', 0)
        self.isrunning = multiprocessing.Value('i', 0)
        self.alive = multiprocessing.Value('i', 0)
        self.init()

    def init(self):
        self.halt.value = 1
        self.alive.value = 1
        args = (
            self.powers,
            self.powers_time,
            self.mem_used,
            self.gpu_utils,
            self.count,
            self.halt,
            self.alive,
            self.isrunning,
            self.prevTime,
            self.interval,
        )
        self.process = multiprocessing.Process(target=self._getGpuPower, args=args)
        self.process.start()

    def _getGpuPower(self, powers, times, mem_used, gpu_utils, count, halt, alive, isrunning, prevTime, interval):
        
        gpu_monitor = None
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
                return  # Exit gracefully instead of exit(1)

            # Load the module directly
            spec = importlib.util.spec_from_file_location("gpu_power", so_path)
            gpu_power = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpu_power)

            # Create and test the monitor
            gpu_monitor = gpu_power.GPUPowerMonitor()
            if not gpu_monitor.initialize():
                print("Warning: Failed to initialize GPU monitoring")
                return  # Exit gracefully if initialization fails
                
        except Exception as e:
            print(f"Warning: Could not initialize GPU monitoring: {e}")
            return  # Exit gracefully instead of exit(1)

        # Main monitoring loop
        while alive.value: 
            while not halt.value:
                isrunning.value = 1
                
                # Get current time BEFORE using it
                new_time = time.time()
                
                try:
                    gpu_readings = gpu_monitor.get_power_readings()
                    for i, gpu in enumerate(gpu_readings):
                        idx = count.value
                        if idx < len(powers[i]):  # Bounds check
                            powers[i][idx] = gpu.card_power
                            times[i][idx] = new_time - prevTime.value
                            #mem_used[i][idx] = 0  # Reading not available
                            #gpu_utils[i][idx] = 0  # Reading not available
                except Exception as e:
                    print(f"Error getting GPU readings: {e}")
                    # Continue loop instead of crashing

                # Wait until next interval
                while (new_time - prevTime.value) < interval.value:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    new_time = time.time()

                # Increment counter and update prevTime
                count.value = (count.value + 1) % len(powers[0])  # Wrap around when full
                prevTime.value = new_time
                isrunning.value = 0

    def start(self):
        self.count.value = 0
        self.prevTime.value = time.time()
        self.halt.value = 0

    def destroy(self):
        self.alive.value = 0
        self.process.join()

    def stop(self):
        for i in range(self.gpus):
            # stats is a dict with keys:
            #   'power'           -> array of power samples (µW)
            #   'time_intervals'  -> array of elapsed‐time between samples (s)
            #   'mem_used_mib'    -> array of total memory used (MiB)
            #   'gpu_util_pct'    -> array of average GPU utilization (%)

            # self.inference_powers.append(np.array(self.powers[i][:self.count.value]))
            self.inference_powers_time.append(np.array(self.powers_time[i][:self.count.value]))
            # self.inference_mem_used.append(np.array(self.mem_used[i][:self.count.value]))
            # self.inference_gpu_utils.append(np.array(self.inference_gpu_utils[i][:self.count.value]))

            self.inference_powers = [None] * self.gpus
            self.inference_mem_used = [None] * self.gpus
            self.inference_gpu_utils = [None] * self.gpus

            self.destroy()

    def get_stats(self, array):
        if array is None or len(array) == 0:
            return 0, 0, 0, 0
        return np.mean(array), np.max(array), np.percentile(array, 50), np.percentile(array, 95)

    def calculate_metrics(self, verbose=True):
        if self.inference_powers[0] is None or len(self.inference_powers) == 0:
            return  {
                # Energy
                "active_energy" : 0,
                "total_energy" : 0,
                # Power (active GPUs)
                "active_power_avg" : 0,
                "active_power_peak" : 0,
                "active_power_p50"  : 0,
                "active_power_p95"  : 0,

                # Power (active + idle GPUs)
                "total_power_avg"   : 0,
                "total_power_peak" : 0,
                "total_power_p50"  : 0,
                "total_power_p95"  : 0,

                # Memory (active GPUs)
                "active_gpu_util_avg"  : 0,
                "active_gpu_util_peak" : 0,
                "active_gpu_util_p50"  : 0,
                "active_gpu_util_p95"  : 0,
                # Memory (active GPUs)
                "active_mem_avg"  : 0,
                "active_mem_peak" : 0,
                "active_mem_p50"  : 0,
                "active_mem_p95"  : 0,
                # Memory (active + idle GPUs)
                "total_mem_avg"  : 0,
                "total_mem_peak" : 0,
                "total_mem_p50"  : 0,     
                "total_mem_p95"  : 0, 

                # GPU Utilization (active GPUs)
                "total_gpu_util_avg"  : 0,
                "total_gpu_util_peak" : 0,
                "total_gpu_util_p50"  : 0,
                "total_gpu_util_p95"  : 0,
            }
        else:
            total_power = None

            if verbose:
                print("\n---------------- Power & Memory -----------------------")

            for gpu_id in range(self.gpus):
                power    = self.inference_powers[gpu_id] # / 1000
                times    = self.inference_powers_time[gpu_id]
                mem_used = np.zeros(len(times)) #temp
                gpu_util = np.zeros(len(times)) #temp

                energy = np.sum(power * times)

                # Aggregate total & active power time series
                if total_power is None:
                    total_power     = power.copy() 
                    active_power    = power.copy()
                    active_energy   = energy
                    total_energy    = energy
                    active_mem_used = mem_used.copy()
                    total_mem_used  = mem_used.copy()
                    active_gpu_util = gpu_util.copy()
                    total_gpu_util  = gpu_util.copy()

                elif gpu_id < self.active_gpus:
                    active_power    += power
                    total_power     += power
                    active_energy   += energy
                    total_energy    += energy
                    active_mem_used += mem_used
                    total_mem_used  += mem_used
                    active_gpu_util += gpu_util
                    total_gpu_util  += gpu_util
                else:
                    total_power     += power
                    total_energy    += energy
                    total_mem_used  += mem_used
                    total_gpu_util  += gpu_util

                if verbose:
                    print(f"GPU {gpu_id}:")
                    print(f"    Power avg      : {np.mean(power): .3f} W")
                    print(f"    Power peak     : {np.max(power): .3f} W")
                    print(f"    Energy         : {energy: .3f} J")
                    print(f"    Memory avg     : {np.mean(mem_used): .3f} MiB")
                    print(f"    Memory peak    : {np.max(mem_used): .3f} MiB")
                    print(f"    GPU util avg   : {np.mean(gpu_util): .2f} %")
                    print(f"    GPU util peak  : {np.max(gpu_util): .2f} %")

            return active_power, total_power, active_mem_used, total_mem_used, active_gpu_util / self.active_gpus, total_gpu_util / self.gpus