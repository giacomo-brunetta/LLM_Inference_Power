import multiprocessing
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from py3nvml.py3nvml import nvmlDeviceGetPowerUsage,  \
    nvmlDeviceGetCount,  \
    nvmlDeviceGetHandleByIndex, \
    nvmlInit, \
    nvmlShutdown

from py3nvml.py3nvml import (
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown
)
import multiprocessing
import time

class gpuPowerProbe(object):
    def __init__(self, interval, gpu_id=-1):
        self.interval = multiprocessing.Value('d', interval)
        self.len = int(7200 / interval)

        self.powers = multiprocessing.Array('d', self.len)
        self.times = multiprocessing.Array('d', self.len)
        self.mem_used = multiprocessing.Array('d', self.len)
        self.gpu_utils = multiprocessing.Array('d', self.len)

        self.gpu_id = multiprocessing.Value('i', gpu_id)
        self.process = None
        self.prevTime = multiprocessing.Value('d', time.time())
        self.halt = multiprocessing.Value('i', 1)
        self.count = multiprocessing.Value('i', 0)
        self.isrunning = multiprocessing.Value('i', 0)
        self.alive = multiprocessing.Value('i', 0)
        self.init()

    def _getGpuPower(self, powers, times, mem_used, gpu_utils,
                     gpu_id, count, halt, alive, isrunning, prevTime, interval):
        nvmlInit()

        while alive.value:
            while not halt.value:
                isrunning.value = 1

                # Determine which GPUs to query
                if gpu_id.value > -1:
                    handles = [nvmlDeviceGetHandleByIndex(gpu_id.value)]
                else:
                    num_gpus = nvmlDeviceGetCount()
                    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

                # --- Power measurement ---
                power = 0
                for h in handles:
                    power += nvmlDeviceGetPowerUsage(h)

                # --- Memory measurement (sum used memory across handles, convert to MiB) ---
                total_mem_used = 0
                for h in handles:
                    mem_info = nvmlDeviceGetMemoryInfo(h)
                    total_mem_used += mem_info.used / (1024 ** 2)  # bytes -> MiB

                # --- Utilization measurement ---
                gpu_util_sum = 0
                for h in handles:
                    util = nvmlDeviceGetUtilizationRates(h)
                    gpu_util_sum += util.gpu
                avg_gpu_util = gpu_util_sum / len(handles)

                # Wait until next interval
                new_time = time.time()
                while (new_time - prevTime.value) < interval.value:
                    new_time = time.time()

                # Log everything at this timestamp index
                idx = count.value
                powers[idx] = power
                times[idx] = new_time - prevTime.value
                mem_used[idx] = total_mem_used
                gpu_utils[idx] = avg_gpu_util

                # Increment counter and update prevTime
                count.value += 1
                prevTime.value = new_time
                isrunning.value = 0

        nvmlShutdown()

    def init(self):
        self.halt.value = 1
        self.alive.value = 1
        args = (
            self.powers,
            self.times,
            self.mem_used,
            self.gpu_utils,
            self.gpu_id,
            self.count,
            self.halt,
            self.alive,
            self.isrunning,
            self.prevTime,
            self.interval,
        )
        self.process = multiprocessing.Process(target=self._getGpuPower, args=args)
        self.process.start()

    def start(self):
        self.count.value = 0
        self.prevTime.value = time.time()
        self.halt.value = 0

    def stop(self):
        self.halt.value = 1
        while self.isrunning.value:
            pass
        return {
            'power': self.powers[:self.count.value],
            'time_intervals': self.times[:self.count.value],
            'mem_used_mib': self.mem_used[:self.count.value],
            'gpu_util_pct': self.gpu_utils[:self.count.value],
        }

    def destroy(self):
        self.alive.value = 0
        self.process.join()

import numpy as np

class Profiler:
    def __init__(self, interval=0.5, gpus=1, active_gpus=1, watched_devices = None):
        self.gpus = gpus
        self.active_gpus = active_gpus

        # Lists to collect per‐GPU time series
        self.inference_powers = []          # list of power arrays (µW) per GPU
        self.inference_powers_time = []     # list of time‐interval arrays per GPU
        self.inference_mem_used = []        # list of memory‐used arrays (MiB) per GPU
        self.inference_gpu_utils = []       # list of GPU_util_pct arrays per GPU

        if watched_devices is None:
            self.watched_devices = range(gpus)
        else:
            self.watched_devices = watched_devices
        # Instantiate one gpuPowerProbe per GPU
        self.power_profiles = [
            gpuPowerProbe(interval=interval, gpu_id=gpu_id)
            for gpu_id in self.watched_devices
        ]

    def start(self):
        for power_profile in self.power_profiles:
            power_profile.start()

    def stop(self):
        for power_profile in self.power_profiles:
            stats = power_profile.stop()
            # stats is a dict with keys:
            #   'power'           -> array of power samples (µW)
            #   'time_intervals'  -> array of elapsed‐time between samples (s)
            #   'mem_used_mib'    -> array of total memory used (MiB)
            #   'gpu_util_pct'    -> array of average GPU utilization (%)

            self.inference_powers.append(np.array(stats['power']))
            self.inference_powers_time.append(np.array(stats['time_intervals']))
            self.inference_mem_used.append(np.array(stats['mem_used_mib']))
            self.inference_gpu_utils.append(np.array(stats['gpu_util_pct']))
            power_profile.destroy()

    def get_stats(self, array):
        return np.mean(array), np.max(array), np.percentile(array, 50), np.percentile(array, 95)

    def calculate_metrics(self, verbose=True):
        total_power = None

        if verbose:
            print("\n---------------- Power & Memory -----------------------")

        # Find the minimum number of samples across GPUs for alignment
        min_sample_num = min([len(p) for p in self.inference_powers])

        for gpu_id in self.watched_devices:
            power    = np.array(self.inference_powers[gpu_id][:min_sample_num]) / 1000
            times    = np.array(self.inference_powers_time[gpu_id][:min_sample_num])
            mem_used = np.array(self.inference_mem_used[gpu_id][:min_sample_num])
            gpu_util = np.array(self.inference_gpu_utils[gpu_id][:min_sample_num])

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

        return active_power, total_power, active_mem_used, total_mem_used, active_gpu_util / self.active_gpus, total_gpu_util / self.gpus, active_energy, total_energy
