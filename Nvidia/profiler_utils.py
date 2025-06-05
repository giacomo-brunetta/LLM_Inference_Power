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
        self.mem_utils = multiprocessing.Array('d', self.len)

        self.gpu_id = multiprocessing.Value('i', gpu_id)
        self.process = None
        self.prevTime = multiprocessing.Value('d', time.time())
        self.halt = multiprocessing.Value('i', 1)
        self.count = multiprocessing.Value('i', 0)
        self.isrunning = multiprocessing.Value('i', 0)
        self.alive = multiprocessing.Value('i', 0)
        self.init()

    def _getGpuPower(self, powers, times, mem_used, gpu_utils, mem_utils,
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
                mem_util_sum = 0
                for h in handles:
                    util = nvmlDeviceGetUtilizationRates(h)
                    gpu_util_sum += util.gpu
                    mem_util_sum += util.memory
                avg_gpu_util = gpu_util_sum / len(handles)
                avg_mem_util = mem_util_sum / len(handles)

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
                mem_utils[idx] = avg_mem_util

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
            self.mem_utils,
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
            'mem_util_pct': self.mem_utils[:self.count.value],
        }

    def destroy(self):
        self.alive.value = 0
        self.process.join()

import numpy as np

class GPUProfiler:
    def __init__(self, interval=0.5, gpus=1, active_gpus=1):
        self.gpus = gpus
        self.active_gpus = active_gpus

        # Lists to collect per‐GPU time series
        self.inference_powers = []          # list of power arrays (µW) per GPU
        self.inference_powers_time = []     # list of time‐interval arrays per GPU
        self.inference_mem_used = []        # list of memory‐used arrays (MiB) per GPU
        self.inference_gpu_utils = []       # list of GPU_util_pct arrays per GPU
        self.inference_mem_utils = []       # list of memory_util_pct arrays per GPU

        # Instantiate one gpuPowerProbe per GPU
        self.power_profiles = [
            gpuPowerProbe(interval=interval, gpu_id=gpu_id)
            for gpu_id in range(self.gpus)
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
            #   'mem_util_pct'    -> array of average memory utilization (%)

            self.inference_powers.append(np.array(stats['power']))
            self.inference_powers_time.append(np.array(stats['time_intervals']))
            self.inference_mem_used.append(np.array(stats['mem_used_mib']))
            self.inference_gpu_utils.append(np.array(stats['gpu_util_pct']))
            self.inference_mem_utils.append(np.array(stats['mem_util_pct']))

            power_profile.destroy()

    def get_stats(self, array):
        return np.mean(array), np.max(array), np.percentile(array, 50), np.percentile(array, 95)

    def calculate_metrics(self, verbose=True):
        total_power = None

        if verbose:
            print("\n---------------- Power & Memory -----------------------")

        # Find the minimum number of samples across GPUs for alignment
        min_sample_num = min([len(p) for p in self.inference_powers])

        for gpu_id in range(self.gpus):
            power    = np.array(self.inference_powers[gpu_id][:min_sample_num]) / 1000
            times    = np.array(self.inference_powers_time[gpu_id][:min_sample_num])
            mem_used = np.array(self.inference_mem_used[gpu_id][:min_sample_num])
            gpu_util = np.array(self.inference_gpu_utils[gpu_id][:min_sample_num])
            mem_util = np.array(self.inference_mem_utils[gpu_id][:min_sample_num])

            energy = np.sum(power * times)

            # Aggregate total & active power time series
            if total_power is None:
                total_power     = power.copy()
                active_power    = power.copy()
                active_energy   = energy
                total_energy    = energy
                active_mem_used = mem_used.copy()
                total_mem_used  = mem_used.copy()
                active_mem_util = mem_util.copy()
                total_mem_util  = mem_util.copy()
                active_gpu_util = gpu_util.copy()
                total_gpu_util  = gpu_util.copy()

            elif gpu_id < self.active_gpus:
                active_power    += power
                total_power     += power
                active_energy   += energy
                total_energy    += energy
                active_mem_used += mem_used
                total_mem_used  += mem_used
                active_mem_util += mem_util
                total_mem_util  += mem_util
                active_gpu_util += gpu_util
                total_gpu_util  += gpu_util
            else:
                total_power     += power
                total_energy    += energy
                total_mem_used  += mem_used
                total_mem_util  += mem_util
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

        # Overall aggregated metrics
        active_power_avg,    active_power_peak,    active_power_p50,    active_power_p95    = self.get_stats(active_power)
        total_power_avg,     total_power_peak,     total_power_p50,     total_power_p95     = self.get_stats(total_power)
        active_mem_avg,      active_mem_peak,      active_mem_p50,      active_mem_p95      = self.get_stats(active_mem_used)
        total_mem_avg,       total_mem_peak,       total_mem_p50,       total_mem_p95       = self.get_stats(total_mem_used)
        active_gpu_util_avg, active_gpu_util_peak, active_gpu_util_p50, active_gpu_util_p95 = self.get_stats(active_gpu_util / self.active_gpus)
        total_gpu_util_avg,  total_gpu_util_peak,  total_gpu_util_p50,  total_gpu_util_p95  = self.get_stats(total_gpu_util  / self.gpus)
        active_mem_util_avg, active_mem_util_peak, active_mem_util_p50, active_mem_util_p95 = self.get_stats(active_mem_util / self.active_gpus)
        total_mem_util_avg,  total_mem_util_peak,  total_mem_util_p50,  total_mem_util_p95  = self.get_stats(total_mem_util  / self.gpus)

        if verbose:
            print()
            print(f"Overall Active (first {self.active_gpus} GPU(s)):")
            print(f"    Power avg      : {active_power_avg: .3f} W")
            print(f"    Power peak     : {active_power_peak: .3f} W")
            print(f"    Energy         : {active_energy: .3f} J")
            print(f"    Memory avg     : {active_mem_avg: .3f} MiB ({active_mem_util_avg: .2f} %)")
            print(f"    Memory peak    : {active_mem_peak: .3f} MiB ({active_mem_util_peak: .2f} %)")
            print(f"    GPU util avg   : {active_gpu_util_avg: .2f} %")
            print(f"    GPU util peak  : {active_gpu_util_peak: .2f} %")
            print()
            print("Overall Total (all GPUs):")
            print(f"    Power avg      : {total_power_avg: .3f} W")
            print(f"    Power peak     : {total_power_peak: .3f} W")
            print(f"    Energy         : {total_energy: .3f} J")
            print(f"    Memory avg     : {total_mem_avg: .3f} MiB ({total_mem_util_avg: .2f} %)")
            print(f"    Memory peak    : {total_mem_peak: .3f} MiB ({total_mem_util_peak: .2f} %)")
            print(f"    GPU util avg   : {total_gpu_util_avg: .2f} %")
            print(f"    GPU util peak  : {total_gpu_util_peak: .2f} %")
            print()
    
        return {
            # Energy
            "active_energy": active_energy,
            "total_energy": total_energy,

            # Power (active GPUs)
            "active_power_avg" : active_power_avg,
            "active_power_peak": active_power_peak,
            "active_power_p50": active_power_p50,
            "active_power_p95": active_power_p95,

            # Power (active + idle GPUs)
            "total_power_avg": total_power_avg,
            "total_power_peak": total_power_peak,
            "total_power_p50": total_power_p50,
            "total_power_p95": total_power_p95,
            
            # Memory (active GPUs)
            "active_gpu_util_avg": active_gpu_util_avg,
            "active_gpu_util_peak": active_gpu_util_peak,
            "active_gpu_util_p50": active_gpu_util_p50,
            "active_gpu_util_p95": active_gpu_util_p95,

            # Memory (active + idle GPUs)
            "active_mem_util_avg": active_mem_util_avg,
            "active_mem_util_peak": active_mem_util_peak,
            "active_mem_util_p50": active_mem_util_p50,
            "active_mem_util_p95": active_mem_util_p95,

            # Memory (active GPUs)
            "active_mem_avg": active_mem_avg,
            "active_mem_peak": active_mem_peak,
            "active_mem_p50": active_mem_p50,
            "active_mem_p95": active_mem_p95,

            # Memory (active + idle GPUs)
            "total_mem_avg": total_mem_avg,
            "total_mem_peak": total_mem_peak,
            "total_mem_p50": total_mem_p50,
            "total_mem_p95": total_mem_p95,

            # GPU Utilization (active GPUs)
            "total_gpu_util_avg": total_gpu_util_avg,
            "total_gpu_util_peak": total_gpu_util_peak,
            "total_gpu_util_p50": total_gpu_util_p50,
            "total_gpu_util_p95": total_gpu_util_p95,

            # GPU Utilization (active + idle GPUs)
            "total_mem_util_avg": total_mem_util_avg,
            "total_mem_util_peak": total_mem_util_peak,
            "total_mem_util_p50": total_mem_util_p50,
            "total_mem_util_p95": total_mem_util_p95,
        }

def metrics(results, gpu_profiler, verbose=True):
    # Get all GPU profiler metrics
    gpu_profile = gpu_profiler.calculate_metrics(verbose)
    
    # get data from results
    ttfts = []
    latencies = []
    latencies_with_queue = []
    decode_times = []
    queue_times = []

    input_tokens = []
    output_tokens = []

    start = results[0].metrics.first_scheduled_time
    end = 0

    # Collect token count, latency, and TTFT for every response
    for resp in results:
        end = max(end, resp.metrics.finished_time)
        ttfts.append(resp.metrics.first_token_time - resp.metrics.first_scheduled_time)
        latencies.append(resp.metrics.finished_time - resp.metrics.first_scheduled_time)
        latencies_with_queue.append(resp.metrics.finished_time - resp.metrics.arrival_time)
        decode_times.append(resp.metrics.last_token_time - resp.metrics.first_token_time)
        queue_times.append(resp.metrics.time_in_queue)

        input_tokens.append(len(resp.prompt_token_ids))
        output_tokens.append(len(resp.outputs[0].token_ids))

    input_tokens = np.array(input_tokens)
    output_tokens = np.array(output_tokens)
    ttfts = np.array(ttfts)
    latencies = np.array(latencies)
    latencies_with_queue = np.array(latencies_with_queue)
    decode_times = np.array(decode_times)
    queue_times = np.array(queue_times)

    # Build the per-response DataFrame
    latency_data = pd.DataFrame({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latencies,
        "latency_with_queue": latencies_with_queue,
        "decode_time": decode_times,
        "queue_time": queue_times,
        "ttft": ttfts,
    })

    # Compute summary statistics for tokens and latencies
    sum_in_tokens   = np.sum(input_tokens)
    mean_in_tokens  = np.mean(input_tokens)
    max_in_tokens   = np.max(input_tokens)
    min_in_tokens   = np.min(input_tokens)
    p50_in_tokens   = np.percentile(input_tokens, 50)
    p95_in_tokens   = np.percentile(input_tokens, 95)

    sum_out_tokens  = np.sum(output_tokens)
    mean_out_tokens = np.mean(output_tokens)
    max_out_tokens  = np.max(output_tokens)
    min_out_tokens  = np.min(output_tokens)
    p50_out_tokens  = np.percentile(output_tokens, 50)
    p95_out_tokens  = np.percentile(output_tokens, 95)

    latency_total = end - start
    mean_latency = np.mean(latencies)
    max_latency  = np.max(latencies)
    min_latency  = np.min(latencies)
    p50_latency  = np.percentile(latencies, 50)
    p95_latency  = np.percentile(latencies, 95)

    mean_ttft = np.mean(ttfts)
    max_ttft  = np.max(ttfts)
    min_ttft  = np.min(ttfts)
    p50_ttft  = np.percentile(ttfts, 50)
    p95_ttft  = np.percentile(ttfts, 95)

    tp_total = (sum_in_tokens + sum_out_tokens) / latency_total
    out_tp_total = sum_out_tokens / latency_total
    in_tp_total = sum_in_tokens / latency_total

    # Compute energy efficiency per token (human‐readable)
    energy_tok_inout = 1000 * gpu_profile['total_energy'] / (sum_in_tokens + sum_out_tokens)
    energy_tok_out   = 1000 * gpu_profile['total_energy'] / sum_out_tokens
    energy_tok_in    = 1000 * gpu_profile['total_energy'] / sum_in_tokens

    if verbose:
        print("\n----------------Tokens-----------------------")
        print(f"In Tokens (total): {sum_in_tokens:.0f}")
        print(f"  Avg:  {mean_in_tokens:.2f},  P50: {p50_in_tokens:.0f},  P95: {p95_in_tokens:.0f},  Max: {max_in_tokens:.0f},  Min: {min_in_tokens:.0f}")
        print()
        print(f"Out Tokens (total): {sum_out_tokens:.0f}")
        print(f"  Avg:  {mean_out_tokens:.2f},  P50: {p50_out_tokens:.0f},  P95: {p95_out_tokens:.0f},  Max: {max_out_tokens:.0f},  Min: {min_out_tokens:.0f}")
        print()

        print("\n----------------Performance-----------------------")
        print(f"Latency (total): {latency_total:.3f} s")
        print(f"  Avg: {mean_latency:.3f} s,  P50: {p50_latency:.3f} s,  P95: {p95_latency:.3f} s,  Max: {max_latency:.3f} s,  Min: {min_latency:.3f} s")
        print()
        print(f"TTFT:  Avg: {mean_ttft:.3f} s,  P50: {p50_ttft:.3f} s,  P95: {p95_ttft:.3f} s,  Max: {max_ttft:.3f} s,  Min: {min_ttft:.3f} s")
        print()

        print(f"Throughput:")
        print(f"In + Out: {tp_total:.2f} tok/s")
        print(f"     Out: {out_tp_total:.2f} tok/s")
        print(f"     In : {in_tp_total:.2f} tok/s")
        print()

        print("\n----------------Efficiency-----------------------")
        print(f"Energy/Tok (in+out): {energy_tok_inout:.3f} J / 1000 tok")
        print(f"Energy/Tok (out):    {energy_tok_out:.3f} J / 1000 tok")
        print(f"Energy/Tok (in):     {energy_tok_in:.3f} J / 1000 tok")
        print()

    # Build a one‐row DataFrame for all GPU profiler metrics
    gpu_profile_df = pd.DataFrame({key: [value] for key, value in gpu_profile.items()})

    # Build a one‐row DataFrame for token/latency stats
    token_stats = {
        # Input tokens
        "In Tokens Total":       sum_in_tokens,
        "In Tokens Avg":         mean_in_tokens,
        "In Tokens P50":         p50_in_tokens,
        "In Tokens P95":         p95_in_tokens,
        "In Tokens Max":         max_in_tokens,
        "In Tokens Min":         min_in_tokens,

        # Output tokens
        "Out Tokens Total":      sum_out_tokens,
        "Out Tokens Avg":        mean_out_tokens,
        "Out Tokens P50":        p50_out_tokens,
        "Out Tokens P95":        p95_out_tokens,
        "Out Tokens Max":        max_out_tokens,
        "Out Tokens Min":        min_out_tokens,

        # Latency
        "Latency Total (s)":     latency_total,
        "Latency Avg (s)":       mean_latency,
        "Latency P50 (s)":       p50_latency,
        "Latency P95 (s)":       p95_latency,
        "Latency Max (s)":       max_latency,
        "Latency Min (s)":       min_latency,

        # TTFT
        "TTFT Avg (s)":          mean_ttft,
        "TTFT P50 (s)":          p50_ttft,
        "TTFT P95 (s)":          p95_ttft,
        "TTFT Max (s)":          max_ttft,
        "TTFT Min (s)":          min_ttft,

        # Throughput
        "Throughput (in+out) tok/s": tp_total,
        "Throughput (out) tok/s":    out_tp_total,
        "Throughput (in) tok/s":     in_tp_total,

        # Energy Efficiency
        "Energy/Tok (in+out) J/1000": energy_tok_inout,
        "Energy/Tok (out) J/1000":    energy_tok_out,
        "Energy/Tok (in) J/1000":     energy_tok_in,
    }
    token_stats_df = pd.DataFrame(token_stats, index=[0])

    # Concatenate all GPU profiler metrics with token/latency stats
    aggregated_data = pd.concat([gpu_profile_df, token_stats_df], axis=1)

    return latency_data, aggregated_data

# Legacy

def power_profile_task(task, interval, active_gpus, total_gpus, create_plot = False, rank = 0):
    inference_powers = []
    inference_powers_time = []
    power_avgs = []
    power_peaks = []
    energies = []
    power_profiles = []

    for id in range(total_gpus):
        power_profiles.append(gpuPowerProbe(interval=interval, gpu_id=id))
        power_profiles[id].start()

    start_time = time.perf_counter()
    task()
    end_time = time.perf_counter()

    latency = end_time - start_time

    for power_profile in power_profiles:
        power, times = power_profile.stop()
        inference_powers.append(power)
        inference_powers_time.append(times)
        power_profile.destroy()

    print("\n----------------Power-----------------------")
    for id in range(total_gpus):
        print(f"GPU {id}:")
        power = np.array(inference_powers[id]) / 1000  # to Watt
        times = np.array(inference_powers_time[id])
        avg_power = np.mean(power)
        peak_power = np.max(power)
        energy = np.sum(power*times)

        power_avgs.append(avg_power)
        power_peaks.append(peak_power)
        energies.append(energy)

        print(f"    Power avg : {avg_power :.3f} W")
        print(f"    Power peak: {peak_power :.3f} W")
        print(f"    Energy    : {energy :.3f} J")
        if create_plot:
            plt.plot(np.cumsum(inference_powers_time[id]), power, label=f'GPU {id}')

    if create_plot:
        plt.title("Power consumption")
        plt.legend()
        plt.xlabel(f'Time ({interval} sec intervals)')
        plt.ylabel('Power Consumption (W)')
        plt.savefig('gpu_power_plot.png')

    return latency, power_avgs, power_peaks, energies

def power_profile_task_verbose(task, interval, total_gpus=4, create_plot = False, rank = 0):
    """
    Profiles the power consumption of multiple GPUs while executing a task.

    Args:
        task: The function to profile.
        interval: The sampling interval for power measurement in seconds.
        total_gpus: The total number of GPUs to profile.
        create_plot: Not used in this version.
        rank: Not used in this version.

    Returns:
        A tuple containing:
            latency (float): The execution time of the task.
            power_avgs (list): List of average power consumption for each GPU (in Watts).
            power_peaks (list): List of peak power consumption for each GPU (in Watts).
            energies (list): List of total energy consumed by each GPU (in Joules).
            power_df (pd.DataFrame): DataFrame containing the raw power (W) and time (s) data for each GPU.
    """
    inference_powers = []
    inference_powers_time = []
    power_avgs = []
    power_peaks = []
    energies = []
    power_profiles = []

    # Start power profiling for each GPU
    for id in range(total_gpus):
        power_profiles.append(gpuPowerProbe(interval=interval, gpu_id=id))
        power_profiles[id].start()

    # Execute the task and measure its latency
    start_time = time.perf_counter()
    task() # Execute the task
    end_time = time.perf_counter()

    latency = end_time - start_time

    # Stop power profiling and collect data
    for power_profile in power_profiles:
        power, times = power_profile.stop()
        inference_powers.append(power)
        inference_powers_time.append(times)
        power_profile.destroy()

    print("\n----------------Power-----------------------")
    # Prepare data for pandas DataFrame and calculate metrics
    power_data_dict = {}
    for id in range(total_gpus):
        print(f"GPU {id}:")
        # Convert power from mW (assuming gpuPowerProbe returns mW) to Watt
        power = np.array(inference_powers[id]) / 1000
        times = np.array(inference_powers_time[id])

        # Store power and time data for each GPU in the dictionary
        # Ensure column names are unique and descriptive
        power_data_dict[f'GPU_{id}_Power_W'] = power
        power_data_dict[f'GPU_{id}_Time_s'] = times


        # Calculate average power, peak power, and energy
        avg_power = np.mean(power)
        peak_power = np.max(power)
        energy = np.sum(power * times) # Energy = sum(Power * delta_time)

        power_avgs.append(avg_power)
        power_peaks.append(peak_power)
        energies.append(energy)

        print(f"    Power avg : {avg_power :.3f} W")
        print(f"    Power peak: {peak_power :.3f} W")
        print(f"    Energy    : {energy :.3f} J")

    # Create pandas DataFrame
    # Ensure all lists have the same length by padding if necessary
    # This handles cases where sampling might stop slightly differently
    max_len = max((len(v) for v in power_data_dict.values()), default=0)
    for key in power_data_dict:
        # Pad shorter lists with NaN to match the longest list
        while len(power_data_dict[key]) < max_len:
            power_data_dict[key] = np.append(power_data_dict[key], np.nan) # Use np.append for numpy arrays


    power_df = pd.DataFrame(power_data_dict)


    # Return the original metrics and the new DataFrame
    return latency, power_avgs, power_peaks, energies, power_df
