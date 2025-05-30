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

class gpuPowerProbe(object):
    """
    From
    File: power_utils.py
    Author: Farah Ferdaus
    Email: fferdaus@anl.gov
    Last updated: Jul 22, 2024
    Description: A power analysis routine using built-in power monitoring tools provided by the vendor.
    """
    def __init__(self, interval, gpu_id=-1):
        self.interval = multiprocessing.Value('d', interval)
        self.len = int(7200/interval)
        self.powers = multiprocessing.Array('d', self.len)
        self.times = multiprocessing.Array('d', self.len)
        self.gpu_id = multiprocessing.Value('i', gpu_id)  
        self.process = None
        self.prevTime = multiprocessing.Value('d',time.time())
        self.halt = multiprocessing.Value('i',1)  
        self.count = multiprocessing.Value('i',0)
        self.isrunning = multiprocessing.Value('i',0)
        self.alive = multiprocessing.Value('i',0)  
        self.init()

    def _getGpuPower(self, powers, times, gpu_id, count, halt, alive, isrunning, prevTime, interval):
        nvmlInit()
        while (alive.value):
            while (not halt.value):
                isrunning.value = 1
                if gpu_id.value > -1:
                    power = nvmlDeviceGetPowerUsage(nvmlDeviceGetHandleByIndex(gpu_id.value))
                else:
                    power = 0
                    num_gpus = nvmlDeviceGetCount()
                    for i in range(num_gpus):
                        power += nvmlDeviceGetPowerUsage(nvmlDeviceGetHandleByIndex(i))
                    
                new_time = time.time()
                while (new_time-prevTime.value < interval.value):
                    new_time = time.time()
                powers[count.value] = power
                times[count.value] = new_time-prevTime.value
                count.value += 1
                prevTime.value = new_time
                isrunning.value = 0
        nvmlShutdown()
        
    def init(self):
        self.halt.value = 1
        self.alive.value = 1
        self.process = multiprocessing.Process(target = self._getGpuPower, args = (self.powers, self.times, self.gpu_id,
                self.count, self.halt, self.alive, self.isrunning, self.prevTime, self.interval))
        self.process.start()

    def start(self):  
        self.count.value = 0
        self.prevTime.value = time.time()
        self.halt.value = 0

    def stop(self):
        self.halt.value = 1
        while (self.isrunning.value):
            pass
        return self.powers[:self.count.value], self.times[:self.count.value]
    
    def destroy(self):
        self.alive.value = 0
        self.process.join()

class PowerProfiler:
    def __init__(self, interval=0.5, gpus = 1, active_gpus = 1):
        self.gpus = gpus
        self.active_gpus = active_gpus
        self.inference_powers = []
        self.inference_powers_time = []

        self.power_profiles = [gpuPowerProbe(interval=interval, gpu_id=id) for id in range(gpus)]

    def start(self):
        for power_profile in self.power_profiles:
            power_profile.start()

    def stop(self):
        for power_profile in self.power_profiles:
            power, times = power_profile.stop()
            self.inference_powers.append(power)
            self.inference_powers_time.append(times)
            power_profile.destroy()

    def calculate_metrics(self, verbose=True):
        power_avgs = []
        power_peaks = []
        energies = []
        total_power = None
        active_power = None
        if verbose:
            print("\n----------------Power-----------------------")


        min_sample_num = min([len(power) for power in self.inference_powers])

        for id in range(self.gpus):
            print(f"GPU {id}:")
            power = np.array(self.inference_powers[id]) / 1000  # to Watt
            times = np.array(self.inference_powers_time[id])
            avg_power = np.mean(power)
            peak_power = np.max(power)
            energy = np.sum(power*times)

            if total_power is None:
                total_power = power[0:min_sample_num]
                active_power = power[0:min_sample_num]

            elif id < self.active_gpus:
                active_power += power[0:min_sample_num]
                total_power += power[0:min_sample_num]

            else:
                total_power += power[0:min_sample_num]
        
            power_avgs.append(avg_power)
            power_peaks.append(peak_power)
            energies.append(energy)

            if verbose:
                print(f"    Power avg : {avg_power :.3f} W")
                print(f"    Power peak: {peak_power :.3f} W")
                print(f"    Energy    : {energy :.3f} J")
            
        
        active_energy = sum(energies[0:self.active_gpus])
        total_energy = sum(energies)
        active_power_avg = sum(power_avgs[0:self.active_gpus])
        avg_total_power = sum(power_avgs)
        total_power_peak = max(total_power)
        active_power_peak = max(active_power)
        
        if verbose:
            print(f"Total:")
            print(f"    Power avg : {avg_total_power :.3f} W")
            print(f"    Power peak: {total_power_peak :.3f} W")
            print(f"    Energy    : {total_energy :.3f} J")
            print(f"Active GPUs:")
            print(f"    Power avg : {active_power_avg :.3f} W")
            print(f"    Power peak: {active_power_peak :.3f} W")
            print(f"    Energy    : {active_energy :.3f} J")
            print()

        return  active_power_avg, avg_total_power, total_energy, active_energy, total_power_peak, active_power_peak

def metrics(results, power_profiler, verbose=True):

    # Get the power data
    active_power_avg, avg_total_power, total_energy, active_energy, total_power_peak, active_power_peak = power_profiler.calculate_metrics(verbose)
    
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

    # Get token count, latency, and ttft for every response
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

    if verbose:
        print("\n----------------Tokens-----------------------")
        print(f"In  Tokens : {np.sum(input_tokens):.3f} (total)")
        print(f"             {np.mean(input_tokens):.3f} (avg)")
        print(f"             {np.max(input_tokens):.3f} (max)")
        print(f"             {np.min(input_tokens):.3f} (min)")
        print()
        print(f"Out Tokens: {np.sum(output_tokens):.3f} (total)")
        print(f"            {np.mean(output_tokens):.3f} (avg)")
        print(f"            {np.max(output_tokens):.3f} (max)")
        print(f"            {np.min(output_tokens):.3f} (min)")
        print()

    latency_total = end - start
    ttfts = np.array(ttfts)
    latencies = np.array(latencies)
    latencies_with_queue = np.array(latencies_with_queue)
    decode_times = np.array(decode_times)
    queue_times = np.array(queue_times)

    if verbose:
        print("\n----------------Performance-----------------------")
        print(f"Latency:    {latency_total:.3f} s (total)")
        print(f"            {np.mean(latencies):.3f} s (avg)")
        print(f"            {np.max(latencies):.3f} s (max)")
        print(f"            {np.min(latencies):.3f} s (min)")
        print()
        print(f"TTFT:       {np.mean(ttfts):.3f} s (mean)")
        print(f"            {np.max(ttfts):.3f} s (max)")
        print(f"            {np.min(ttfts):.3f} s (min)")
        print()

    tp_total = (np.sum(input_tokens) + np.sum(output_tokens)) / latency_total
    out_tp_total = np.sum(output_tokens) / latency_total
    in_tp_total = np.sum(input_tokens)  / latency_total

    if verbose:
        print(f"Throughput: {tp_total:.3f} tok/s (in + out)")
        print(f"  total     {out_tp_total:.3f} tok/s (out)")
        print(f"            {in_tp_total:.3f} tok/s (in)")
        print()

    # Datagrame with latencies for each response
    latency_data =  pd.DataFrame({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latencies,
        "latency_with_queue": latencies_with_queue,
        "decode_time": decode_times,
        "queue_time": queue_times,
        "ttft": ttfts,
    })

    print("\n----------------Efficiency-----------------------")
    print(f"Energy/Tok: {(1000*total_energy/(np.sum(latency_data['input_tokens']) + np.sum(latency_data['output_tokens']))):.3f} J / 1000 tok (in + out)")
    print(f"            {1000*total_energy/np.sum(latency_data['output_tokens']):.3f} J / 1000 tok (out)")
    print(f"            {1000*total_energy/np.sum(latency_data['input_tokens']):.3f} J / 1000 tok (in)")
    print()

    aggregated_data = pd.DataFrame({
        "Total Avg Power (W)": [avg_total_power],
        "Active Avg Power (W)": [active_power_avg],
        "Total Peak Power (W)": [total_power_peak],
        "Active Peak Power (W)": [active_power_peak],
        "Total Energy (J)": [total_energy],
        "Active Energy (J)": [active_energy],

        "Total Input Tokens": [np.sum(input_tokens)],
        "Avg Input Tokens": [np.mean(input_tokens)],
        "Max Input Tokens": [np.max(input_tokens)],
        "Min Input Tokens": [np.min(input_tokens)],

        "Total Output Tokens": [np.sum(output_tokens)],
        "Avg Output Tokens": [np.mean(output_tokens)],
        "Max Output Tokens": [np.max(output_tokens)],
        "Min Output Tokens": [np.min(output_tokens)],

        "Total Latency (data['s)": [latency_total],
        "Avg Latency (s)": [np.mean(latencies)],
        "Max Latency (s)": [np.max(latencies)],
        "Min Latency (s)": [np.min(latencies)],

        "Mean TTFT (s)": [np.mean(ttfts)],
        "Max TTFT (s)": [np.max(ttfts)],
        "Min TTFT (s)": [np.min(ttfts)],

        "Total Throughput (tok/s) (in+out)": [tp_total],
        "Total Throughput (tok/s) (out)": [out_tp_total],
        "Total Throughput (tok/s) (in)": [in_tp_total],

        "Energy/Tok (J/1000tok) (in+out)": [(1000*total_energy/(np.sum(input_tokens) + np.sum(output_tokens)))],
        "Energy/Tok (J/1000tok) (out)": [(1000*total_energy/np.sum(output_tokens))],
        "Energy/Tok (J/1000tok) (in)": [(1000*total_energy/np.sum(input_tokens))]
    })
    
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