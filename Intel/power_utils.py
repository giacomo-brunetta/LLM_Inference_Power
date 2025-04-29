import subprocess
import time
import re
import threading
import numpy as np
import matplotlib.pyplot as plt

def get_gpu_power(device_id):
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", f"-d{device_id}"],
            capture_output=True,
            text=True
        )
        output = result.stdout

        # Regex pattern to extract GPU power for Tile 0 and Tile 1
        power_pattern = r"GPU Power \(W\)\s*\|\s*Tile 0:\s*(\d+);\s*Tile 1:\s*(\d+)"
        match = re.search(power_pattern, output)

        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
def power_probe(device_id, stop_event, tile1_pow, tile2_pow, intervals):
    while not stop_event.is_set():
        begin = time.time()
        pow1, pow2 = get_gpu_power(device_id)
        interval = time.time() - begin
        tile1_pow.append(pow1)
        tile2_pow.append(pow2)
        intervals.append(interval)

def power_profile_task(func, duration, total_xpus):
    inference_powers = []
    inference_powers_time = []
    power_probing_threads = []
    
    stop = threading.Event()  # Create the stopping event

    # Create and start power probing threads
    for xpu in range(total_xpus):
        tile1_pow = []
        tile2_pow = []
        intervals = []
        inference_powers.append(tile1_pow)
        inference_powers.append(tile2_pow)
        inference_powers_time.append(intervals)
        
        thread = threading.Thread(target=power_probe, args=(xpu, stop, tile1_pow, tile2_pow, intervals))
        power_probing_threads.append(thread)
        thread.start()

    start_time = time.perf_counter()
    latency = None

    while time.perf_counter() - start_time < duration:
        func()  # Run the provided function for duration seconds

        if latency is None:
            latency = time.perf_counter() - start_time

    stop.set()  # Stop power probing threads

    for thread in power_probing_threads:
        thread.join()

    power_avgs = []
    power_peaks = []
    energies = []

    print("\n----------------Power-----------------------")
    for id in range(total_xpus):

        print(f"GPU {id}:")
        power1 = np.array(inference_powers[id*2])
        power2 = np.array(inference_powers[id*2 +1])
        times = np.array(inference_powers_time[id])

        avg_power1 = np.mean(power1)
        peak_power1 = np.max(power1)
        energy1 = np.sum(power1*times)

        avg_power2 = np.mean(power2)
        peak_power2 = np.max(power2)
        energy2 = np.sum(power2*times)

        power_avgs.append(avg_power1)
        power_avgs.append(avg_power2)
        power_peaks.append(peak_power1)
        power_peaks.append(peak_power2)
        energies.append(energy1)
        energies.append(energy2)

        print("  Tile 0:")
        print(f"    Power avg : {avg_power1 :.3f} W")
        print(f"    Power peak: {peak_power1 :.3f} W")
        print(f"    Energy    : {energy1 :.3f} J")

        print("  Tile 1:")
        print(f"    Power avg : {avg_power2 :.3f} W")
        print(f"    Power peak: {peak_power2 :.3f} W")
        print(f"    Energy    : {energy2 :.3f} J")
    
    return latency, power_avgs, power_peaks, energies