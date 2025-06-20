import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import importlib.util
import numpy as np

class GPUProfiler:
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

            # Overall aggregated metrics
            active_power_avg,    active_power_peak,    active_power_p50,    active_power_p95    = self.get_stats(active_power)
            total_power_avg,     total_power_peak,     total_power_p50,     total_power_p95     = self.get_stats(total_power)
            active_mem_avg,      active_mem_peak,      active_mem_p50,      active_mem_p95      = self.get_stats(active_mem_used)
            total_mem_avg,       total_mem_peak,       total_mem_p50,       total_mem_p95       = self.get_stats(total_mem_used)
            active_gpu_util_avg, active_gpu_util_peak, active_gpu_util_p50, active_gpu_util_p95 = self.get_stats(active_gpu_util / self.active_gpus)
            total_gpu_util_avg,  total_gpu_util_peak,  total_gpu_util_p50,  total_gpu_util_p95  = self.get_stats(total_gpu_util / self.gpus) 

            if verbose:
                print()
                print(f"Overall Active (first {self.active_gpus} GPU(s)):")
                print(f"    Power avg      : {active_power_avg: .3f} W")
                print(f"    Power peak     : {active_power_peak: .3f} W")
                print(f"    Energy         : {active_energy: .3f} J")
                print(f"    Memory avg     : {active_mem_avg: .3f} MiB")
                print(f"    Memory peak    : {active_mem_peak: .3f} MiB")
                print(f"    GPU util avg   : {active_gpu_util_avg: .2f} %")
                print(f"    GPU util peak  : {active_gpu_util_peak: .2f} %")
                print()
                print("Overall Total (all GPUs):")
                print(f"    Power avg      : {total_power_avg: .3f} W")
                print(f"    Power peak     : {total_power_peak: .3f} W")
                print(f"    Energy         : {total_energy: .3f} J")
                print(f"    Memory avg     : {total_mem_avg: .3f} MiB")
                print(f"    Memory peak    : {total_mem_peak: .3f} MiB")
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
                "active_power_p50" : active_power_p50,
                "active_power_p95" : active_power_p95,

                # Power (active + idle GPUs)
                "total_power_avg"  : total_power_avg,
                "total_power_peak" : total_power_peak,
                "total_power_p50"  : total_power_p50,
                "total_power_p95"  : total_power_p95,

                # Memory (active GPUs)
                "active_gpu_util_avg" : active_gpu_util_avg,
                "active_gpu_util_peak": active_gpu_util_peak,
                "active_gpu_util_p50" : active_gpu_util_p50,
                "active_gpu_util_p95" : active_gpu_util_p95,

                # Memory (active GPUs)
                "active_mem_avg"     : active_mem_avg,
                "active_mem_peak"    : active_mem_peak,
                "active_mem_p50"     : active_mem_p50,
                "active_mem_p95"     : active_mem_p95,

                # Memory (active + idle GPUs)
                "total_mem_avg"      : total_mem_avg,
                "total_mem_peak"     : total_mem_peak,
                "total_mem_p50"      : total_mem_p50,
                "total_mem_p95"      : total_mem_p95,

                # GPU Utilization (active GPUs)
                "total_gpu_util_avg" : total_gpu_util_avg,
                "total_gpu_util_peak": total_gpu_util_peak,
                "total_gpu_util_p50" : total_gpu_util_p50,
                "total_gpu_util_p95" : total_gpu_util_p95,
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

