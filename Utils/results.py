import numpy as np
import pandas as pd
import torch
import os

# Merge engine and profiler metrics

def get_stats(array):
    return np.mean(array), np.max(array), np.percentile(array, 50), np.percentile(array, 95)

def get_metrics_from_profiler(profiler, verbose=True):
    
    active_power, total_power, active_mem_used, total_mem_used, active_gpu_util, total_gpu_util, active_energy, total_energy = profiler.calculate_metrics()

    active_power_avg,    active_power_peak,    active_power_p50,    active_power_p95    = get_stats(active_power)
    total_power_avg,     total_power_peak,     total_power_p50,     total_power_p95     = get_stats(total_power)
    active_mem_avg,      active_mem_peak,      active_mem_p50,      active_mem_p95      = get_stats(active_mem_used)
    total_mem_avg,       total_mem_peak,       total_mem_p50,       total_mem_p95       = get_stats(total_mem_used)
    active_gpu_util_avg, active_gpu_util_peak, active_gpu_util_p50, active_gpu_util_p95 = get_stats(active_gpu_util / profiler.active_gpus)
    total_gpu_util_avg,  total_gpu_util_peak,  total_gpu_util_p50,  total_gpu_util_p95  = get_stats(total_gpu_util / profiler.gpus) 

    if verbose:
        print()
        print(f"Overall Active (first {profiler.active_gpus} GPU(s)):")
        print(f"    Power avg      : {active_power_avg: .3f} W")
        print(f"    Power peak     : {active_power_peak: .3f} W")
        print(f"    Energy         : {active_energy: .3f} J")
        print(f"    Memory avg     : {active_mem_avg: .3f} MiB")
        print(f"    Memory peak    : {active_mem_peak: .3f} MiB")
        print(f"    GPU util avg   : {active_gpu_util_avg: .2f} %")
        print(f"    GPU util peak  : {active_gpu_util_peak: .2f} %")
        print()
        print(f"Overall Total (all {profiler.gpus} GPUs):")
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
    }

def get_latency_data(results, verbose=True):
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

    latency_total = end - start

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

        print(f"Throughput:    (In+Out) : {tp_total:.2f} tok/s,   Out: {out_tp_total:.2f} tok/s,   In : {in_tp_total:.2f} tok/s")
        print()

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
    }
    token_stats = pd.DataFrame(token_stats, index=[0])

    return latency_data, token_stats

def metrics(results, gpu_profiler, verbose=True):
    # Get all GPU profiler metrics
    gpu_profile = get_metrics_from_profiler(gpu_profiler)

    latency_data, token_stats = get_latency_data(results)

    # Compute energy efficiency per token (human‐readable)
    sum_in_tokens = token_stats["In Tokens Total"][0]
    sum_out_tokens = token_stats["Out Tokens Total"][0]

    energy_tok_inout = 1000 * gpu_profile['total_energy'] / (sum_in_tokens + sum_out_tokens)
    energy_tok_out   = 1000 * gpu_profile['total_energy'] / sum_out_tokens
    energy_tok_in    = 1000 * gpu_profile['total_energy'] / sum_in_tokens

    if verbose:
        print()

        print("\n----------------Efficiency-----------------------")
        print(f"Energy/Tok (in+out): {energy_tok_inout:.3f} J / 1000 tok")
        print(f"Energy/Tok (out):    {energy_tok_out:.3f} J / 1000 tok")
        print(f"Energy/Tok (in):     {energy_tok_in:.3f} J / 1000 tok")
        print()
    
    # Build a one‐row DataFrame for all GPU profiler metrics
    gpu_profile_df = pd.DataFrame({key: [value] for key, value in gpu_profile.items()})

    # Build a one‐row DataFrame for token/latency stats
    token_stats["Energy/Tok (in+out) J/1000"] = energy_tok_inout
    token_stats["Energy/Tok (out) J/1000"] = energy_tok_out
    token_stats["Energy/Tok (in) J/1000"] = energy_tok_in

    token_stats_df = pd.DataFrame(token_stats, index=[0])

    # Concatenate all GPU profiler metrics with token/latency stats
    aggregated_data = pd.concat([gpu_profile_df, token_stats_df], axis=1)

    return latency_data, aggregated_data

# Save results to CSV

def save_results(args, aggregated_data, path, batch_size = None):

    aggregated_data['Model Name'] = args.model_name
    aggregated_data['FrameWork'] = 'vLLM'
    aggregated_data['Hardware type'] = torch.cuda.get_device_name(torch.cuda.current_device())
    aggregated_data['TP Size'] = 1 if args.expert_parallel else args.tensor_parallel
    aggregated_data['PP Size'] = 1 if args.expert_parallel else args.pipeline_parallel
    aggregated_data['EP Size'] = args.tensor_parallel * args.pipeline_parallel if args.expert_parallel else 1
    aggregated_data['DP Size'] = args.data_parallel
    aggregated_data['Precision'] = args.data_type

    if batch_size is not None:
        aggregated_data['Batch Size'] = batch_size
    else:
        aggregated_data['Batch Size'] = args.batch_size

    if os.path.exists(path):
        aggregated_data.to_csv(path, mode='a', index=False, header=False)
    else:
        aggregated_data.to_csv(path, mode='w', index=False, header=True)

def save_results_with_power(model_name, framework, hw, num, dtype, batch_size, in_len, out_len, ttft, latency, power_avg, active_power_avg, power_peak, active_power_peak, energy, active_energy):
    data = {
        'Model Name': [model_name],
        'FrameWork': [framework],
        'Hardware type': [hw],
        'Count': [num],
        'Precision': [dtype],
        'Batch Size': [batch_size],
        'In tokens': [in_len],
        'Out tokens': [out_len],
        'TTFT': [ttft * 1000],  # Convert TTFT to ms
        'Latency': [latency],
        'Power Avg': [power_avg],
        'Power Avg (Active)': [active_power_avg],
        'Power Peak': [power_peak],
        'Power Peak (Active)': [active_power_peak],
        'Energy': [energy],
        'Energy (Active)': [active_energy]
    }

    new_data_df = pd.DataFrame(data)
    file_path = '../Results/results.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_data_df], ignore_index=True)
    else:
        df = new_data_df

    df.to_csv(file_path, index=False)
