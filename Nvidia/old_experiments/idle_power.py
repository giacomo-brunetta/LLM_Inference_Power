import time
from LLM_Inference_Power.Nvidia.profiler_utils import power_profile_task

def do_nothing():
    time.sleep(60)

latency, power_avgs, power_peaks, energies = power_profile_task(do_nothing,0.5,4, 4,True,0)

print("Latency: ", latency)
print("Power Avgs: ", power_avgs)
print("Power Peaks: ", power_peaks)
print("Energies: ", energies)