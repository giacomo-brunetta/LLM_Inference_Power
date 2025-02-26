import subprocess
import time
import re

def get_gpu_power(device_id, tile):
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", f"-d", str(device_id)],
            capture_output=True,
            text=True
        )
        output = result.stdout
        
        # Regex to extract GPU Power for the specified tile
        power_pattern = rf"GPU Power \(W\)\s+\| Tile {tile}: (\d+)"
        match = re.search(power_pattern, output)
        if match:
            return int(match.group(1))
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def power_profile_task(func, duration, interval, device_id=0, tile=0):
    start_time = time.time()
    power_readings = []
    
    while time.time() - start_time < duration:
        func()  # Run the provided function
        power = get_gpu_power(device_id, tile)
        if power is not None:
            power_readings.append(power)
        time.sleep(interval)
    
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
    print(f"Average Power Consumption: {avg_power:.2f} W")