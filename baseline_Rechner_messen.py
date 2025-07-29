import psutil
import time

# 10 Messungen über 30 Sekunden
ram_values = []
cpu_values = []

for i in range(10):
    # RAM in MB
    ram_mb = psutil.virtual_memory().used / 1024 / 1024
    # CPU in %
    cpu_percent = psutil.cpu_percent(interval=1)
    
    ram_values.append(ram_mb)
    cpu_values.append(cpu_percent)
    
    time.sleep(3)  # 3 Sekunden warten

# Durchschnitt berechnen
baseline_ram = sum(ram_values) / len(ram_values)
baseline_cpu = sum(cpu_values) / len(cpu_values)

print(f"Baseline RAM: {baseline_ram:.1f} MB")
print(f"Baseline CPU: {baseline_cpu:.1f}%")