import numpy as np
import time

# install pyyaml for better output
np.show_config()

n = 4096

x = np.random.randn(n, n).astype(np.float32)
y = np.random.randn(n, n).astype(np.float32)

# warm up
np.dot(x, y)

start = time.perf_counter_ns()
z = np.dot(x, y)
end = time.perf_counter_ns()

elapsed_ns = end - start

total_flops = 2 * n**3

gflops = total_flops / (elapsed_ns / 1e9) / 1e9

total_bytes = 3 * n * n * 4
total_gb = total_bytes / 1e9

print(f"Matrix size: {n}x{n}")
print(f"Total operations: {total_flops / 1e9:,.2f} GFLOP")
print(f"Total memory footprint: {total_gb:,.2f} GB")
print(f"Time taken matmul: {elapsed_ns / 1e6:.2f} ms")
print(f"Performance: {gflops:,.2f} GFLOP/s")

# Adjust this to your system ;D
print("\n" + "-" * 60)
print("System: AMD Ryzen 9 9950X, 16 cores @ 4.3GHz, DDR4-6000 CL30")

fp32_peak_tflops = 4.3e9 * 2 * 2 * 16 * 16 / 1e12
fp64_peak_tflops = 4.3e9 * 2 * 2 * 8 * 16 / 1e12

print("freq * FMA_per_cycle * 2 (flops-FMA) * n_elements * cores")
print(f"Theoretical peak FP32: {fp32_peak_tflops:.2f} TFLOP/s")
print(f"Theoretical peak FP64: {fp64_peak_tflops:.2f} TFLOP/s")
print("-" * 60)
