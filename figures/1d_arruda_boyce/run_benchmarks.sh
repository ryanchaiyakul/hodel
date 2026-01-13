#!/bin/bash
methods=("hodel" "pinn" "deq" "node")
noise_levels=(0.01 0.03)
seeds=100
for m in "${methods[@]}"; do
    for n in "${noise_levels[@]}"; do
        echo "------------------------------------------------"
        echo "Running Method: $m | Noise: $n | Seeds: $seeds"
        echo "------------------------------------------------"
        uv run benchmark.py --method "$m" --eta "$n" --seeds "$seeds"
    done
done
echo "All benchmark runs completed!"
