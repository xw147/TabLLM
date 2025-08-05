#!/bin/bash

# ICO fraud detection experiments only
dataset="ico"
variants="list"  # You can add more variants like "list_permuted" if needed

for variant in $variants; do
    for shots in 0 4 8 16 32; do
        for seed in 13 21 42 87 100; do
            echo "Running ${dataset}_${variant} with ${shots} shots, seed ${seed}"
            
            python -m src.models.run \
                --config_file configs/global.json \
                --model_config_file configs/t03b.json \
                --adapter_config_file configs/ia3.json \
                --dataset ${dataset}_${variant} \
                --num_shot $shots \
                --few_shot_random_seed $seed \
                --output_dir outputs/ico_${variant}_${shots}shot_seed${seed}
        done
    done
done