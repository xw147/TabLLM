#!/bin/bash

# ICO fraud detection - Zero-shot only (no fine-tuning)
dataset="ico"
variants="list"

for variant in $variants; do
    for seed in 13 21 42 87 100; do
        echo "Running ${dataset}_${variant} zero-shot, seed ${seed}"
        
        python -m src.models.run \
            --config_file configs/global.json \
            --model_config_file configs/t03b.json \
            --adapter_config_file configs/ia3.json \
            --dataset ${dataset}_${variant} \
            --num_shot 0 \
            --few_shot_random_seed $seed \
            --output_dir outputs/ico_${variant}_zeroshot_seed${seed}
    done
done