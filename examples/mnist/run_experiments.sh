#!/bin/bash



for mode in "xentropy" "soft_bootstrap" "hard_bootstrap"
do
    for noise_fraction in 0.3 0.34 0.38 0.42 0.46 0.5
    do
        python main.py run --mode=$mode --noise_fraction=$noise_fraction --num_epochs 30 --with_trains=${WITH_TRAINS:-False}
    done
done

# Run with as_pseudo_label=False
for noise_fraction in 0.3 0.34 0.38 0.42 0.46 0.5
do
    python main.py run --mode=soft_bootstrap --noise_fraction=$noise_fraction --num_epochs 30 --as_pseudo_label=False --with_trains=${WITH_TRAINS:-False}
done
