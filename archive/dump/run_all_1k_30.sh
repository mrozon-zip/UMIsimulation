#!/bin/bash

# Example for PCR method:
# PCR has additional parameter --cycles with values 15, 25, and 30.
for cycles in 15 25 30; do
  for mutation_rate in 0.002 0.005 0.01; do
    python3 main.py amplify --method "pcr" \
      --mutation_rate "$mutation_rate" \
      --cycles "$cycles" \
      --output "mutation_${mutation_rate}_${cycles}" \
      --no_plot
  done
done

# Example for Bridge method:
# Bridge has parameters --S_radius, --density, --success_prob, and --deviation.
for s_radius in 5 10 15; do
  for density in 10 20 50; do
    for success_prob in 0.80 0.85 0.9; do
      for deviation in 0.01 0.05 0.1; do
        for mutation_rate in 0 0.001 0.002 0.005 0.01; do
          python3 main.py amplify --method "bridge" \
            --mutation_rate "$mutation_rate" \
            --S_radius "$s_radius" \
            --density "$density" \
            --success_prob "$success_prob" \
            --deviation "$deviation" \
            --no_simulate \
            --no_plot \
            --output "bridge_mut_${mutation_rate}_Sr${s_radius}_dens${density}_SP${success_prob}_dev${deviation}.csv"
        done
      done
    done
  done
done

# Example for Polonies Amplification method:
# Polonies has similar parameters as bridge but with different S_radius values.
for s_radius in 5000 10000 15000; do
  for density in 10 20 50; do
    for success_prob in 0.80 0.85 0.9; do
      for deviation in 0.01 0.05 0.1; do
        for mutation_rate in 0 0.001 0.002 0.005 0.01; do
          python3 main.py amplify --method "polonies_amplification" \
            --mutation_rate "$mutation_rate" \
            --S_radius "$s_radius" \
            --density "$density" \
            --success_prob "$success_prob" \
            --deviation "$deviation" \
            --no_simulate \
            --no_plot \
            --output "polonies_mut_${mutation_rate}_Sr${s_radius}_dens${density}_SP${success_prob}_dev${deviation}.csv"
        done
      done
    done
  done
done

python3 hamming_distance.py results/*.csv --metric both