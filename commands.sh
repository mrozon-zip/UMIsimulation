#!/bin/bash

python3 main.py amplify --method bridge --S_radius 5 --density 5 --success_prob 0.85 --deviation 0.05 --mutation_rate 0.002 --output mut_0.002_Sr_5_dens_5_SP_0.85_dev_0.05.csv
python3 main.py amplify --method bridge --S_radius 7.5 --density 5 --success_prob 0.85 --deviation 0.05 --mutation_rate 0 --output mut_0_Sr_7.5_dens_5_SP_0.85_dev_0.05.csv
python3 main.py amplify --method bridge --S_radius 7.5 --density 5 --success_prob 0.85 --deviation 0.05 --mutation_rate 0.001 --output mut_0.001_Sr_7.5_dens_5_SP_0.85_dev_0.05.csv
python3 main.py amplify --method bridge --S_radius 7.5 --density 5 --success_prob 0.85 --deviation 0.05 --mutation_rate 0.002 --output mut_0.002_Sr_7.5_dens_5_SP_0.85_dev_0.05.csv