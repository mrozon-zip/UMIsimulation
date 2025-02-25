#!/bin/bash
python3 main.py amplify --method --pcr --mutation_rate 0.002 --output mutation_0.002_1 \
--no_plot
python3 main.py amplify --method --pcr --mutation_rate 0.005 --output mutation_0.005_1 \
--no_plot
python3 main.py amplify --method --pcr --mutation_rate 0.01 --output mutation_0.01 \
--no_plot

python3 main.py amplify --method --bridge --mutation_rate 0 --S_radius 10 --density 10 \
--no_plot --output no_mutation_1
python3 main.py amplify --method --bridge --mutation_rate 0.001 --S_radius 10 --density 10 \
--no_plot --output mutation_0.001_1
python3 main.py amplify --method --bridge --mutation_rate 0.002 --S_radius 10 --density 10 \
--no_plot --output mutation_0.002_1
python3 main.py amplify --method --bridge --mutation_rate 0.005 --S_radius 10 --density 10 \
--no_plot --output mutation_0.005_1
python3 main.py amplify --method --bridge --mutation_rate 0.01 --S_radius 10 --density 10 \
--no_plot --output mutation_0.01_1

python3 main.py amplify --method --bridge --mutation_rate 0 --S_radius 100 --density 15 \
--no_plot --output no_mutation_2
python3 main.py amplify --method --bridge --mutation_rate 0.001 --S_radius 100 --density 15 \
--no_plot --output mutation_0.001_2
python3 main.py amplify --method --bridge --mutation_rate 0.002 --S_radius 100 --density 15 \
--no_plot --output mutation_0.002_2
python3 main.py amplify --method --bridge --mutation_rate 0.005 --S_radius 100 --density 15 \
--no_plot --output mutation_0.005_2
python3 main.py amplify --method --bridge --mutation_rate 0.01 --S_radius 100 --density 15 \
--no_plot --output mutation_0.01_2

python3 main.py amplify --method --polonies_amplification --mutation_rate 0 --S_radius 10 --density 10 \
--no_plot --output no_mutation_1
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.001 --S_radius 10 --density 10 \
--no_plot --output mutation_0.001_1
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.002 --S_radius 10 --density 10 \
--no_plot --output mutation_0.002_1
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.005 --S_radius 10 --density 10 \
--no_plot --output mutation_0.005_1
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.01 --S_radius 10 --density 10 \
--no_plot --output mutation_0.01_1

python3 main.py amplify --method --polonies_amplification --mutation_rate 0 --S_radius 10 --density 15 \
--no_plot --output no_mutation_2
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.001 --S_radius 10 --density 15 \
--no_plot --output mutation_0.001_2
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.002 --S_radius 10 --density 15 \
--no_plot --output mutation_0.002_2
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.005 --S_radius 10 --density 15 \
--no_plot --output mutation_0.005_2
python3 main.py amplify --method --polonies_amplification --mutation_rate 0.01 --S_radius 10 --density 15 \
--no_plot --output mutation_0.01_2