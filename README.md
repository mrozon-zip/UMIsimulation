

# 📘 Barcode error correction - PCR vs polonies amplification

_This pipeline generates random barcode sequences, amplifies them by either PCR or polonies amplification manner,
denoises erroneous sequences and then conducts an analysis. A project is tailored for Master's Thesis purposes._

---

## 📂 Project Structure

```
PCR_simulation/
│
├── analysis/                   # Scripts for analysis module
├── archive/                    # Old scripts, parts of development process not used in final version
├── results_amplified/          # Output for amplified files
├── results_denoised/           # Output for denoised files
├── results_metrics/            # Stores analysis results - denoiser effectiveness metrics
├── venv/                       # Virtual environment
├── generate.py                 # Generates true barcodes
├── main.py                     # Runs all modules
├── pcr_amplifying.py           # PCR amplification simulator
├── polonies_amplifying.py      # Polonies amplification simulator
├── README.md                   # This file
├── requirements.txt            # File with dependencies of the project
├── run_all_1k_30.py            # Runs whole pipeline
└── support.py                  # Stores supporting functions
```

---

## 🔁 Pipeline Overview

_This section explains how the scripts orchestrate a single computational pipeline, step by step._

1. **`main.py`**  
   Arguments parser, runs scripts. Is called in `run_all_1k_30.py`. 

2. **`run_all_1k_30.py`**  
   Runs whole pipeline by parsing commands with different parameters.

3. **`generate.py`, `pcr_amplifying.py`, `polonies_amplifying.py`, `support.py` and scripts in `analysis/`**  
   Described as modules in the Master's Thesis. Store functions for the purpose described in the filename.

**!!! Important** - "Denoised module" is contained in the `support.py`.

---

## 🗂 Directory Contents

- **`analysis/`**  
    Scripts that constitute analysis module. Scripts that were used to analyse effectiveness of denoiser differences in 
polonies and PCR amplification simulators.
- **`archive/`**  
    Old scripts. Not necessarry for the pipeline to run properly but they contain some scripts that were used in the
development process. For curious minds to delve into.
- **`results_amplified/`**
    Output directory for amplification simulators. Files with "pcr" in name contain results for pcr amplification 
simulator, correspondingly for polonies.
- **`results_denoised/`**  
  Output directory for denoised files. Files that are produced by a denoiser.
- **`results_metrics/`**
    Output directory for denoiser effectiveness analysis script.

---

## 💻 Example Usage

The usage is simple. Just run `run_all_1k_30.py`. In that file it is possible to chang parameters for amplification 
simulators. If one prefers to run scripts manually, here is a couple of examples:

I want to generate 30 unique sequences that are of 30 nucleotide length. Let the output be "true_barcodes.csv":
```bash
python3 main.py generate --num 30 --length 30 --output true_barcodes.csv  
```

I want to amplify sequences that are in the "true_barcodes.csv" using PCR method. I want to simulate 30 cycles and 
have mutation rate of 0.005. I also want to set substrate capacity to certain value and have the result in "test1.csv".
```bash
python3 main.py amplify --method pcr --cycles 30 --mutation_rate 0.005 --no_simulate --substrate_capacity 282600 \
--no_plot --output test1.csv
```

I want to amplify the same sequences but using polonies amplification method, with corresponding parameters and
specific output filename.
```bash
python3 main.py amplify --method polonies_amplification --S_radius 30 --density 100 --deviation 0.05 \
 --mutation_rate 0.005 --AOE_radius 1 --success_prob 0.85 --no_simulate --no_plot --output test2.csv
```

I want to denoise file with amplified sequences. I also want to set threshold (threshold defined in the thesis) to 2.
```bash
python3 main.py denoise --input results_amplified/polonies_test.csv --treshold 2
```

Most of the analysis files can be run like that:
```bash
python3 analysis_file.py
```
Inputs of the files in that matter are to be set in the code.

---

## 🧪 Dependencies

Install required Python packages with:

```bash
pip install -r requirements.txt
```

---

## 📝 Thesis

You can find the thesis published at DiVA portal [here](https://www.diva-portal.org/smash/record.jsf?dswid=5474&pid=diva2%3A1970045&c=1&searchType=SIMPLE&language=en&query=%27Improving+DNA+barcode+error+handling+accuracy+in+network-based+spatial+transcriptomics&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all)

---

## 📫 Contact

For questions, reach out to [Krzysztof Mrozik] at [krzysm20009042@gmail.com].