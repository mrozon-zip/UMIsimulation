import random
from typing import List, Dict
import csv
import logging

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def generate_sequences(num: int, length: int, unique: bool, output_filename: str) -> List[Dict[str, any]]:
    """
    Generate a list of random sequences of given length.
    Each sequence is a dictionary with keys:
      - 'sequence': The nucleotide string.
      - 'N0': Initial copy number (set to 1).
    The sequences are written to a CSV file.
    """
    sequences = []
    if unique:
        seq_set = set()
        while len(seq_set) < num:
            seq = ''.join(random.choices(NUCLEOTIDES, k=length))
            seq_set.add(seq)
        for seq in seq_set:
            sequences.append({'sequence': seq, 'N0': 1})
    else:
        for _ in range(num):
            seq = ''.join(random.choices(NUCLEOTIDES, k=length))
            sequences.append({'sequence': seq, 'N0': 1})

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['sequence', 'N0']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for seq_dict in sequences:
            writer.writerow(seq_dict)
    logging.info(f"Generated {len(sequences)} sequences and saved to {output_filename}")
    return sequences

