import logging
from support import compute_global_p, process_mutation
import random
from typing import List, Dict, Tuple, Any
import os


def pcr_amplification(sequences: List[Dict[str, Any]],
                      cycles: int,
                      mutation_rate: float,
                      mutation_probabilities: Dict[str, float],
                      substrate_capacity_initial: float,
                      s: float,
                      c: float,
                      output: str) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """
    Perform PCR amplification simulation.
    For each cycle:
      - Compute global amplification probability based on current total copies and substrate capacity.
      - For each sequence that passes the amplification check, replicate as many times as its current N0.
      - For each replication event, process nucleotide mutations.
      - If no mutation occurs, increment N0; if mutation occurs, add a new sequence with N0 = 1.
      - Update the substrate capacity by subtracting the newly created copies.
      - Record the total number of unique sequences at the end of the cycle.
    Returns the final sequence list and a list of total sequence counts per cycle.
    """
    total_sequences_history = []
    remaining_substrate = substrate_capacity_initial
    k = s * 10
    for cycle in range(1, cycles + 1):
        logging.info(f"PCR Amplification: Cycle {cycle} starting with {len(sequences)} sequences.")
        current_n = sum(seq['N0'] for seq in sequences)
        p = compute_global_p(current_n, remaining_substrate, substrate_capacity_initial, s, k, c)
        logging.info(f"Cycle {cycle}: Global amplification probability = {p:.4f}")
        new_sequences = []
        for seq_dict in sequences:
            if random.random() < p:
                replication_count = seq_dict['N0']
                no_mutation_count = 0
                for _ in range(replication_count):
                    mutated_seq, mutation_occurred = process_mutation(
                        seq_dict['sequence'],
                        mutation_rate,
                        mutation_probabilities
                    )
                    if mutation_occurred:
                        new_sequences.append({'sequence': mutated_seq, 'N0': 1})
                    else:
                        no_mutation_count += 1
                seq_dict['N0'] += no_mutation_count
        sequences.extend(new_sequences)
        new_total = sum(seq['N0'] for seq in sequences)
        delta_n = new_total - current_n
        remaining_substrate = max(0, remaining_substrate - delta_n)
        total_sequences_history.append(new_total)
        logging.info(
            f"Cycle {cycle} complete. Total unique sequences: {len(sequences)}; Remaining substrate: {remaining_substrate}"
        )
    base, ext = os.path.splitext(output)
    pcr_output = f"results1/pcr_{base}{ext}"
    return sequences, total_sequences_history, pcr_output