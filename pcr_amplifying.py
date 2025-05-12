import logging
from support import compute_global_p, process_mutation
import random
from typing import List, Dict, Tuple, Any
import os

base_folder = "/Users/krzysztofmrozik/Desktop/SciLifeLab/Projects/PCR simulation/"

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
    total_possible_ids = substrate_capacity_initial + len(sequences)
    available_ids = list(range(total_possible_ids))
    random.shuffle(available_ids)
    for sequence in sequences:
        sequence['id'] = available_ids.pop()
        sequence['parent_id'] = ''
        sequence['born'] = 0
        sequence['active'] = 0
    k = s * 10
    for cycle in range(1, cycles + 1):
        logging.info(f"PCR Amplification: Cycle {cycle} starting with {len(sequences)} sequences.")
        current_n = sum(seq['N0'] for seq in sequences)
        p = compute_global_p(current_n, remaining_substrate, substrate_capacity_initial, s, k, c)
        logging.info(f"Cycle {cycle}: Global amplification probability = {p:.4f}")
        new_sequences = []
        if remaining_substrate < len(sequences):
            for seq_dict in sequences[:remaining_substrate]:
                if random.random() < p:
                    mutated_seq, mutation_occurred = process_mutation(
                        seq_dict['sequence'],
                        mutation_rate,
                        mutation_probabilities
                    )
                    new_sequences.append({'sequence': mutated_seq,
                                          'N0': 1,
                                          'id': available_ids.pop(),
                                          'parent_id': seq_dict["id"],
                                          'born': cycle,
                                          'active': 0})
                seq_dict['active'] += 1
        else:
            for seq_dict in sequences:
                if random.random() < p:
                    mutated_seq, mutation_occurred = process_mutation(
                        seq_dict['sequence'],
                        mutation_rate,
                        mutation_probabilities
                    )
                    new_sequences.append({'sequence': mutated_seq,
                                          'N0': 1,
                                          'id': available_ids.pop(),
                                          'parent_id': seq_dict["id"],
                                          'born': cycle,
                                          'active': 0
                                          })
                seq_dict['active'] += 1
        sequences.extend(new_sequences)
        new_total = sum(seq['N0'] for seq in sequences)
        delta_n = new_total - current_n
        remaining_substrate = max(0, remaining_substrate - delta_n)
        total_sequences_history.append(new_total)
        logging.info(
            f"Cycle {cycle} complete. Total unique sequences: {len(sequences)}; Remaining substrate: {remaining_substrate}"
        )
    if sequences:
        # Adjust each sequence's length to match the length of the first sequence.
        desired_length = len(sequences[0]['sequence'])
        for seq in sequences:
            if len(seq["sequence"]) > desired_length:
                seq["sequence"] = seq["sequence"][ :desired_length]
            elif len(seq["sequence"]) < desired_length:
                seq["sequence"] = seq["sequence"] + ''.join(random.choices(['A', 'C', 'G', 'T'], k=desired_length - len(seq["sequence"])))

    # Collapse sequences with identical sequence string
    collapsed = {}
    for seq in sequences:
        key = seq['sequence']
        if key not in collapsed:
            collapsed[key] = {
                'sequence': key,
                'N0': seq['N0'],
                'id': [seq['id']],
                'parent_id': [seq['parent_id']],
                'born': [seq['born']],
                'active': [seq['active']]
            }
        else:
            collapsed[key]['N0'] += seq['N0']
            collapsed[key]['id'].append(seq['id'])
            collapsed[key]['parent_id'].append(seq['parent_id'])
            collapsed[key]['born'].append(seq['born'])
            collapsed[key]['active'].append(seq['active'])
    sequences = list(collapsed.values())

    base, ext = os.path.splitext(output)
    pcr_output = f"{base_folder}results_amplified/pcr_{base}{ext}"
    return sequences, total_sequences_history, pcr_output