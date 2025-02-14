#!/usr/bin/env python3
"""
DNA Amplification Simulation Tool

Subcommands:
  generate   Generate true barcode sequences.
  amplify    Amplify sequences using PCR and/or Bridge amplification.
  denoise    Denoise amplified sequences.
"""

import argparse
import csv
import random
import math
import logging
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

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


def compute_global_p(current_N: float, remaining_substrate: float, substrate_capacity_initial: float,
                     S: float, K: float, C: float) -> float:
    """
    Compute the global amplification probability using the provided PCR formula.
    current_N: Total copies in the system.
    remaining_substrate: Remaining substrate capacity.
    substrate_capacity_initial: The initial substrate capacity.
    S: Threshold parameter.
    K: Half-saturation constant (typically S * 10).
    C: Sharpness of phase transition.
    """
    sub_depletion = remaining_substrate / substrate_capacity_initial
    if current_N < S:
        p_equation = K / (K + S)
    else:
        p_equation = (K / (K + current_N)) * ((1 + math.exp(-C * (current_N / (S - 1)))) / 2)
    p = p_equation * sub_depletion
    return p


def process_mutation(sequence: str, mutation_rate: float,
                     mutation_probabilities: Dict[str, float]) -> Tuple[str, bool]:
    """
    Process mutation over a sequence replication event.
    Each nucleotide is checked with probability `mutation_rate` for mutation.
    If a mutation occurs, one of three types is chosen:
      - substitution: replace nucleotide with a different one.
      - deletion: remove the nucleotide.
      - insertion: insert a new nucleotide before the current one.
    Returns a tuple of (possibly mutated sequence, mutation_occurred flag).
    """
    seq_list = list(sequence)
    mutated = False
    i = 0
    while i < len(seq_list):
        if random.random() < mutation_rate:
            mutated = True
            mutation_type = random.choices(
                population=['substitution', 'deletion', 'insertion'],
                weights=[mutation_probabilities['substitution'],
                         mutation_probabilities['deletion'],
                         mutation_probabilities['insertion']],
                k=1
            )[0]
            if mutation_type == 'substitution':
                current_nuc = seq_list[i]
                options = [n for n in NUCLEOTIDES if n != current_nuc]
                seq_list[i] = random.choice(options)
                i += 1
            elif mutation_type == 'deletion':
                del seq_list[i]
                # Do not increment i; next nucleotide shifts into this position.
            elif mutation_type == 'insertion':
                inserted = random.choice(NUCLEOTIDES)
                seq_list.insert(i, inserted)
                i += 2  # Skip the inserted nucleotide.
        else:
            i += 1
    return ''.join(seq_list), mutated


def pcr_amplification(sequences: List[Dict[str, any]],
                      cycles: int,
                      mutation_rate: float,
                      mutation_probabilities: Dict[str, float],
                      substrate_capacity_initial: float,
                      S: float,
                      C: float,
                      plot: bool) -> Tuple[List[Dict[str, any]], List[int]]:
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
    K = S * 10
    for cycle in range(1, cycles + 1):
        logging.info(f"PCR Amplification: Cycle {cycle} starting with {len(sequences)} sequences.")
        current_N = sum(seq['N0'] for seq in sequences)
        p = compute_global_p(current_N, remaining_substrate, substrate_capacity_initial, S, K, C)
        logging.info(f"Cycle {cycle}: Global amplification probability = {p:.4f}")
        new_sequences = []
        for seq_dict in sequences:
            if random.random() < p:
                replication_count = seq_dict['N0']
                no_mutation_count = 0
                for _ in range(replication_count):
                    mutated_seq, mutation_occurred = process_mutation(seq_dict['sequence'],
                                                                      mutation_rate,
                                                                      mutation_probabilities)
                    if mutation_occurred:
                        new_sequences.append({'sequence': mutated_seq, 'N0': 1})
                    else:
                        no_mutation_count += 1
                seq_dict['N0'] += no_mutation_count
        sequences.extend(new_sequences)
        new_total = sum(seq['N0'] for seq in sequences)
        delta_N = new_total - current_N
        remaining_substrate = max(0, remaining_substrate - delta_N)     # if delta_N > remaining_substrate then remaining substrate = 0
        total_sequences_history.append(len(sequences))
        logging.info(f"Cycle {cycle} complete. Total unique sequences: {len(sequences)}; Remaining substrate: {remaining_substrate}")
    if plot:
        plt.figure()
        plt.plot(range(1, cycles + 1), total_sequences_history, marker='o')
        plt.xlabel("Cycle Number")
        plt.ylabel("Total Number of Unique Sequences")
        plt.title("PCR Amplification: Total Sequences per Cycle")
        plt.grid(True)
        plt.show()
    return sequences, total_sequences_history


def bridge_amplification(sequences: List[Dict[str, any]],
                         cycles: int,
                         mutation_rate: float,
                         mutation_probabilities: Dict[str, float],
                         S_area: float,
                         density: float,
                         success_prob: float,
                         deviation: float,
                         plot: bool) -> Tuple[List[Dict[str, any]], List[int], List[int]]:
    """
    Perform Bridge amplification using a global pool of P points.

    Parameters:
      - sequences: List of input sequences (each with keys 'sequence' and 'N0').
      - cycles: Maximum number of cycles to simulate.
      - mutation_rate: Mutation rate per replication event.
      - mutation_probabilities: Dictionary with keys 'substitution', 'deletion', 'insertion'.
      - S_area: Total area of the flow cell (used to compute the global pool of P points).
      - density: Density of P points per unit area.
      - success_prob: Base replication success probability.
      - deviation: Fractional deviation to apply to success_prob per sequence.
      - plot: Whether to show a plot of total unique sequences per cycle and remaining P points.

    Process:
      1. Compute the global pool: global_P = int(density * S_area)
      2. For each cycle, for each sequence and for each of its copies (N0), if global_P > 0:
           - Attempt replication using the sequence’s effective success probability.
           - Consume one P point per replication event.
           - If replication occurs:
               - Process mutation: if a mutation occurs, add a new sequence (with N0=1) to the dataset.
                 Otherwise, increment the parent's N0.
      3. Continue cycles until global_P is exhausted or no replication occurs in a cycle.
      4. Record and plot the remaining global P points in each cycle.

    Returns:
      - Updated list of sequences.
      - A history list recording the total number of unique sequences per cycle.
      - A history list recording the remaining global P points per cycle.
    """
    # Compute global pool of P points (fixed for the simulation)
    global_P = int(density * S_area)
    logging.info(f"Global pool of P points: {global_P}")

    # Initialize effective success probability for each input sequence if not already assigned.
    for seq in sequences:
        if 'effective_success_prob' not in seq:
            # Assign effective success probability with deviation.
            seq['effective_success_prob'] = min(success_prob * (1 + random.uniform(-deviation, deviation)), 1.0)

    seq_history = []  # Total unique sequences per cycle.
    P_history = []  # Global P points remaining per cycle.
    cycle = 0

    # Loop over cycles until global P points are exhausted or maximum cycles reached.
    while global_P > 0 and cycle < cycles:
        cycle += 1
        replication_occurred = False
        new_sequences = []  # To collect any mutated new sequences in this cycle

        logging.info(f"Cycle {cycle} starting with {len(sequences)} sequences and global_P = {global_P}.")

        # For each sequence in the dataset:
        for seq in sequences:
            # We'll attempt replication for each copy of this sequence.
            copies = seq['N0']
            for i in range(copies):
                if global_P <= 0:
                    break  # No more available P points.
                # Attempt replication event based on effective success probability.
                if random.random() < seq['effective_success_prob']:
                    global_P -= 1  # Consume one global P point.
                    replication_occurred = True
                    # Process mutation on the replication event.
                    mutated_seq, mutation_occurred = process_mutation(seq['sequence'],
                                                                      mutation_rate,
                                                                      mutation_probabilities)
                    if mutation_occurred:
                        # If mutation occurs, add a new sequence with N0 = 1.
                        new_seq = {
                            'sequence': mutated_seq,
                            'N0': 1,
                            'effective_success_prob': min(success_prob * (1 + random.uniform(-deviation, deviation)),
                                                          1.0)
                        }
                        new_sequences.append(new_seq)
                    else:
                        # If no mutation, increment the parent's copy count.
                        seq['N0'] += 1

        # Add any new mutated sequences to the main dataset.
        if new_sequences:
            sequences.extend(new_sequences)

        seq_history.append(len(sequences))
        P_history.append(global_P)
        logging.info(
            f"Cycle {cycle} complete. Total unique sequences: {len(sequences)}; Remaining global_P: {global_P}")

        # If no replication occurred in this cycle, break the loop.
        if not replication_occurred:
            logging.info("No replication events occurred in this cycle. Ending simulation.")
            break

    if plot:
        plt.figure(figsize=(10, 5))
        cycles_range = range(1, len(seq_history) + 1)
        plt.plot(cycles_range, seq_history, marker='o', label="Unique Sequences")
        plt.plot(cycles_range, P_history, marker='s', label="Remaining Global P Points")
        plt.xlabel("Cycle Number")
        plt.ylabel("Count")
        plt.title("Bridge Amplification: Unique Sequences and Remaining Global P Points per Cycle")
        plt.legend()
        plt.grid(True)
        plt.show()

    return sequences, seq_history, P_history


# Simplified denoiser class using parts of your provided code.
class Denoiser:
    def __init__(self, input_csv: str):
        self.input_csv = input_csv
        self.data = []
        self.load_data()

    def load_data(self):
        with open(self.input_csv, 'r') as f:
            reader = csv.DictReader(f)
            self.data = [row for row in reader]
        logging.info(f"Denoiser loaded {len(self.data)} sequences from {self.input_csv}")

    def simple(self, threshold: int, output_csv: str, plot: bool = True):
        valid_sequences = [row for row in self.data if int(row.get('N0', row.get('amount', 0))) >= threshold]
        with open(output_csv, 'w', newline='') as f:
            if valid_sequences:
                writer = csv.DictWriter(f, fieldnames=valid_sequences[0].keys())
                writer.writeheader()
                writer.writerows(valid_sequences)
        logging.info(f"Simple denoising complete. {len(valid_sequences)} sequences saved to {output_csv}")
        if plot:
            amounts = [int(row.get('N0', row.get('amount', 0))) for row in valid_sequences]
            plt.figure()
            plt.hist(amounts, bins=20, color='skyblue', edgecolor='black')
            plt.xlabel("N0 / Amount")
            plt.ylabel("Frequency")
            plt.title("Distribution of Sequence Counts after Simple Denoising")
            plt.grid(True)
            plt.show()
        return valid_sequences

    def directional(self, output_csv: str, plot: bool = True):
        # Group sequences by their string and sum counts.
        sequence_map = {}
        for row in self.data:
            seq = row['sequence'] if 'sequence' in row else row.get('Sequence', '')
            count = int(row.get('N0', row.get('amount', 1)))
            sequence_map[seq] = sequence_map.get(seq, 0) + count
        result = [{'sequence': seq, 'count': count} for seq, count in sequence_map.items() if count >= 2]
        with open(output_csv, 'w', newline='') as f:
            if result:
                writer = csv.DictWriter(f, fieldnames=['sequence', 'count'])
                writer.writeheader()
                writer.writerows(result)
        logging.info(f"Directional denoising complete. {len(result)} sequences saved to {output_csv}")
        if plot:
            counts = [item['count'] for item in result]
            plt.figure()
            plt.hist(counts, bins=20, color='lightgreen', edgecolor='black')
            plt.xlabel("Sequence Count")
            plt.ylabel("Frequency")
            plt.title("Directional Denoising: Sequence Count Distribution")
            plt.grid(True)
            plt.show()
        return result


def main():
    parser = argparse.ArgumentParser(description="DNA Amplification Simulation Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: generate
    generate_parser = subparsers.add_parser('generate', help="Generate true barcode sequences")
    generate_parser.add_argument('--num', type=int, required=True, help="Number of sequences to generate")
    generate_parser.add_argument('--length', type=int, required=True, help="Length of each sequence")
    generate_parser.add_argument('--unique', action='store_true', help="Ensure sequences are unique")
    generate_parser.add_argument('--output', type=str, default='true_barcodes.csv', help="Output CSV filename")

    # Subcommand: amplify
    amplify_parser = subparsers.add_parser('amplify', help="Amplify sequences using PCR and/or Bridge amplification")
    amplify_parser.add_argument('--method', type=str, choices=['pcr', 'bridge', 'both'], required=True,
                                help="Amplification method to use")
    amplify_parser.add_argument('--cycles', type=int, default=30, help="Number of amplification cycles")
    amplify_parser.add_argument('--mutation_rate', type=float, default=0.001, help="Mutation rate per nucleotide per replication event")
    amplify_parser.add_argument('--substitution_prob', type=float, default=0.4, help="Probability of substitution mutation")
    amplify_parser.add_argument('--deletion_prob', type=float, default=0.3, help="Probability of deletion mutation")
    amplify_parser.add_argument('--insertion_prob', type=float, default=0.3, help="Probability of insertion mutation")
    amplify_parser.add_argument('--substrate_capacity', type=float, default=(2**18), help="Initial substrate capacity")
    amplify_parser.add_argument('--S', type=float, default=700_000_000, help="Threshold S parameter")
    amplify_parser.add_argument('--S_area', type=float, default=5.0, help="S_area parameter for Bridge Amplification (used to compute available area for P points)")
    amplify_parser.add_argument('--C', type=float, default=1e-9, help="Sharpness parameter C")
    amplify_parser.add_argument('--density', type=float, default=5, help="Density parameter for Bridge amplification")
    amplify_parser.add_argument('--success_prob', type=float, default=0.85, help="Success probability for Bridge amplification")
    amplify_parser.add_argument('--deviation', type=float, default=0.01, help="Deviation for Bridge amplification parameters (e.g., 0.1 for 10%)")
    amplify_parser.add_argument('--input', type=str, default='true_barcodes.csv', help="Input CSV filename with true barcodes")
    amplify_parser.add_argument('--plot', dest='plot', action='store_true', help="Enable plotting (default)", default=True)
    amplify_parser.add_argument('--no_plot', dest='plot', action='store_false', help="Disable plotting")

    # Subcommand: denoise
    denoise_parser = subparsers.add_parser('denoise', help="Denoise amplified sequences")
    denoise_parser.add_argument('--input', type=str, required=True, help="Input CSV filename for denoising")
    denoise_parser.add_argument('--method', type=str, choices=['simple', 'directional'], required=True,
                                help="Denoising method to use")
    denoise_parser.add_argument('--threshold', type=int, default=300, help="Threshold for simple denoising")
    denoise_parser.add_argument('--output', type=str, default='denoised.csv', help="Output CSV filename for denoised sequences")
    denoise_parser.add_argument('--plot', dest='plot', action='store_true', help="Enable plotting (default)", default=True)
    denoise_parser.add_argument('--no_plot', dest='plot', action='store_false', help="Disable plotting")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if args.command == 'generate':
        generate_sequences(args.num, args.length, args.unique, args.output)

    elif args.command == 'amplify':
        # Load true barcodes from CSV.
        sequences = []
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sequences.append({'sequence': row['sequence'], 'N0': int(row['N0'])})
        mutation_probabilities = {
            'substitution': args.substitution_prob,
            'deletion': args.deletion_prob,
            'insertion': args.insertion_prob
        }
        total_prob = sum(mutation_probabilities.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-2):
            logging.error("Cumulative mutation probabilities must equal 1.0")
            return

        if args.method in ['pcr', 'both']:
            sequences_pcr = [dict(seq) for seq in sequences]
            logging.info("Starting PCR amplification...")
            sequences_pcr, history_pcr = pcr_amplification(
                sequences_pcr,
                cycles=args.cycles,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                substrate_capacity_initial=args.substrate_capacity,
                S=args.S,
                C=args.C,
                plot=args.plot
            )
            pcr_output = 'pcr_amplified.csv'
            with open(pcr_output, 'w', newline='') as f:
                fieldnames = ['sequence', 'N0']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for seq in sequences_pcr:
                    writer.writerow(seq)
            logging.info(f"PCR amplification complete. Results saved to {pcr_output}.")

        if args.method in ['bridge', 'both']:
            sequences_bridge = [dict(seq) for seq in sequences]
            logging.info("Starting Bridge amplification...")
            sequences_bridge, history_bridge = bridge_amplification(
                sequences_bridge,
                cycles=args.cycles,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                S=args.S,
                density=args.density,
                success_prob=args.success_prob,
                deviation=args.deviation,
                plot=args.plot
            )
            bridge_output = 'bridge_amplified.csv'
            with open(bridge_output, 'w', newline='') as f:
                fieldnames = ['sequence', 'N0']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for seq in sequences_bridge:
                    writer.writerow(seq)
            logging.info(f"Bridge amplification complete. Results saved to {bridge_output}.")

        if args.method == 'both' and args.plot:
            plt.figure()
            plt.plot(range(1, args.cycles + 1), history_pcr, marker='o', label='PCR Amplification')
            plt.plot(range(1, args.cycles + 1), history_bridge, marker='s', label='Bridge Amplification')
            plt.xlabel("Cycle Number")
            plt.ylabel("Total Unique Sequences")
            plt.title("Total Sequences per Cycle Comparison")
            plt.legend()
            plt.grid(True)
            plt.show()

    elif args.command == 'denoise':
        denoiser = Denoiser(args.input)
        if args.method == 'simple':
            denoiser.simple(args.threshold, args.output, plot=args.plot)
        elif args.method == 'directional':
            denoiser.directional(args.output, plot=args.plot)


if __name__ == '__main__':
    main()