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
from typing import List, Dict, Tuple, Any
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
        remaining_substrate = max(0, remaining_substrate - delta_N)
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
                         mutation_rate: float,
                         mutation_probabilities: Dict[str, float],
                         substrate_capacity_initial: float,
                         S_radius: float,
                         AOE_radius: float,
                         density: float,
                         success_prob: float,
                         deviation: float,) -> list[dict[str, Any] | dict[str, str | int]]:
    """
    Perform Bridge amplification simulation.
    Each cycle applies a random deviation (±10% by default) to parameters S, density, and success probability.
    The effective success probability (after deviation) is used as the chance for amplification.
    Mutation processing and substrate capacity updating are handled similarly to PCR.
    Returns the final sequence list and a history of total unique sequences per cycle.
    """
    total_sequences_history = []
    remaining_substrate = substrate_capacity_initial

    simulation_index = 0  # To track which simulation we're in
    all_cycle_counts = []  # To store cycle counts list from each simulation
    merged_sequences = []  # To accumulate A_points_dict values from all simulations
    check = 0

    for seq_dict in sequences:
        # Calculating parameters for every sequence
        effective_S_radius = S_radius * (1 + random.uniform(-deviation, deviation))
        effective_density = density * (1 + random.uniform(-deviation, deviation))
        effective_S = math.pi * effective_S_radius ** 2
        num_P = int(effective_density * effective_S)
        effective_success_prob = success_prob * (1 + random.uniform(-deviation, deviation))
        effective_success_prob = min(effective_success_prob, 1.0)
        effective_AOE_radius = AOE_radius * (1 + random.uniform(-deviation, deviation))

        P_points = []
        for i in range(num_P):
            # Use polar coordinates to ensure a uniform distribution.
            r = effective_S_radius * math.sqrt(random.random())
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            P_points.append({'id': i, 'x': x, 'y': y})
        global_P = P_points  # available P points

        # --- Define the A point class ---
        class APoint:
            def __init__(self, sequence, x, y):
                self.sequence = sequence
                self.x = x
                self.y = y
                self.active = True  # True while it finds available P's within its AOE

            def distance_to(self, p):
                return math.hypot(self.x - p['x'], self.y - p['y'])

        # --- Initialize with one A point at the center ---
        A_points = []
        # Here we initialize the simulation's local list with one dictionary taken from seq_dict.
        local_seq_list = [{"sequence": seq_dict["sequence"], "N0": seq_dict["N0"]}]
        initial_A = APoint(seq_dict["sequence"], 0, 0)
        A_points.append(initial_A)
        active_A = [initial_A]

        # Initialize a list to track the total number of A_points at each cycle.
        cycle_counts = []
        cycle = 0

        # --- For the first simulation only, set up the animation figure ---
        if simulation_index == 0:
            fig, ax = plt.subplots()

        # --- Main simulation loop ---
        while global_P and active_A:
            # At the start of each cycle, record the current total number of A_points.
            cycle_counts.append(len(A_points))

            # For the first simulation, update the animation at the start of the cycle.
            if simulation_index == 0:
                ax.cla()
                # Plot available P points as blue small dots.
                p_x = [p['x'] for p in global_P]
                p_y = [p['y'] for p in global_P]
                ax.scatter(p_x, p_y, color='blue', s=5, label='P points')

                # Plot all A points as red larger dots.
                a_x = [a.x for a in A_points]
                a_y = [a.y for a in A_points]
                ax.scatter(a_x, a_y, color='red', s=10, label='A points')

                # Draw the AOE for each active A point as a green transparent circle.
                for a in active_A:
                    circle = plt.Circle((a.x, a.y), effective_AOE_radius, color='green', alpha=0.3)
                    ax.add_patch(circle)

                ax.set_xlim(-effective_S_radius, effective_S_radius)
                ax.set_ylim(-effective_S_radius, effective_S_radius)
                ax.set_aspect('equal')
                ax.legend(loc='upper right')
                ax.set_title(f"Cycle {cycle}")
                plt.pause(0.5)

            # Each active A checks for at least one available P point within its AOE.
            pending_A = {}
            for a in active_A:
                candidates = [p for p in global_P if a.distance_to(p) <= AOE_radius]
                if not candidates:
                    a.active = False  # Mark A as inactive if no candidate P is found.
                else:
                    pending_A[a.x] = a.x
                    pending_A[a.y] = a.y

            # Update active_A list.
            active_A = [a for a in active_A if a.active]
            if not active_A:
                break

            # Each active A (with candidates) tries to form a connection.
            pending = set(active_A)
            while pending:
                # proposals: maps candidate P's id to a list of APoint instances that propose it.
                proposals = {}
                remove_from_pending = set()

                # Each active A (in pending) looks for candidate P points within its AOE.
                for a in list(pending):
                    candidates = [p for p in global_P if a.distance_to(p) <= effective_AOE_radius]
                    if not candidates:
                        # No candidate found: remove this APoint from pending.
                        remove_from_pending.add(a)
                    else:
                        # Randomly choose one candidate for a proposal.
                        chosen = random.choice(candidates)
                        proposals.setdefault(chosen['id'], []).append(a)

                # Remove those APoints that found no candidates.
                pending -= remove_from_pending
                if not proposals:
                    break

                # Process each candidate P’s proposals.
                for p_id, a_list in proposals.items():
                    # Find the candidate P point (if it’s still available).
                    p_obj = next((p for p in global_P if p['id'] == p_id), None)
                    if p_obj is None:
                        continue  # Already taken.

                    success_list = []
                    # For each APoint proposing this candidate, check the connection probability.
                    for a in a_list:
                        if a in pending and random.random() < effective_success_prob:
                            success_list.append(a)

                    if not success_list:
                        # If none succeed, remove all these APoints from pending.
                        for a in a_list:
                            pending.discard(a)
                    else:
                        # If one or more succeed, choose a winner at random.
                        winner = random.choice(success_list)
                        mutated_seq, mutation_occurred = process_mutation(winner.sequence,
                                                                          mutation_rate,
                                                                          mutation_probabilities)
                        found = False
                        for local_dict in local_seq_list:
                            if local_dict["sequence"] == mutated_seq:
                                local_dict["N0"] += 1
                                found = True
                                break

                        if not found:
                            local_seq_list.append({"sequence": mutated_seq, "N0": 1})
                        # Create a new APoint using the winner's (possibly mutated) sequence and the candidate's location.
                        new_A = APoint(mutated_seq, p_obj['x'], p_obj['y'])
                        A_points.append(new_A)
                        active_A.append(new_A)
                        # Remove the candidate P from the available points.
                        global_P = [p for p in global_P if p['id'] != p_id]
                        # Remove all APoints that proposed this candidate from pending.
                        for a in a_list:
                            pending.discard(a)
            # End proposals.
            active_A = [a for a in A_points if a.active]
            cycle += 1

        # End of simulation for this seq_dict.
        all_cycle_counts.append(cycle_counts)
        print(f"Length of local_seq_list: {len(local_seq_list)}")
        # --- Merge this simulation's local_seq_list into the global merged_sequences ---
        for d in local_seq_list:
            found = False
            for md in merged_sequences:
                if check == 0:      # It is here to ensure that merged_sequences is empty at the start
                    merged_sequences = []
                    check = 1
                if md["sequence"] == d["sequence"]:
                    md["N0"] += d["N0"]
                    found = True
                    break
            if not found:
                merged_sequences.append(d)
        print(f"Length of merged_sequences: {len(merged_sequences)}")
        # For the first simulation, leave the final frame displayed.
        if simulation_index == 0:
            plt.show(block=False)

        simulation_index += 1


    # After processing all seq_dict in sequences, summarize the cycle counts.
    # This performs an element-wise sum over all cycle_counts lists.
    history_bridge = [sum(x) for x in zip(*all_cycle_counts)]
    print(type(history_bridge))
    print(len(history_bridge))
    print(f"Length of merged_sequences: {len(merged_sequences)}")
    return merged_sequences, history_bridge


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
    amplify_parser.add_argument('--mutation_rate', type=float, default=0.01, help="Mutation rate per nucleotide per replication event")
    amplify_parser.add_argument('--substitution_prob', type=float, default=0.4, help="Probability of substitution mutation")
    amplify_parser.add_argument('--deletion_prob', type=float, default=0.3, help="Probability of deletion mutation")
    amplify_parser.add_argument('--insertion_prob', type=float, default=0.3, help="Probability of insertion mutation")
    amplify_parser.add_argument('--substrate_capacity', type=float, default=(2**18), help="Initial substrate capacity")
    amplify_parser.add_argument('--S', type=float, default=700_000_000, help="Threshold S parameter")
    amplify_parser.add_argument('--S_radius', type=float, default=10, help="Threshold S parameter")
    amplify_parser.add_argument('--AOE_radius', type=float, default=1, help="Threshold S parameter")
    amplify_parser.add_argument('--C', type=float, default=1e-9, help="Sharpness parameter C")
    amplify_parser.add_argument('--density', type=float, default=10, help="Density parameter for Bridge amplification")
    amplify_parser.add_argument('--success_prob', type=float, default=0.85, help="Success probability for Bridge amplification")
    amplify_parser.add_argument('--deviation', type=float, default=0.1, help="Deviation for Bridge amplification parameters (e.g., 0.1 for 10%)")
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
            sequences_bridge_amp, history_bridge = bridge_amplification(
                sequences_bridge,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                substrate_capacity_initial=args.substrate_capacity,
                S_radius=args.S_radius,
                AOE_radius=args.AOE_radius,
                density=args.density,
                success_prob=args.success_prob,
                deviation=args.deviation,
            )
            bridge_output = 'bridge_amplified.csv'
            with open(bridge_output, 'w', newline='') as csvfile:
                fieldnames = ['sequence', 'N0']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for seq_dict in sequences_bridge_amp:
                    writer.writerow(seq_dict)
            logging.info(f"Generated {len(sequences_bridge_amp)} sequences and saved to {bridge_output}")

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
