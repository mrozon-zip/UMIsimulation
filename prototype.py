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
import logging
import matplotlib.pyplot as plt
import random
import math
from typing import List, Dict, Tuple

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


def compute_global_p(current_n: float, remaining_substrate: float, substrate_capacity_initial: float,
                     s: float, k: float, c: float) -> float:
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
    if current_n < s:
        p_equation = k / (k + s)
    else:
        p_equation = (k / (k + current_n)) * ((1 + math.exp(-c * (current_n / (s - 1)))) / 2)
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
                      s: float,
                      c: float,
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
        delta_n = new_total - current_n
        remaining_substrate = max(0, remaining_substrate - delta_n)
        total_sequences_history.append(len(sequences))
        logging.info(
            f"Cycle {cycle} complete. Total unique sequences: {len(sequences)}; Remaining substrate: {remaining_substrate}")
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
                         simulate: bool,
                         mutation_rate: float,
                         mutation_probabilities: Dict[str, float],
                         s_radius: float,
                         aoe_radius: float,
                         density: float,
                         success_prob: float,
                         deviation: float, ) -> Tuple[List[Dict[str, any]], List[int]]:
    """
    Perform Bridge amplification simulation.
    Each cycle applies a random deviation (±10% by default) to parameters S, density, and success probability.
    The effective success probability (after deviation) is used as the chance for amplification.
    Mutation processing and substrate capacity updating are handled similarly to PCR.
    Returns the final sequence list and a history of total unique sequences per cycle.
    """

    simulation_index = simulate  # To track which simulation we're in
    all_cycle_counts = []  # To store cycle counts list from each simulation
    merged_sequences = []  # To accumulate A_points_dict values from all simulations
    check = 0

    for seq_dict in sequences:
        # Calculating parameters for every sequence
        effective_s_radius = s_radius * (1 + random.uniform(-deviation, deviation))
        effective_density = density * (1 + random.uniform(-deviation, deviation))
        effective_s = math.pi * effective_s_radius ** 2
        num_p = int(effective_density * effective_s)
        effective_success_prob = success_prob * (1 + random.uniform(-deviation, deviation))
        effective_success_prob = min(effective_success_prob, 1.0)
        effective_aoe_radius = aoe_radius * (1 + random.uniform(-deviation, deviation))

        p_points = []
        for i in range(num_p):
            # Use polar coordinates to ensure a uniform distribution.
            r = effective_s_radius * math.sqrt(random.random())
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p_points.append({'id': i, 'x': x, 'y': y})
        global_p = p_points  # available P points

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
        a_points = []
        # Here we initialize the simulation's local list with one dictionary taken from seq_dict.
        local_seq_list = [{"sequence": seq_dict["sequence"], "N0": seq_dict["N0"]}]
        initial_a = APoint(seq_dict["sequence"], 0, 0)
        a_points.append(initial_a)
        active_a = [initial_a]

        # Initialize a list to track the total number of a_points at each cycle.
        cycle_counts = []
        cycle = 0

        # --- For the first simulation only, set up the animation figure ---
        if simulation_index:
            fig, ax = plt.subplots()

        # --- Main simulation loop ---
        while global_p and active_a:
            # At the start of each cycle, record the current total number of a_points.
            cycle_counts.append(len(a_points))

            # For the first simulation, update the animation at the start of the cycle.
            if simulation_index:
                ax.cla()
                # Plot available P points as blue small dots.
                p_x = [p['x'] for p in global_p]
                p_y = [p['y'] for p in global_p]
                ax.scatter(p_x, p_y, color='blue', s=5, label='P points')

                # Plot all A points as red larger dots.
                a_x = [a.x for a in a_points]
                a_y = [a.y for a in a_points]
                ax.scatter(a_x, a_y, color='red', s=10, label='A points')

                # Draw the AOE for each active A point as a green transparent circle.
                for a in active_a:
                    circle = plt.Circle((a.x, a.y), effective_aoe_radius, color='green', alpha=0.3)
                    ax.add_patch(circle)

                ax.set_xlim(-effective_s_radius, effective_s_radius)
                ax.set_ylim(-effective_s_radius, effective_s_radius)
                ax.set_aspect('equal')
                ax.legend(loc='upper right')
                ax.set_title(f"Cycle {cycle}")
                plt.pause(0.5)

            # Each active A checks for at least one available P point within its AOE.
            pending_a = {}
            for a in active_a:
                candidates = [p for p in global_p if a.distance_to(p) <= aoe_radius]
                if not candidates:
                    a.active = False  # Mark A as inactive if no candidate P is found.
                else:
                    pending_a[a.x] = a.x
                    pending_a[a.y] = a.y

            # Update active_a list.
            active_a = [a for a in active_a if a.active]
            if not active_a:
                break

            # Each active A (with candidates) tries to form a connection.
            pending = set(active_a)
            while pending:
                # proposals: maps candidate P's id to a list of APoint instances that propose it.
                proposals = {}
                remove_from_pending = set()

                # Each active A (in pending) looks for candidate P points within its AOE.
                for a in list(pending):
                    candidates = [p for p in global_p if a.distance_to(p) <= effective_aoe_radius]
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
                    p_obj = next((p for p in global_p if p['id'] == p_id), None)
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
                        new_a = APoint(mutated_seq, p_obj['x'], p_obj['y'])
                        a_points.append(new_a)
                        active_a.append(new_a)
                        # Remove the candidate P from the available points.
                        global_p = [p for p in global_p if p['id'] != p_id]
                        # Remove all APoints that proposed this candidate from pending.
                        for a in a_list:
                            pending.discard(a)
            # End proposals.
            active_a = [a for a in a_points if a.active]
            cycle += 1

        # End of simulation for this seq_dict.
        all_cycle_counts.append(cycle_counts)
        print(all_cycle_counts[0:2])
        print(f"Length of local_seq_list: {len(local_seq_list)}")
        # --- Merge this simulation's local_seq_list into the global merged_sequences ---
        for d in local_seq_list:
            found = False
            for md in merged_sequences:
                if check == 0:  # It is here to ensure that merged_sequences is empty at the start
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
        if simulation_index:
            plt.show(block=False)

        simulation_index = False

    # After processing all seq_dict in sequences, summarize the cycle counts.

    print(max(len(x) for x in all_cycle_counts))

    # Step 1: Find the length of the longest list
    max_length = max(len(x) for x in all_cycle_counts)

    # Step 2: Pad each list to match the length of the longest list
    padded_all_cycle_counts = []
    for lst in all_cycle_counts:
        if len(lst) < max_length:
            # Append the last element until the list reaches the max length
            lst.extend([lst[-1]] * (max_length - len(lst)))
        padded_all_cycle_counts.append(lst)

    # This performs an element-wise sum over all cycle_counts lists.
    history_bridge = [sum(x) for x in zip(*all_cycle_counts)]
    print(type(history_bridge))
    print(len(history_bridge))
    print(f"Length of merged_sequences: {len(merged_sequences)}")
    return merged_sequences, history_bridge


def polonies_amplification(sequences: List[Dict[str, any]],
                           simulate: bool,
                           mutation_rate: float,
                           mutation_probabilities: Dict[str, float],
                           s_radius: float,
                           aoe_radius: float,
                           density: float,
                           success_prob: float,
                           deviation: float) -> Tuple[List[Dict[str, any]], List[int]]:
    """
    Perform modified Bridge amplification simulation with visual representation of points and AOEs at each cycle.
    Now P points are divided into B and D points.
    A_point (reacting with B points) and C_point (reacting with D points)
    are created from the same starting sequence.
    A_point acts first in each cycle and C_point acts second.
    Points are displayed with different colors.
    """

    simulation_index = simulate  # To track whether to animate.
    all_cycle_counts = []  # To store cycle counts list from each simulation
    merged_sequences = []  # To accumulate sequence dictionaries from all simulations
    check = 0

    # Calculate parameters for every sequence.
    effective_s_radius = s_radius * (1 + random.uniform(-deviation, deviation))
    effective_density = density * (1 + random.uniform(-deviation, deviation))
    effective_s = math.pi * effective_s_radius ** 2
    num_p = int(effective_density * effective_s)
    effective_success_prob = success_prob * (1 + random.uniform(-deviation, deviation))
    effective_success_prob = min(effective_success_prob, 1.0)
    effective_aoe_radius = aoe_radius * (1 + random.uniform(-deviation, deviation))

    # --- Divide generated P points into b_points and d_points ---
    b_points = []
    d_points = []
    for i in range(num_p):
        # Uniform distribution in a circle (using polar coordinates)
        r = effective_s_radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        point = {'id': i, 'x': x, 'y': y}
        # Randomly assign to B or D (here 50/50 chance)
        if random.random() < 0.5:
            b_points.append(point)
        else:
            d_points.append(point)

    # --- Define the A point class (reacts with B points) ---
    class APoint:
        def __init__(self, sequence, x, y):
            self.sequence = sequence
            self.x = x
            self.y = y
            self.active = True  # Active while it can find available B's within its AOE

        def distance_to(self, p):
            return math.hypot(self.x - p['x'], self.y - p['y'])

    # --- Define the C point class (reacts with D points) ---
    class CPoint:
        def __init__(self, sequence, x, y):
            self.sequence = sequence
            self.x = x
            self.y = y
            self.active = True  # Active while it can find available D's within its AOE

        def distance_to(self, p):
            return math.hypot(self.x - p['x'], self.y - p['y'])

    # --- Initialize with one A_point and one C_point at the centre ---
    local_seq_list = sequences
    initial_a = []
    initial_c = []

    for i in range(0, len(local_seq_list) - 1, 2):
        dict_n = local_seq_list[i]
        dict_n_plus_1 = local_seq_list[i + 1]

        # Generate random coordinates for A_point within the S_area
        r_a = effective_s_radius * math.sqrt(random.random())
        theta_a = random.uniform(0, 2 * math.pi)
        x_a = r_a * math.cos(theta_a)
        y_a = r_a * math.sin(theta_a)

        # Generate random coordinates for C_point within the S_area
        r_c = effective_s_radius * math.sqrt(random.random())
        theta_c = random.uniform(0, 2 * math.pi)
        x_c = r_c * math.cos(theta_c)
        y_c = r_c * math.sin(theta_c)

        # Create new points with these coordinates
        new_a = APoint(dict_n["sequence"], x_a, y_a)
        new_c = CPoint(dict_n_plus_1["sequence"], x_c, y_c)

        # Append them to your initial lists
        initial_a.append(new_a)
        initial_c.append(new_c)

    a_points = initial_a
    c_points = initial_c
    active_a = initial_a
    active_c = initial_c

    # To track total number of points (both A and C) per cycle.
    cycle_counts = []
    cycle = 0

    # --- For the first simulation only, set up the animation figure ---
    if simulation_index:
        fig, ax = plt.subplots()

    # --- Main simulation loop ---
    # Continue while at least one group has both active points and available candidate points.
    while (b_points and active_a) or (d_points and active_c):
        # Record total number of A and C points.
        cycle_counts.append(len(a_points) + len(c_points))

        # --- Animation update ---
        if simulation_index:
            ax.cla()
            # Plot b_points (blue) and d_points (magenta)
            b_x = [p['x'] for p in b_points]
            b_y = [p['y'] for p in b_points]
            ax.scatter(b_x, b_y, color='blue', s=5, label='B points')
            d_x = [p['x'] for p in d_points]
            d_y = [p['y'] for p in d_points]
            ax.scatter(d_x, d_y, color='magenta', s=5, label='D points')
            # Plot a_points (red) and c_points (orange)
            a_x = [a.x for a in a_points]
            a_y = [a.y for a in a_points]
            ax.scatter(a_x, a_y, color='red', s=10, label='A points')
            c_x = [c.x for c in c_points]
            c_y = [c.y for c in c_points]
            ax.scatter(c_x, c_y, color='orange', s=10, label='C points')
            # Draw AOE for each active A point (green) and C point (purple)
            for a in active_a:
                circle = plt.Circle((a.x, a.y), effective_aoe_radius, color='green', alpha=0.3)
                ax.add_patch(circle)
            for c in active_c:
                circle = plt.Circle((c.x, c.y), effective_aoe_radius, color='purple', alpha=0.3)
                ax.add_patch(circle)
            ax.set_xlim(-effective_s_radius, effective_s_radius)
            ax.set_ylim(-effective_s_radius, effective_s_radius)
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            ax.set_title(f"Cycle {cycle}")

            # Force a redraw and update the plot window
            plt.draw()
            plt.pause(0.5)  # Ensure the plot is updated in real-time after each cycle

        # --- Process a_points (react with b_points) first ---
        pending_a = set(active_a)
        while pending_a:
            proposals = {}
            remove_from_pending = set()
            for a in list(pending_a):
                candidates = [p for p in b_points if a.distance_to(p) <= effective_aoe_radius]
                if not candidates:
                    a.active = False
                    remove_from_pending.add(a)
                else:
                    chosen = random.choice(candidates)
                    proposals.setdefault(chosen['id'], []).append(a)
            pending_a -= remove_from_pending
            if not proposals:
                break
            for p_id, a_list in proposals.items():
                p_obj = next((p for p in b_points if p['id'] == p_id), None)
                if p_obj is None:
                    continue  # Candidate already taken.
                success_list = []
                for a in a_list:
                    if a in pending_a and random.random() < effective_success_prob:
                        success_list.append(a)
                if not success_list:
                    for a in a_list:
                        pending_a.discard(a)
                else:
                    winner = random.choice(success_list)
                    mutated_seq, mutation_occurred = process_mutation(winner.sequence,
                                                                      mutation_rate,
                                                                      mutation_probabilities)
                    # Update local_seq_list as before.
                    found = False
                    for local_dict in local_seq_list:
                        if local_dict["sequence"] == mutated_seq:
                            local_dict["N0"] += 1
                            found = True
                            break
                    if not found:
                        local_seq_list.append({"sequence": mutated_seq, "N0": 1})
                    # Create a new APoint using the winner's (possibly mutated) sequence.
                    new_a = APoint(mutated_seq, p_obj['x'], p_obj['y'])
                    a_points.append(new_a)
                    active_a.append(new_a)
                    # Remove the candidate B point.
                    b_points = [p for p in b_points if p['id'] != p_id]
                    for a in a_list:
                        pending_a.discard(a)
        active_a = [a for a in active_a if a.active]

        # --- Process c_points (react with d_points) second ---
        pending_c = set(active_c)
        while pending_c:
            proposals = {}
            remove_from_pending = set()
            for c in list(pending_c):
                candidates = [p for p in d_points if c.distance_to(p) <= effective_aoe_radius]
                if not candidates:
                    c.active = False
                    remove_from_pending.add(c)
                else:
                    chosen = random.choice(candidates)
                    proposals.setdefault(chosen['id'], []).append(c)
            pending_c -= remove_from_pending
            if not proposals:
                break
            for p_id, c_list in proposals.items():
                p_obj = next((p for p in d_points if p['id'] == p_id), None)
                if p_obj is None:
                    continue
                success_list = []
                for c in c_list:
                    if c in pending_c and random.random() < effective_success_prob:
                        success_list.append(c)
                if not success_list:
                    for c in c_list:
                        pending_c.discard(c)
                else:
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
                    # Create a new CPoint using the winner's (possibly mutated) sequence.
                    new_c = CPoint(mutated_seq, p_obj['x'], p_obj['y'])
                    c_points.append(new_c)
                    active_c.append(new_c)
                    # Remove the candidate D point.
                    d_points = [p for p in d_points if p['id'] != p_id]
                    for c in c_list:
                        pending_c.discard(c)
        active_c = [c for c in active_c if c.active]

        # Update the plot after each cycle
        plt.show(block=False)  # Allow continuous updates after each cycle
        cycle += 1

    # After the cycles are completed, show the final plot without AOEs
    if simulation_index:
        # Plot only points (no AOEs)
        fig, ax = plt.subplots()
        ax.cla()

        # Plot b_points (blue) and d_points (magenta)
        b_x = [p['x'] for p in b_points]
        b_y = [p['y'] for p in b_points]
        ax.scatter(b_x, b_y, color='blue', s=5, label='B points')
        d_x = [p['x'] for p in d_points]
        d_y = [p['y'] for p in d_points]
        ax.scatter(d_x, d_y, color='magenta', s=5, label='D points')

        # Plot a_points (red) and c_points (orange)
        a_x = [a.x for a in a_points]
        a_y = [a.y for a in a_points]
        ax.scatter(a_x, a_y, color='red', s=10, label='A points')
        c_x = [c.x for c in c_points]
        c_y = [c.y for c in c_points]
        ax.scatter(c_x, c_y, color='orange', s=10, label='C points')

        ax.set_xlim(-effective_s_radius, effective_s_radius)
        ax.set_ylim(-effective_s_radius, effective_s_radius)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title("Final State (Without AOEs)")
        plt.show(block=True)  # Show the plot after the last cycle is complete

    # After all cycles complete, merge the sequences
    all_cycle_counts.append(cycle_counts)

    # --- Merge this simulation's local_seq_list into the global merged_sequences ---
    for d in local_seq_list:
        found = False
        for md in merged_sequences:
            if check == 0:  # Ensure merged_sequences is empty at the start.
                merged_sequences = []
                check = 1
            if md["sequence"] == d["sequence"]:
                md["N0"] += d["N0"]
                found = True
                break
        if not found:
            merged_sequences.append(d)
    print(f"Length of merged_sequences: {len(merged_sequences)}")

    print(max(len(x) for x in all_cycle_counts))

    # Pad cycle_counts lists to the length of the longest list.
    max_length = max(len(x) for x in all_cycle_counts)
    padded_all_cycle_counts = []
    for lst in all_cycle_counts:
        if len(lst) < max_length:
            lst.extend([lst[-1]] * (max_length - len(lst)))
        padded_all_cycle_counts.append(lst)
    history_bridge = [sum(x) for x in zip(*all_cycle_counts)]
    print(f"Length of merged_sequences: {len(merged_sequences)}")
    return merged_sequences, history_bridge

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

    def simple(self, threshold):
        if not self.data:
            print("Error: Data is not initialized. Please provide a CSV file.")
            return None

        # Filter the data based on the threshold
        valid_sequences = [
            row for row in self.data if int(row['amount']) >= threshold
        ]

        # Save the filtered data to a CSV file
        with open('simple_result.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=valid_sequences[0].keys())
            writer.writeheader()
            writer.writerows(valid_sequences)

        print(f"Enhanced results saved to 'simple_result.csv' with {len(valid_sequences)} rows.")

        return valid_sequences

    def directional_networks(self, show=3):
        if not self.data:
            print("Error: Data is not initializded. Please provide a CSV file.")
            return None

        unique_rows = {row['Sequence']: row for row in self.data}.values()
        unique_molecules = len(unique_rows)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Add all nodes to the graph
        for row in unique_rows:
            before_graph.add_node(row['Sequence'],
                                  sequence=row['Sequence'],
                                  amount=int(row['amount']),
                                  molecule=row['molecule'])

        # Add edges based on the condition
        for row_a in unique_rows:
            for row_b in unique_rows:
                if row_a['Sequence'] != row_b['Sequence']:
                    value_a = int(row_a['amount'])
                    value_b = int(row_b['amount'])

                    if value_a >= 2 * value_b - 1:
                        before_graph.add_edge(row_a['Sequence'], row_b['Sequence'])

        print("Graph before filtering edges (value condition only):")
        print(f"Number of nodes: {before_graph.number_of_nodes()}, Number of edges: {before_graph.number_of_edges()}")

        # Show the "before" graph if requested
        if show in (1, 3):
            nx.draw_spring(before_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("Before Graph")
            plt.show()

        before_graph.remove_edges_from(nx.selfloop_edges(before_graph))

        # Initialize the "after" graph
        after_graph = before_graph.copy()

        # Remove edges from the "after" graph based on edit distance condition
        for u, v in list(after_graph.edges):
            sequence_u = after_graph.nodes[u]['sequence']
            sequence_v = after_graph.nodes[v]['sequence']
            edit_distance = self.levenshtein(sequence_u, sequence_v)
            if edit_distance >= 2:
                after_graph.remove_edge(u, v)

        print("Graph after filtering edges (edit distance < 2):")
        print(f"Number of nodes: {after_graph.number_of_nodes()}, Number of edges: {after_graph.number_of_edges()}")

        # Show the "after" graph if requested
        if show in (2, 3):
            nx.draw_spring(after_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("After Graph")
            plt.show()

        return before_graph, after_graph, unique_molecules

    @staticmethod
    def levenshtein(seq1, seq2):
        if len(seq1) < len(seq2):
            return Denoiser.levenshtein(seq2, seq1)

        if len(seq2) == 0:
            return len(seq1)

        previous_row = range(len(seq2) + 1)
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def networks_resolver(graph, toggle="central_node"):
        """
        Resolve networks in different ways based on the toggle parameter.

        Parameters:
        - graph: The NetworkX graph to analyze.
        - toggle: The resolution method to use (default: "central_node").

        Returns:
        - A list of dictionaries containing resolved network data.
        """
        central_nodes_data = []

        if toggle == "central_node":
            # Resolve networks using the central node approach
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                central_node = max(
                    subgraph.nodes(data=True), key=lambda x: x[1].get('amount', 0)
                )
                central_node_id = central_node[0]
                central_node_amount = central_node[1].get('amount', 0)
                sequence = central_node[1].get('sequence', '')

                # Calculate the total amount of all connected nodes
                total_amount = central_node_amount
                for node in subgraph.nodes(data=True):
                    total_amount += node[1].get('amount', 0)

                # Add the resolved data to the list
                central_nodes_data.append({
                    'Sequence': sequence,
                    'Central Node Count': central_node_amount,
                    'Network Nodes Count': total_amount
                })

            # Save the results to a CSV file
            with open('directional_results.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=central_nodes_data[0].keys())
                writer.writeheader()
                writer.writerows(central_nodes_data)

            print("Central nodes saved to 'directional_results.csv'.")

        else:
            print(f"Toggle '{toggle}' is not implemented.")
            return None

        return central_nodes_data

    def analysis(self, denoised_file, true_umis_file, amplified_umis_file="amplified_UMIs.csv"):
        def load_file(file):
            with open(file, mode='r') as f:
                return {row['Sequence'] for row in csv.DictReader(f)}

        sequences_denoised = load_file(denoised_file)
        sequences_true = load_file(true_umis_file)
        sequences_amplified = load_file(amplified_umis_file)

        tp = len(sequences_denoised & sequences_true)
        fn = len(sequences_true - sequences_denoised)
        fp = len(sequences_denoised - sequences_true)
        tn = len(sequences_amplified - sequences_denoised - sequences_true)

        cm = [[tp, fp], [fn, tn]]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"])
        plt.title("Confusion Matrix")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")

        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    @staticmethod
    def node_probe(graph, tier1=5, tier2=3):
        """
        Draws a subgraph centered on a randomly selected node with at least `tier1` connections.
        The subgraph includes tier-1 connections and up to `tier2` random tier-2 connections.

        Parameters:
        - graph: The input graph (NetworkX object) to probe.
        - tier1: Minimum number of connections required for a node to be considered (default: 5).
        - tier2: Maximum number of tier-2 connections to display for each tier-1 node (default: 3).

        Returns:
        - None. Displays the subgraph visualization.
        """
        # Find nodes with at least `tier1` connections
        eligible_nodes = [node for node in graph.nodes if len(list(graph.neighbors(node))) >= tier1]

        if not eligible_nodes:
            print(f"No nodes found with at least {tier1} connections.")
            return

        # Randomly select a node from eligible nodes
        selected_node = random.choice(eligible_nodes)
        print(f"Selected node: {selected_node}")

        # Get tier-1 connections
        tier1_nodes = list(graph.neighbors(selected_node))

        # Build the subgraph
        subgraph_nodes = {selected_node}
        subgraph_nodes.update(tier1_nodes)  # Add tier-1 nodes

        for tier1_node in tier1_nodes:
            tier2_neighbors = list(graph.neighbors(tier1_node))
            # Select up to `tier2` random tier-2 neighbors
            random_tier2 = random.sample(tier2_neighbors, min(len(tier2_neighbors), tier2))
            subgraph_nodes.update(random_tier2)

        # Create the subgraph
        subgraph = graph.subgraph(subgraph_nodes)

        # Visualize the subgraph
        print(f"Visualizing subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        nx.draw_spring(subgraph, with_labels=True, node_size=500, node_color="lightblue", font_size=10)
        plt.title("Node Probe Subgraph")
        plt.show()

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
    amplify_parser.add_argument('--method', type=str, choices=['pcr', 'bridge', 'polonies_amplification', 'both_12',
                                                               'both_13'], required=True,
                                help="Amplification method to use")
    amplify_parser.add_argument('--cycles', type=int, default=30, help="Number of amplification cycles")
    amplify_parser.add_argument('--mutation_rate', type=float, default=0.01,
                                help="Mutation rate per nucleotide per replication event")
    amplify_parser.add_argument('--substitution_prob', type=float, default=0.4,
                                help="Probability of substitution mutation")
    amplify_parser.add_argument('--deletion_prob', type=float, default=0.3, help="Probability of deletion mutation")
    amplify_parser.add_argument('--insertion_prob', type=float, default=0.3, help="Probability of insertion mutation")
    amplify_parser.add_argument('--substrate_capacity', type=float, default=(2 ** 18),
                                help="Initial substrate capacity")
    amplify_parser.add_argument('--S', type=float, default=700_000_000, help="Threshold S parameter")
    amplify_parser.add_argument('--input', type=str, default='true_barcodes.csv',
                                help="Input CSV filename with true barcodes")
    amplify_parser.add_argument('--plot', dest='plot', action='store_true', help="Enable plotting (default)",
                                default=True)
    amplify_parser.add_argument('--no_plot', dest='plot', action='store_false', help="Disable plotting")

    # Bridge amplification specific parsers:
    amplify_parser.add_argument('--S_radius', type=float, default=10,
                                help="Radius of S area where points are generated")
    amplify_parser.add_argument('--AOE_radius', type=float, default=1, help="Radius of AOE of every active A point")
    amplify_parser.add_argument('--simulate', action='store_true', help="Number of amplification cycles")
    amplify_parser.add_argument('--density', type=float, default=10, help="Density parameter for Bridge amplification")
    amplify_parser.add_argument('--success_prob', type=float, default=0.85,
                                help="Success probability for Bridge amplification")
    amplify_parser.add_argument('--deviation', type=float, default=0.1,
                                help="Deviation for Bridge amplification parameters (e.g., 0.1 for 10%)")
    amplify_parser.add_argument('--C', type=float, default=1e-9, help="Sharpness parameter C")

    # Subcommand: denoise
    denoise_parser = subparsers.add_parser('denoise', help="Denoise amplified sequences")
    denoise_parser.add_argument('--input', type=str, required=True, help="Input CSV filename for denoising")
    denoise_parser.add_argument('--method', type=str, choices=['simple', 'directional'], required=True,
                                help="Denoising method to use")
    denoise_parser.add_argument('--threshold', type=int, default=300, help="Threshold for simple denoising")
    denoise_parser.add_argument('--output', type=str, default='denoised.csv',
                                help="Output CSV filename for denoised sequences")
    denoise_parser.add_argument('--plot', dest='plot', action='store_true', help="Enable plotting (default)",
                                default=True)
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

        if args.method in ['pcr', 'both_12', 'both_13']:
            sequences_pcr = [dict(seq) for seq in sequences]
            logging.info("Starting PCR amplification...")
            sequences_pcr, history_pcr = pcr_amplification(
                sequences_pcr,
                cycles=args.cycles,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                substrate_capacity_initial=args.substrate_capacity,
                s=args.S,
                c=args.C,
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

        if args.method in ['bridge', 'both_12']:
            sequences_bridge = [dict(seq) for seq in sequences]
            logging.info("Starting Bridge amplification...")
            sequences_bridge_amp, history_bridge = bridge_amplification(
                sequences_bridge,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                simulate=args.simulate,
                s_radius=args.S_radius,
                aoe_radius=args.AOE_radius,
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

        if args.method in ['polonies_amplification', 'both_13']:
            sequences_bridge = [dict(seq) for seq in sequences]
            logging.info("Starting Bridge amplification...")
            sequences_polony_amp, hist_polony_amp = polonies_amplification(
                sequences_bridge,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                simulate=args.simulate,
                s_radius=args.S_radius,
                aoe_radius=args.AOE_radius,
                density=args.density,
                success_prob=args.success_prob,
                deviation=args.deviation,
            )
            polony_output = 'bridgeABCD_amplified.csv'
            with open(polony_output, 'w', newline='') as csvfile:
                fieldnames = ['sequence', 'N0']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for seq_dict in sequences_polony_amp:
                    writer.writerow(seq_dict)
            logging.info(f"Generated {len(sequences_polony_amp)} sequences and saved to {polony_output}")

        if args.method == 'both_12' and args.plot:
            plt.figure()
            plt.plot(range(1, len(history_pcr) + 1), history_pcr, marker='o', label='PCR Amplification')
            plt.plot(range(1, len(history_bridge) + 1), history_bridge, marker='s', label='Bridge Amplification')
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
