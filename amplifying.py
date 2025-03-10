import logging
import matplotlib.pyplot as plt
import random
import math
from typing import List, Dict, Tuple, Any
from support import compute_global_p, process_mutation
import os
import concurrent.futures

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


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
    pcr_output = f"results/pcr_{base}{ext}"
    return sequences, total_sequences_history, pcr_output


def simulate_seq(seq_dict: Dict[str, Any],
                 num_p: int,
                 effective_s_radius: float,
                 aoe_radius: float,
                 effective_aoe_radius: float,
                 effective_success_prob: float,
                 mutation_rate: float,
                 mutation_probabilities: Dict[str, float]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Simulate bridge amplification for a single sequence entry.
    Returns a tuple: (local_seq_list, cycle_counts) where local_seq_list is a list of sequence dicts
    and cycle_counts is a list of total active A points per cycle.
    """
    # Generate P points uniformly in a circle.
    p_points = []
    for i in range(num_p):
        r = effective_s_radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        p_points.append({'id': i, 'x': x, 'y': y})
    global_p = p_points  # Available P points

    # Define the A point class.
    class APoint:
        def __init__(self, sequence, x, y):
            self.sequence = sequence
            self.x = x
            self.y = y
            self.active = True  # Remains active while candidates exist

        def distance_to(self, p):
            return math.hypot(self.x - p['x'], self.y - p['y'])

    # Initialize simulation with one A point at the center.
    a_points = []
    local_seq_list = [{"sequence": seq_dict["sequence"], "N0": seq_dict["N0"]}]
    initial_a = APoint(seq_dict["sequence"], 0, 0)
    a_points.append(initial_a)
    active_a = [initial_a]

    # Track the number of A points at each cycle.
    cycle_counts = []
    cycle = 0

    # Main simulation loop.
    while global_p and active_a:
        cycle_counts.append(len(a_points))

        # Each active A checks for at least one available P point within its AOE.
        for a in active_a:
            candidates = [p for p in global_p if a.distance_to(p) <= aoe_radius]
            if not candidates:
                a.active = False

        active_a = [a for a in active_a if a.active]
        if not active_a:
            break

        pending = set(active_a)
        # (Optional) Print number of pending A points.

        # Process connection proposals.
        while pending:
            proposals = {}
            remove_from_pending = set()
            for a in list(pending):
                candidates = [p for p in global_p if a.distance_to(p) <= effective_aoe_radius]
                if not candidates:
                    remove_from_pending.add(a)
                else:
                    chosen = random.choice(candidates)
                    proposals.setdefault(chosen['id'], []).append(a)
            pending -= remove_from_pending
            if not proposals:
                break
            for p_id, a_list in proposals.items():
                p_obj = next((p for p in global_p if p['id'] == p_id), None)
                if p_obj is None:
                    continue  # Candidate already taken.
                success_list = []
                for a in a_list:
                    if a in pending and random.random() < effective_success_prob:
                        success_list.append(a)
                if not success_list:
                    for a in a_list:
                        pending.discard(a)
                else:
                    winner = random.choice(success_list)
                    mutated_seq, mutation_occurred = process_mutation(
                        winner.sequence,
                        mutation_rate,
                        mutation_probabilities
                    )
                    # Update the local sequence list.
                    found = False
                    for local_dict in local_seq_list:
                        if local_dict["sequence"] == mutated_seq:
                            local_dict["N0"] += 1
                            found = True
                            break
                    if not found:
                        local_seq_list.append({"sequence": mutated_seq, "N0": 1})
                    # Create a new A point.
                    new_a = APoint(mutated_seq, p_obj['x'], p_obj['y'])
                    a_points.append(new_a)
                    active_a.append(new_a)
                    # Remove the candidate P point.
                    global_p = [p for p in global_p if p['id'] != p_id]
                    for a in a_list:
                        pending.discard(a)
        active_a = [a for a in a_points if a.active]
        cycle += 1

    return local_seq_list, cycle_counts


def bridge_amplification(sequences: List[Dict[str, Any]],
                         simulate: bool,
                         mutation_rate: float,
                         mutation_probabilities: Dict[str, float],
                         s_radius: float,
                         aoe_radius: float,
                         density: float,
                         success_prob: float,
                         deviation: float,
                         output: str) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """
    Perform Bridge amplification simulation.
    Each cycle applies a random deviation (±10% by default) to parameters S, density, and success probability.
    The effective success probability (after deviation) is used as the chance for amplification.
    Returns the final merged sequence list and a history of total unique sequences per cycle.
    """
    simulation_index = simulate  # To track whether to animate (unused in parallel execution)
    # Calculate common parameters (applied to all simulations).
    effective_s_radius = s_radius * (1 + random.uniform(-deviation, deviation))
    effective_density = density * (1 + random.uniform(-deviation, deviation))
    effective_s = math.pi * effective_s_radius ** 2
    num_p = int(effective_density * effective_s)
    effective_success_prob = success_prob * (1 + random.uniform(-deviation, deviation))
    effective_success_prob = min(effective_success_prob, 1.0)
    effective_aoe_radius = aoe_radius * (1 + random.uniform(-deviation, deviation))

    # Use parallel processing to run each simulation (each seq_dict) independently.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            simulate_seq,
            sequences,
            [num_p] * len(sequences),
            [effective_s_radius] * len(sequences),
            [aoe_radius] * len(sequences),
            [effective_aoe_radius] * len(sequences),
            [effective_success_prob] * len(sequences),
            [mutation_rate] * len(sequences),
            [mutation_probabilities] * len(sequences)
        ))

    # Merge results from all simulations.
    merged_sequences = []
    all_cycle_counts = []
    for local_seq_list, cycle_counts in results:
        all_cycle_counts.append(cycle_counts)
        for d in local_seq_list:
            found = False
            for md in merged_sequences:
                if md["sequence"] == d["sequence"]:
                    md["N0"] += d["N0"]
                    found = True
                    break
            if not found:
                merged_sequences.append(d)

    # Determine the maximum cycle count length and pad each cycle_counts list.
    if all_cycle_counts:
        max_length = max(len(x) for x in all_cycle_counts)
        padded_all_cycle_counts = []
        for lst in all_cycle_counts:
            if len(lst) < max_length:
                lst.extend([lst[-1]] * (max_length - len(lst)))
            padded_all_cycle_counts.append(lst)
        # Compute element-wise sum over all cycle_counts lists.
        history_bridge = [sum(x) for x in zip(*padded_all_cycle_counts)]
    else:
        history_bridge = []

    base, ext = os.path.splitext(output)
    bridge_output = f"results/bridge_{base}{ext}"
    print(f"Length of merged_sequences: {len(merged_sequences)}")
    return merged_sequences, history_bridge, bridge_output


def polonies_amplification(sequences: List[Dict[str, Any]],
                           simulate: bool,
                           mutation_rate: float,
                           mutation_probabilities: Dict[str, float],
                           s_radius: float,
                           aoe_radius: float,
                           density: float,
                           success_prob: float,
                           deviation: float,
                           output: str) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """
    Perform modified Bridge amplification simulation with visual representation of points and AOEs at each cycle.
    Now P points are divided into B and D points.
    A_point (reacting with B points) and C_point (reacting with D points)
    are created from the same starting sequence.
    A_point acts first in each cycle and C_point acts second.
    Points are displayed with different colors.
    """
    simulation_index = simulate  # To track whether to animate.
    all_cycle_counts = []
    merged_sequences = []
    # Calculate parameters.
    effective_s_radius = s_radius * (1 + random.uniform(-deviation, deviation))
    effective_density = density * (1 + random.uniform(-deviation, deviation))
    effective_s = math.pi * effective_s_radius ** 2
    num_p = int(effective_density * effective_s)
    effective_success_prob = success_prob * (1 + random.uniform(-deviation, deviation))
    effective_success_prob = min(effective_success_prob, 1.0)
    effective_aoe_radius = aoe_radius * (1 + random.uniform(-deviation, deviation))

    # Divide generated P points into b_points and d_points.
    b_points = []
    d_points = []
    for i in range(num_p):
        r = effective_s_radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        point = {'id': i, 'x': x, 'y': y}
        if random.random() < 0.5:
            b_points.append(point)
        else:
            d_points.append(point)

    # Define the A point (reacts with B points) and C point (reacts with D points) classes.
    class APoint:
        def __init__(self, sequence, x, y):
            self.sequence = sequence
            self.x = x
            self.y = y
            self.active = True

        def distance_to(self, p):
            return math.hypot(self.x - p['x'], self.y - p['y'])

    class CPoint:
        def __init__(self, sequence, x, y):
            self.sequence = sequence
            self.x = x
            self.y = y
            self.active = True

        def distance_to(self, p):
            return math.hypot(self.x - p['x'], self.y - p['y'])

    # Initialize with one A_point and one C_point at the center.
    local_seq_list = sequences  # Assume sequences already has the starting sequences.
    initial_a = []
    initial_c = []
    for i in range(0, len(local_seq_list) - 1, 2):
        dict_n = local_seq_list[i]
        dict_n_plus_1 = local_seq_list[i + 1]
        r_a = effective_s_radius * math.sqrt(random.random())
        theta_a = random.uniform(0, 2 * math.pi)
        x_a = r_a * math.cos(theta_a)
        y_a = r_a * math.sin(theta_a)
        r_c = effective_s_radius * math.sqrt(random.random())
        theta_c = random.uniform(0, 2 * math.pi)
        x_c = r_c * math.cos(theta_c)
        y_c = r_c * math.sin(theta_c)
        new_a = APoint(dict_n["sequence"], x_a, y_a)
        new_c = CPoint(dict_n_plus_1["sequence"], x_c, y_c)
        initial_a.append(new_a)
        initial_c.append(new_c)

    a_points = initial_a
    c_points = initial_c
    active_a = initial_a
    active_c = initial_c

    cycle_counts = []
    cycle = 0

    if simulation_index:
        fig, ax = plt.subplots()

    while (b_points and active_a) or (d_points and active_c):
        cycle_counts.append(len(a_points) + len(c_points))
        if simulation_index:
            ax.cla()
            b_x = [p['x'] for p in b_points]
            b_y = [p['y'] for p in b_points]
            ax.scatter(b_x, b_y, color='blue', s=5, label='B points')
            d_x = [p['x'] for p in d_points]
            d_y = [p['y'] for p in d_points]
            ax.scatter(d_x, d_y, color='magenta', s=5, label='D points')
            a_x = [a.x for a in a_points]
            a_y = [a.y for a in a_points]
            ax.scatter(a_x, a_y, color='red', s=10, label='A points')
            c_x = [c.x for c in c_points]
            c_y = [c.y for c in c_points]
            ax.scatter(c_x, c_y, color='orange', s=10, label='C points')
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
            plt.draw()
            plt.pause(0.5)

        # Process A points (react with B points)
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
                    continue
                success_list = []
                for a in a_list:
                    if a in pending_a and random.random() < effective_success_prob:
                        success_list.append(a)
                if not success_list:
                    for a in a_list:
                        pending_a.discard(a)
                else:
                    winner = random.choice(success_list)
                    mutated_seq, mutation_occurred = process_mutation(
                        winner.sequence,
                        mutation_rate,
                        mutation_probabilities
                    )
                    found = False
                    for local_dict in local_seq_list:
                        if local_dict["sequence"] == mutated_seq:
                            local_dict["N0"] += 1
                            found = True
                            break
                    if not found:
                        local_seq_list.append({"sequence": mutated_seq, "N0": 1})
                    new_a = APoint(mutated_seq, p_obj['x'], p_obj['y'])
                    a_points.append(new_a)
                    active_a.append(new_a)
                    b_points = [p for p in b_points if p['id'] != p_id]
                    for a in a_list:
                        pending_a.discard(a)
        active_a = [a for a in active_a if a.active]

        # Process C points (react with D points)
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
                    mutated_seq, mutation_occurred = process_mutation(
                        winner.sequence,
                        mutation_rate,
                        mutation_probabilities
                    )
                    found = False
                    for local_dict in local_seq_list:
                        if local_dict["sequence"] == mutated_seq:
                            local_dict["N0"] += 1
                            found = True
                            break
                    if not found:
                        local_seq_list.append({"sequence": mutated_seq, "N0": 1})
                    new_c = CPoint(mutated_seq, p_obj['x'], p_obj['y'])
                    c_points.append(new_c)
                    active_c.append(new_c)
                    d_points = [p for p in d_points if p['id'] != p_id]
                    for c in c_list:
                        pending_c.discard(c)
        active_c = [c for c in active_c if c.active]

        plt.show(block=False)
        cycle += 1

    if simulation_index:
        fig, ax = plt.subplots()
        ax.cla()
        b_x = [p['x'] for p in b_points]
        b_y = [p['y'] for p in b_points]
        ax.scatter(b_x, b_y, color='blue', s=5, label='B points')
        d_x = [p['x'] for p in d_points]
        d_y = [p['y'] for p in d_points]
        ax.scatter(d_x, d_y, color='magenta', s=5, label='D points')
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
        plt.show(block=True)

    all_cycle_counts.append(cycle_counts)
    for d in local_seq_list:
        found = False
        for md in merged_sequences:
            if md["sequence"] == d["sequence"]:
                md["N0"] += d["N0"]
                found = True
                break
        if not found:
            merged_sequences.append(d)
    print(f"Length of merged_sequences: {len(merged_sequences)}")
    if all_cycle_counts:
        max_length = max(len(x) for x in all_cycle_counts)
        padded_all_cycle_counts = []
        for lst in all_cycle_counts:
            if len(lst) < max_length:
                lst.extend([lst[-1]] * (max_length - len(lst)))
            padded_all_cycle_counts.append(lst)
        history_bridge = [sum(x) for x in zip(*padded_all_cycle_counts)]
    else:
        history_bridge = []
    base, ext = os.path.splitext(output)
    polonies_output = f"results/polonies_{base}{ext}"
    return merged_sequences, history_bridge, polonies_output
