import logging
from support import compute_global_p, process_mutation, decode, encode
import os
import math
import random
from typing import List, Dict, Tuple, Any
import numpy as np
import csv
from scipy.spatial import cKDTree
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
    pcr_output = f"results1/pcr_{base}{ext}"
    return sequences, total_sequences_history, pcr_output


def animate_simulation(s_radius, density, true_barcodes, aoe_radius, success_prob):
    """
    Illustrative animation for simulation.

    - s_radius: Radius of the simulation circle.
    - density: Number of P points per unit area.
    - true_barcodes: List (or similar) of sequences representing initial A points.
    - aoe_radius: The area-of-effect radius for each A point.
    - success_prob: Probability that a connection attempt succeeds.

    The animation shows two frames per cycle:
      Frame 1: A points (red) and P points (blue) with AOE circles (green dashed).
      Frame 2: After connection attempts, P points that are connected are converted to A points.
    """
    # --- Generate initial A points from true_barcodes ---
    num_a = len(true_barcodes)
    # uniformly sample A points within a circle of radius s_radius
    a_r = s_radius * np.sqrt(np.random.rand(num_a))
    a_theta = 2 * np.pi * np.random.rand(num_a)
    a_x = a_r * np.cos(a_theta)
    a_y = a_r * np.sin(a_theta)
    # A points as an array of shape (num_a,2)
    a_points = np.column_stack((a_x, a_y))

    # --- Generate P points ---
    total_area = np.pi * s_radius ** 2
    num_p = int(density * total_area)
    p_r = s_radius * np.sqrt(np.random.rand(num_p))
    p_theta = 2 * np.pi * np.random.rand(num_p)
    p_x = p_r * np.cos(p_theta)
    p_y = p_r * np.sin(p_theta)
    p_points = np.column_stack((p_x, p_y))

    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-s_radius - 1, s_radius + 1)
    ax.set_ylim(-s_radius - 1, s_radius + 1)
    ax.set_aspect('equal')

    cycle = 0
    max_cycles = 10  # for illustration; in a real simulation you might continue until a condition is met.

    while cycle < max_cycles:
        cycle += 1
        # ---------------------------
        # Frame 1: Show current state
        # ---------------------------
        ax.clear()
        ax.set_xlim(-s_radius - 1, s_radius + 1)
        ax.set_ylim(-s_radius - 1, s_radius + 1)
        ax.set_title(f"Cycle {cycle} - Frame 1: AOE Display")
        # Plot A points (red) and P points (blue)
        if len(a_points) > 0:
            ax.scatter(a_points[:, 0], a_points[:, 1], color='red', label="A points", s=0.4)
        if len(p_points) > 0:
            ax.scatter(p_points[:, 0], p_points[:, 1], color='blue', label="P points", s=0.2)
        # Plot AOE circles around each A point (green dashed)
        for pt in a_points:
            circ = Circle((pt[0], pt[1]), aoe_radius, color='green', fill=False, linestyle='dashed', alpha=0.3)
            ax.add_patch(circ)
        ax.legend()
        plt.draw()
        plt.pause(2)  # pause to show frame 1

        # ---------------------------
        # Frame 2: Simulate one connection attempt per cycle.
        # ---------------------------
        # Identify candidate P points that are within the AOE of any A point.
        candidate_indices = []
        for i, pt in enumerate(p_points):
            distances = np.linalg.norm(a_points - pt, axis=1)
            in_aoe = np.where(distances <= aoe_radius)[0]
            if in_aoe.size > 0:
                candidate_indices.append(i)

        # Prepare lists for new A points and for remaining P points.
        new_a_points = []
        # We'll start by assuming all P points remain unless one is converted.
        remaining_p_points = p_points.copy()

        if candidate_indices:
            # Randomly choose one candidate P point from among those candidates.
            chosen_idx = random.choice(candidate_indices)
            chosen_pt = p_points[chosen_idx]

            # For the chosen P point, find which A points are within its AOE.
            distances = np.linalg.norm(a_points - chosen_pt, axis=1)
            in_aoe = np.where(distances <= aoe_radius)[0]

            # If there are any A points in range, randomly pick one (this resolves collision).
            if in_aoe.size > 0:
                candidate_a_idx = random.choice(in_aoe.tolist())
                # Attempt connection with success probability.
                if random.random() < success_prob:
                    # Connection successful: convert the chosen P point into an A point.
                    new_a_points.append(chosen_pt)
                    # Remove this P point from the pool.
                    remaining_p_points = np.delete(p_points, chosen_idx, axis=0)

        # Update pools: add the newly converted A point (if any) to the A points pool.
        if new_a_points:
            new_a_points = np.array(new_a_points)
            a_points = np.concatenate((a_points, new_a_points), axis=0)
        p_points = remaining_p_points

        # Now display Frame 2.
        ax.clear()
        ax.set_xlim(-s_radius - 1, s_radius + 1)
        ax.set_ylim(-s_radius - 1, s_radius + 1)
        ax.set_title(f"Cycle {cycle} - Frame 2: After One Connection")
        if len(a_points) > 0:
            ax.scatter(a_points[:, 0], a_points[:, 1], color='red', label="A points", s=0.4)
        if len(p_points) > 0:
            ax.scatter(p_points[:, 0], p_points[:, 1], color='blue', label="P points", s=0.2)
        ax.legend()
        plt.draw()
        plt.pause(1)

        # Optionally, you might decide to break if no more P points remain.
        if len(p_points) == 0:
            break

    plt.ioff()
    plt.show()


def generate_points(s_radius: float, density: float) -> np.ndarray:
    """
    Generate p_points as a numpy array of (x, y) coordinates uniformly distributed within
    a circle of radius s_radius. The total number is given by density * (π * s_radius^2).
    """
    total_points = int(density * math.pi * (s_radius ** 2))
    # Use polar coordinates for uniform sampling.
    r = s_radius * np.sqrt(np.random.rand(total_points))
    theta = 2 * np.pi * np.random.rand(total_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def generate_a_points(sequences: list, s_radius: float) -> np.ndarray:
    """
    Generate a numpy structured array for active a points.
    Each entry gets a random coordinate within the circle of radius s_radius and
    carries the sequence and its N0 value.
    """
    # Define a structured dtype: x, y (floats), sequence (object), N0 (int)
    dtype = np.dtype([('x', np.float64),
                      ('y', np.float64),
                      ('sequence', 'O'),
                      ('N0', np.int64),
                      ('id', np.int64),
                      ('parent_id', np.int64),
                      ('mutation_cycle', np.float64),
                      ('born', np.int32),
                      ('active', np.int32)])
    n = len(sequences)
    a_points = np.empty(n, dtype=dtype)

    # Generate random coordinates in circle
    r = s_radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    a_points['x'] = r * np.cos(theta)
    a_points['y'] = r * np.sin(theta)

    a_points['id'] = 0
    a_points['parent_id'] = 0
    a_points['mutation_cycle'] = 0
    a_points['born'] = 0
    a_points['active'] = 0

    # Assign sequence and N0 from input dictionaries.
    for i, seq_dict in enumerate(sequences):
        a_points['sequence'][i] = encode(seq_dict["sequence"])
        a_points['N0'][i] = seq_dict["N0"]
    return a_points


def generate_new_a_points(new_a_points_list: list) -> np.ndarray:
    """
    Generate a numpy structured array for active a points.
    Each entry gets a random coordinate within the circle of radius s_radius and
    carries the sequence and its N0 value.
    """
    # Define a structured dtype: x, y (floats), sequence (object), N0 (int)
    dtype = np.dtype([('x', np.float64),
                      ('y', np.float64),
                      ('sequence', 'O'),
                      ('N0', np.int64),
                      ('id', np.float64),
                      ('parent_id', np.float64),
                      ('mutation_cycle', np.float64),
                      ('born', np.int32),
                      ('active', np.int32)])
    n = len(new_a_points_list)
    new_a_points_array = np.empty(n, dtype=dtype)

    # Assign sequence and N0 from input dictionaries.
    for i, seq_dict in enumerate(new_a_points_list):
        new_a_points_array['sequence'][i] = seq_dict["sequence"]
        new_a_points_array['N0'][i] = seq_dict["N0"]
        new_a_points_array['x'][i] = seq_dict['x']
        new_a_points_array['y'][i] = seq_dict['y']
        new_a_points_array['id'][i] = seq_dict['id']
        new_a_points_array['parent_id'][i] = seq_dict['parent_id']
        new_a_points_array['mutation_cycle'][i] = seq_dict['mutation_cycle']
        new_a_points_array['born'][i] = seq_dict['born']
        new_a_points_array['active'][i] = seq_dict['active']
    return new_a_points_array


def save_a_points_to_file(folder_path: str, cycle_num: int, cleared_sequences: list):
    """
    Saves a list of dictionaries to a CSV file with the columns:
    sequence, N0, parent_id, mutation_cycle, active, id, born.

    Parameters:
        folder_path (str): The folder where the CSV file will be saved.
        cycle_num (int): A cycle number used to name the CSV file.
        cleared_sequences (list): A list of dictionaries. Each dictionary should have the following keys:
                                  'sequence', 'N0', 'parent_id', 'mutation_cycle', 'active', 'id', and 'born'.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Define the CSV file path with cycle number
    csv_file_path = os.path.join(folder_path, f'cleared_sequences{cycle_num}.csv')

    # Write the data to the CSV file with the specified column order
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header in the specified order
        writer.writerow(['sequence', 'N0', 'parent_id', 'mutation_cycle', 'active', 'id', 'born'])

        # Write each dictionary's values in the corresponding order
        for record in cleared_sequences:
            writer.writerow([
                record.get('sequence', ''),
                record.get('N0', ''),
                record.get('id', ''),
                record.get('parent_id', ''),
                record.get('mutation_cycle', ''),
                record.get('active', ''),
                record.get('born', '')
            ])


def conjugate_files(folder_path: str):
    # Create a list to hold the rows from all CSV files
    all_rows = []
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Open the CSV file and read its content
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                # Skip the header if it's not the first file
                if all_rows:
                    next(reader)  # Skip header row
                # Add all rows from this file to the all_rows list
                all_rows.extend(reader)

            # Delete the original CSV file
            os.remove(file_path)
    return all_rows


def polonies_amplification(s_radius: float,
                           density: float,
                           sequences: list,
                           aoe_radius: float,
                           success_prob: float,
                           deviation: float,
                           simulate: bool,
                           mutation_rate: float,
                           mutation_probabilities: Dict[str, float],
                           output: str = "final_output.csv"):

    # Generate initial points.
    bd_points = generate_points(s_radius, density)  # np.array
    ac_points = generate_a_points(sequences, s_radius)  # np.array

    # Create a pool of unique IDs.
    # Total pool size is the sum of all initial points (a + b points).
    total_possible_ids = len(ac_points) + len(bd_points)
    available_ids = list(range(total_possible_ids))
    random.shuffle(available_ids)  # Shuffle so that pop() returns a random id

    # Randomly permute indices
    indices1 = np.random.permutation(len(ac_points))
    indices2 = np.random.permutation(len(bd_points))

    # Compute the split index. If the length is odd, one array will have one more element.
    mid1 = len(ac_points) // 2
    mid2 = len(bd_points) // 2

    # Split the data using the permuted indices
    a_points_active = ac_points[indices1[:mid1]]
    c_points_active = ac_points[indices1[mid1:]]

    b_points = bd_points[indices2[:mid2]]
    d_points = bd_points[indices2[mid2:]]

    # Set id and parent_id fields for a_points_active and c_points_active.
    new_ids = [available_ids.pop() for _ in range(len(a_points_active))]
    a_points_active['id'] = new_ids
    a_points_active['parent_id'] = new_ids

    # Dictionary to collect sequences from a points that are removed (cleared from memory).
    folder_name = "results1/helping_folder"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Define the folder where the supporting files will be saved
    base, ext = os.path.splitext(output)
    folder_path_a = f"results1/helping_folder/a_supporting_{base}"
    folder_path_c = f"results1/helping_folder/c_supporting_{base}"
    aoe_radius = s_radius * aoe_radius/100
    # Define the output file for concatenated results
    output_file_path = f"results1/helping_folder/cleared_sequences_{base}.csv"
    cleared_sequences_outside = []
    cycle_num = 0
    saved_ids = set()
    while True:
        cleared_sequences = []
        cycle_num += 1
        print(f"Cycle {cycle_num}: {len(a_points_active)} active a points, {len(b_points)} remaining b points")
        print(f"Cycle {cycle_num}: {len(c_points_active)} active a points, {len(d_points)} remaining d points")

        # Two fail-safe if statements:
        # Termination: if no active a points remain, break.
        if len(a_points_active) == 0 or len(c_points_active) == 0:
            for arr in [a_points_active, c_points_active]:
                for point in arr:
                    seq = point['sequence']
                    cleared_sequences_outside.append({
                        "sequence": seq,
                        "N0": point['N0'],
                        "id": point['id'],
                        "parent_id": point['parent_id'],
                        "mutation_cycle": point['mutation_cycle'],
                        "born": point['born'],
                        "active": point['active']
                    })
            break

        # --- Determine pending p points ---
        # Search for nearby P points by building a KDTree on p_points.
        b_tree = cKDTree(b_points)
        d_tree = cKDTree(d_points)
        # Get coordinates of active a points.
        coords_a = np.column_stack((a_points_active['x'], a_points_active['y']))
        coords_c = np.column_stack((c_points_active['x'], c_points_active['y']))
        # For each active a point, get indices of p_points within its aoe.
        query_results_b = b_tree.query_ball_point(coords_a, r=aoe_radius)
        query_results_d = d_tree.query_ball_point(coords_c, r=aoe_radius)

        # Identify a_points that do NOT have any p point within their aoe.
        no_pending_mask_b = np.array([len(indices) == 0 for indices in query_results_b])
        no_pending_mask_d = np.array([len(indices) == 0 for indices in query_results_d])

        if np.any(no_pending_mask_b):
            # Remove these a points and add their sequences to cleared_sequences.
            for point in a_points_active[no_pending_mask_b]:
                seq = point['sequence']
                cleared_sequences.append({
                    "sequence": seq,
                    "N0": point['N0'],
                    "id": point['id'],
                    "parent_id": point['parent_id'],
                    "mutation_cycle": point['mutation_cycle'],
                    "born": point['born'],
                    "active": point['active']
                })
            # save inactive a points to cycle specific csv file
            save_a_points_to_file(folder_path_a, cycle_num, cleared_sequences)
            # Add saved IDs to the set
            for record in cleared_sequences:
                saved_ids.add(record["id"])
            # Keep only those with at least one nearby p point.
            a_points_active = a_points_active[~no_pending_mask_b]
            # Also update coords_a and query_results accordingly.
            coords_a = np.column_stack((a_points_active['x'], a_points_active['y']))
            query_results_b = b_tree.query_ball_point(coords_a, r=aoe_radius)

        if np.any(no_pending_mask_d):
            # Remove these a points and add their sequences to cleared_sequences.
            for point in c_points_active[no_pending_mask_d]:
                seq = point['sequence']
                cleared_sequences.append({
                    "sequence": seq,
                    "N0": point['N0'],
                    "id": point['id'],
                    "parent_id": point['parent_id'],
                    "mutation_cycle": point['mutation_cycle'],
                    "born": point['born'],
                    "active": point['active']
                })
            # save inactive a points to cycle specific csv file
            save_a_points_to_file(folder_path_c, cycle_num, cleared_sequences)
            # Add saved IDs to the set
            for record in cleared_sequences:
                saved_ids.add(record["id"])
            # Keep only those with at least one nearby p point.
            c_points_active = c_points_active[~no_pending_mask_d]
            # Also update coords_a and query_results accordingly.
            coords_c = np.column_stack((c_points_active['x'], c_points_active['y']))
            query_results_d = d_tree.query_ball_point(coords_c, r=aoe_radius)

        a_points_active["active"] += 1
        c_points_active["active"] += 1

        # Create a boolean mask for b_points that are within any active a point's aoe.
        pending_mask_b = np.zeros(len(b_points), dtype=bool)
        for indices in query_results_b:
            if indices:
                pending_mask_b[indices] = True
        # Extract pending points.
        pending_b_points = b_points[pending_mask_b]
        # Remove these from b_points.
        b_points = b_points[~pending_mask_b]
        print(f"{len(b_points)} left after removing pending points.")

        # Create a boolean mask for d_points that are within any active c point's aoe.
        pending_mask_d = np.zeros(len(d_points), dtype=bool)
        for indices in query_results_d:
            if indices:
                pending_mask_d[indices] = True
        # Extract pending points.
        pending_d_points = d_points[pending_mask_d]
        # Remove these from p_points.
        d_points = d_points[~pending_mask_d]
        print(f"{len(d_points)} left after removing pending points.")

        # --- Connection attempts ---
        next_active_a_points_list = []
        next_active_c_points_list = []
        while (pending_b_points.size > 0 and
               a_points_active.size > 0 and
               pending_d_points.size > 0 and
               c_points_active.size > 0):            # Build a KDTree for the current pending p points.
            pending_tree_b = cKDTree(pending_b_points)
            pending_tree_d = cKDTree(pending_d_points)

            # Record connection attempts:
            connection_attempts_a = {}
            for i, point in enumerate(a_points_active):
                pt = [point['x'], point['y']]
                candidate_indices = pending_tree_b.query_ball_point(pt, r=aoe_radius)
                if candidate_indices:
                    chosen = random.choice(candidate_indices)
                    connection_attempts_a.setdefault(chosen, []).append(i)
            # If no a point found any pending p point, break out.
            if not connection_attempts_a:
                break

            # Record connection attempts:
            connection_attempts_c = {}
            for i, point in enumerate(c_points_active):
                pt = [point['x'], point['y']]
                candidate_indices = pending_tree_d.query_ball_point(pt, r=aoe_radius)
                if candidate_indices:
                    chosen = random.choice(candidate_indices)
                    connection_attempts_c.setdefault(chosen, []).append(i)
            # If no a point found any pending p point, break out.
            if not connection_attempts_c:
                for arr in [a_points_active, c_points_active]:
                    for point in arr:
                        seq = point['sequence']
                        cleared_sequences_outside.append({
                            "sequence": seq,
                            "N0": point['N0'],
                            "id": point['id'],
                            "parent_id": point['parent_id'],
                            "mutation_cycle": point['mutation_cycle'],
                            "born": point['born'],
                            "active": point['active']
                        })
                break

            # Collision resolution and success check.
            successful_connections_ab = []  # list of tuples: (a_point_index, pending_index)
            for pending_idx, a_indices in connection_attempts_a.items():
                chosen_a_index = random.choice(a_indices)
                if random.random() < success_prob:
                    successful_connections_ab.append((chosen_a_index, pending_idx))

            # Collision resolution and success check.
            successful_connections_cd = []  # list of tuples: (a_point_index, pending_index)
            for pending_idx, c_indices in connection_attempts_c.items():
                chosen_a_index = random.choice(c_indices)
                if random.random() < success_prob:
                    successful_connections_cd.append((chosen_a_index, pending_idx))

            if not successful_connections_cd and not successful_connections_ab:
                break

            indices_to_remove_a = set()
            pending_indices_to_remove_b = set()
            for a_idx, pending_idx in successful_connections_ab:
                a_point = a_points_active[a_idx]
                b_point = pending_b_points[pending_idx]
                new_seq, mutated = process_mutation(a_point['sequence'], mutation_rate, mutation_probabilities)
                # Create a new point with the p_point's coordinates.
                new_point = {
                    "x": b_point[0],
                    "y": b_point[1],
                    "sequence": new_seq,
                    "N0": 1,
                    "id": available_ids.pop(),  # assign a new unique id
                    "parent_id": a_point["id"],  # parent's id
                    "mutation_cycle": cycle_num,
                    "born": cycle_num,
                    "active": 0
                }
                next_active_a_points_list.append(a_point)
                next_active_a_points_list.append(new_point)
                indices_to_remove_a.add(a_idx)
                pending_indices_to_remove_b.add(pending_idx)

            indices_to_remove_c = set()
            pending_indices_to_remove_d = set()
            for c_idx, pending_idx in successful_connections_cd:
                c_point = c_points_active[c_idx]
                d_point = pending_d_points[pending_idx]
                new_seq, mutated = process_mutation(c_point['sequence'], mutation_rate, mutation_probabilities)
                # Create a new point with the p_point's coordinates.
                new_point = {
                    "x": d_point[0],
                    "y": d_point[1],
                    "sequence": new_seq,
                    "N0": 1,
                    "id": available_ids.pop(),  # assign a new unique id
                    "parent_id": c_point["id"],  # parent's id
                    "mutation_cycle": cycle_num,
                    "born": cycle_num,
                    "active": 0
                }
                next_active_c_points_list.append(c_point)
                next_active_c_points_list.append(new_point)
                indices_to_remove_c.add(c_idx)
                pending_indices_to_remove_d.add(pending_idx)

            # Remove the successful a points from a_points_active.
            if indices_to_remove_a:
                mask = np.ones(len(a_points_active), dtype=bool)
                mask[list(indices_to_remove_a)] = False
                a_points_active = a_points_active[mask]

            # Remove the successful c points from c_points_active.
            if indices_to_remove_c:
                mask = np.ones(len(c_points_active), dtype=bool)
                mask[list(indices_to_remove_c)] = False
                c_points_active = c_points_active[mask]

            # Remove used pending p points.
            if len(pending_indices_to_remove_b) > 0:
                mask_pending = np.ones(len(pending_b_points), dtype=bool)
                mask_pending[list(pending_indices_to_remove_b)] = False
                pending_b_points = pending_b_points[mask_pending]

            # Remove used pending p points.
            if len(pending_indices_to_remove_d) > 0:
                mask_pending = np.ones(len(pending_d_points), dtype=bool)
                mask_pending[list(pending_indices_to_remove_d)] = False
                pending_d_points = pending_d_points[mask_pending]

            # (Remaining a_points_active will try again if there are still pending p points.)


        # *** Merge leftover pending points back into the main pool ***
        # Any pending p_points that were not used are returned to p_points.
        b_points = np.concatenate((b_points, pending_b_points))
        print(f"After merging, b_points has {len(b_points)} points.")

        d_points = np.concatenate((d_points, pending_d_points))
        print(f"After merging, d_points has {len(d_points)} points.")

        if next_active_a_points_list:
            a_points_active = generate_new_a_points(next_active_a_points_list)
        else:
            a_points_active = np.empty(0, dtype=a_points_active.dtype)

        if next_active_c_points_list:
            c_points_active = generate_new_a_points(next_active_c_points_list)
        else:
            c_points_active = np.empty(0, dtype=c_points_active.dtype)
        # Else, leave c_points_active unchanged.
        if cycle_num == 1 and simulate:
            true_barcodes = "true_barcodes.csv"
            animate_simulation(s_radius, density, true_barcodes, aoe_radius, success_prob)

    # conjugation of files from each cycle
    rows_a = conjugate_files(folder_path_a)
    rows_c = conjugate_files(folder_path_c)
    all_rows = rows_a + rows_c

    # Save remaining active a and c points to cleared_sequences
    # Save remaining active a and c points to separate cleared_sequences lists
    for arr in [a_points_active, c_points_active]:
        local_cleared = []
        for point in arr:
            local_cleared.append({
                "sequence": point['sequence'],
                "N0": point['N0'],
                "id": point['id'],
                "parent_id": point['parent_id'],
                "mutation_cycle": point['mutation_cycle'],
                "born": point['born'],
                "active": point['active']
            })
        if arr is a_points_active:
            save_a_points_to_file(folder_path_a, cycle_num + 1, local_cleared)
        elif arr is c_points_active:
            save_a_points_to_file(folder_path_c, cycle_num + 1, local_cleared)

    # Write the combined rows into the final 'cleared_sequences.csv' file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with all the desired fields.
        writer.writerow(['sequence', 'N0', 'id', 'parent_id', 'mutation_cycle', 'born', 'active'])
        # Write the combined data (each row should have 7 elements in the correct order)
        writer.writerows(all_rows)

    # Delete the folder after the files are combined and deleted
    shutil.rmtree(folder_path_a)
    shutil.rmtree(folder_path_c)

    # End simulation cycles.
    print("Simulation ended.")

    seen = set()
    sequences_polony_amp = []

    with open(output_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        sequences_polony_amp = []

        for row in reader:
            if row["sequence"] == "sequence":
                continue  # Skip potential duplicate headers

            decoded_seq = decode(row["sequence"])
            entry = {
                "sequence": decoded_seq,
                "N0": int(row["N0"]),
                "id": row["id"],
                "parent_id": row["parent_id"],
                "mutation_cycle": row["mutation_cycle"],
                "born": row["born"],
                "active": row["active"]
            }
            sequences_polony_amp.append(entry)

    polonies_output = f"results1/polonies_{base}{ext}"

    return sequences_polony_amp, polonies_output
