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
    pcr_output = f"results/pcr_{base}{ext}"
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
    dtype = np.dtype([('x', np.float64), ('y', np.float64), ('sequence', 'O'), ('N0', np.int64)])
    n = len(sequences)
    a_points = np.empty(n, dtype=dtype)

    # Generate random coordinates in circle
    r = s_radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    a_points['x'] = r * np.cos(theta)
    a_points['y'] = r * np.sin(theta)

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
    dtype = np.dtype([('x', np.float64), ('y', np.float64), ('sequence', 'O'), ('N0', np.int64)])
    n = len(new_a_points_list)
    new_a_points_array = np.empty(n, dtype=dtype)

    # Assign sequence and N0 from input dictionaries.
    for i, seq_dict in enumerate(new_a_points_list):
        new_a_points_array['sequence'][i] = seq_dict["sequence"]
        new_a_points_array['N0'][i] = seq_dict["N0"]
        new_a_points_array['x'][i] = seq_dict['x']
        new_a_points_array['y'][i] = seq_dict['y']
    return new_a_points_array


def save_a_points_to_file(folder_path: str, cycle_num: int, cleared_sequences: dict):
    # make sure the folder exists, if not, create it
    os.makedirs(folder_path, exist_ok=True)

    # Define the CSV file path with cycle number
    csv_file_path = os.path.join(folder_path, f'cleared_sequences{cycle_num}.csv')

    # Write the dictionary to a CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['sequence', 'N0'])

        # Write the data (sequence, N0)
        for seq, N0 in cleared_sequences.items():
            writer.writerow([seq, N0])


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
    """
    Runs the simulation.

    - p_points: All points within the circle of radius s_radius (generated using density).
    - a_points_active: Each active a point (with coordinate, sequence, and N0) is generated from
      the input list 'sequences'.
    - aoe_radius is directly provided and represents the radius of the area of effect for each a point.
    - In each cycle, all p_points that lie within any a point's aoe are removed from p_points and
      assigned to pending_p_points. Then every active a point tries to pick one p point (from pending)
      that lies within its own aoe – if successful (according to success_prob) then that connection
      is accepted (collisions resolved randomly), processed via process_mutation, and the "a" point is
      moved to the next cycle. Active a points that at the beginning of a cycle have no p point
      in their aoe are removed (their sequence is saved for CSV output).
    - The simulation repeats cycles until no active a points remain.
    - Finally, a CSV file is written with header: sequence,N0, where duplicate sequences are
      collapsed with N0 counts summed.
    """
    # Generate initial points.
    p_points = generate_points(s_radius, density)  # np.array
    a_points_active = generate_a_points(sequences, s_radius)  # np.array

    # Dictionary to collect sequences from a points that are removed (cleared from memory).
    cleared_sequences = {}  # key: sequence, value: cumulative N0

    # Define the folder where the supporting files will be saved
    folder_path = 'supporting_folder'
    aoe_radius = s_radius * aoe_radius/100
    # Define the output file for concatenated results
    output_file_path = 'cleared_sequences.csv'

    cycle_num = 0
    while True:
        cycle_num += 1
        print(f"Cycle {cycle_num}: {len(a_points_active)} active a points, {len(p_points)} remaining p points")

        # Two fail-safe if statements:
        # Termination: if no active a points remain, break.
        if len(a_points_active) == 0:
            break

        # --- Determine pending p points ---
        # Build a KDTree on p_points.
        p_tree = cKDTree(p_points)
        # Get coordinates of active a points.
        coords_a = np.column_stack((a_points_active['x'], a_points_active['y']))
        # For each active a point, get indices of p_points within its aoe.
        query_results = p_tree.query_ball_point(coords_a, r=aoe_radius)

        # Identify a_points that do NOT have any p point within their aoe.
        no_pending_mask = np.array([len(indices) == 0 for indices in query_results])
        if np.any(no_pending_mask):
            # Remove these a points and add their sequences to cleared_sequences.
            for point in a_points_active[no_pending_mask]:
                seq = point['sequence']
                cleared_sequences[seq] = cleared_sequences.get(seq, 0) + point['N0']

            # save inactive a points to cycle specific csv file
            save_a_points_to_file(folder_path, cycle_num, cleared_sequences)

            # Keep only those with at least one nearby p point.
            a_points_active = a_points_active[~no_pending_mask]
            # Also update coords_a and query_results accordingly.
            coords_a = np.column_stack((a_points_active['x'], a_points_active['y']))
            query_results = p_tree.query_ball_point(coords_a, r=aoe_radius)

        # Create a boolean mask for p_points that are within any active a point's aoe.
        pending_mask = np.zeros(len(p_points), dtype=bool)
        for indices in query_results:
            if indices:
                pending_mask[indices] = True
        # Extract pending points.
        pending_p_points = p_points[pending_mask]
        # Remove these from p_points.
        p_points = p_points[~pending_mask]
        print(f"{len(p_points)} left after removing pending points.")

        # --- Connection attempts ---
        next_active_a_points_list = []
        while len(pending_p_points) > 0 and len(a_points_active) > 0:
            # Build a KDTree for the current pending p points.
            pending_tree = cKDTree(pending_p_points)

            # Record connection attempts:
            connection_attempts = {}
            for i, point in enumerate(a_points_active):
                pt = [point['x'], point['y']]
                candidate_indices = pending_tree.query_ball_point(pt, r=aoe_radius)
                if candidate_indices:
                    chosen = random.choice(candidate_indices)
                    connection_attempts.setdefault(chosen, []).append(i)
            # If no a point found any pending p point, break out.
            if not connection_attempts:
                break

            # Collision resolution and success check.
            successful_connections = []  # list of tuples: (a_point_index, pending_index)
            for pending_idx, a_indices in connection_attempts.items():
                chosen_a_index = random.choice(a_indices)
                if random.random() < success_prob:
                    successful_connections.append((chosen_a_index, pending_idx))

            if not successful_connections:
                break

            indices_to_remove = set()
            pending_indices_to_remove = set()
            for a_idx, pending_idx in successful_connections:
                a_point = a_points_active[a_idx]
                p_point = pending_p_points[pending_idx]
                new_seq, mutated = process_mutation(a_point['sequence'], mutation_rate, mutation_probabilities)
                # Create a new point with the p_point's coordinates.
                new_point = {"x": p_point[0], "y": p_point[1], "sequence": new_seq, "N0": 1}
                next_active_a_points_list.append(a_point)
                next_active_a_points_list.append(new_point)
                indices_to_remove.add(a_idx)
                pending_indices_to_remove.add(pending_idx)

            # Remove the successful a points from a_points_active.
            if indices_to_remove:
                mask = np.ones(len(a_points_active), dtype=bool)
                mask[list(indices_to_remove)] = False
                a_points_active = a_points_active[mask]


            # Remove used pending p points.
            if len(pending_indices_to_remove) > 0:
                mask_pending = np.ones(len(pending_p_points), dtype=bool)
                mask_pending[list(pending_indices_to_remove)] = False
                pending_p_points = pending_p_points[mask_pending]

            # (Remaining a_points_active will try again if there are still pending p points.)

        # *** Merge leftover pending points back into the main pool ***
        # Any pending p_points that were not used are returned to p_points.
        p_points = np.concatenate((p_points, pending_p_points))
        print(f"After merging, p_points has {len(p_points)} points.")

        # End of connection attempts for this cycle.
        if next_active_a_points_list:
            a_points_active = generate_new_a_points(next_active_a_points_list)
        else:
            a_points_active = np.empty(0, dtype=a_points_active.dtype)
        if cycle_num == 1 and simulate:
            true_barcodes = "true_barcodes.csv"
            animate_simulation(s_radius, density, true_barcodes, aoe_radius, success_prob)

    # conjugation of files from each cycle
    all_rows = conjugate_files(folder_path)

    # Write the combined rows into the final 'cleared_sequences.csv' file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['sequence', 'N0'])

        # Write the combined data
        writer.writerows(all_rows)

    # Delete the folder after the files are combined and deleted
    shutil.rmtree(folder_path)

    # End simulation cycles.
    print("Simulation ended.")

    # Create a dictionary to collapse duplicates.
    dedup = {}

    with open("cleared_sequences.csv", "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip rows that are duplicate headers.
            if row["sequence"] == "sequence":
                continue
            encoded_seq = row["sequence"]
            try:
                n0 = int(row["N0"])
            except ValueError:
                continue  # skip rows with invalid N0 values
            # If the encoded sequence is already seen, add the count; otherwise, create new entry.
            if encoded_seq in dedup:
                dedup[encoded_seq] += n0
            else:
                dedup[encoded_seq] = n0

    # Build the final list of deduplicated and decoded rows.
    sequences_polony_amp = []
    for encoded_seq, total_n0 in dedup.items():
        decoded_seq = decode(encoded_seq)
        sequences_polony_amp.append({"sequence": decoded_seq, "N0": total_n0})

    bridge_output = output

    return sequences_polony_amp, bridge_output
