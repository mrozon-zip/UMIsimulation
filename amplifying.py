import logging
from support import compute_global_p, process_mutation, batch_mutate_sequences, decode, encode
import os
import cupy as cp
import math, random
from typing import List, Dict, Tuple, Any
import numpy as np
import csv
from scipy.spatial import cKDTree
from support import process_mutation
import shutil

# Example DNA base mapping and inverse mapping:
base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# For "point_type" in the grid:
TYPE_B = 0
TYPE_D = 1
TYPE_A = 2
TYPE_C = 3
TYPE_UNAVAILABLE = -1

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def encode_sequence(seq: str, max_len: int) -> cp.ndarray:
    """
    Convert a string DNA sequence to a CuPy array of length `max_len`,
    integer-encoded. If seq is shorter than max_len, pad with "A" (0).
    """
    arr = [base_to_int[ch] for ch in seq]
    # Pad if needed
    arr += [0]*(max_len - len(arr))
    return cp.array(arr, dtype=cp.int8)

def decode_sequence(seq_array: cp.ndarray) -> str:
    """
    Convert a CuPy array of integers back to a DNA string (no trimming).
    """
    return "".join(int_to_base[int(x)] for x in seq_array)

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

########################################
# MAIN FUNCTION: barcodes_amplification
########################################
def bridge_amplification(
        sequences: List[Dict[str, Any]],
        s_radius: float,
        simulate: bool,
        aoe_radius: float,
        density: float,
        success_prob: float,
        deviation: float,
        mutation_rate: float,
        mutation_probabilities: Dict[str, float],
        output: str
) -> Tuple[List[Dict[str, Any]], List[int], cp.ndarray, str]:
    """
    Runs the polonies amplification simulation sequentially for each input sequence.
    For each input sequence, the entire simulation is run on a fresh grid:
      - A single point A is placed at the center (using that sequence).
      - The simulation converts neighboring TYPE_B cells into TYPE_A cells.

    Aggregation:
      - Merged output: A list of dictionaries for all simulations.
      - Cycle counts: Elementwise aggregated counts across simulation runs.
      - Final grid: Taken from the first simulation run.
      - Output string: Taken from the first simulation run.
    """
    aggregated_merged_list = []
    aggregated_cycle_counts = []
    final_grid = None
    final_output_string = ""

    # Loop over each input sequence and run the simulation sequentially.
    for idx, seq_dict in enumerate(sequences):
        seq = seq_dict['sequence']
        merged_list, cycle_counts, grid, out_str = simulate_single_sequence(
            seq,
            s_radius,
            aoe_radius,
            density,
            success_prob,
            deviation,
            mutation_rate,
            mutation_probabilities,
            output
        )
        aggregated_merged_list.extend(merged_list)
        # Aggregate cycle counts elementwise.
        for i, count in enumerate(cycle_counts):
            if i < len(aggregated_cycle_counts):
                aggregated_cycle_counts[i] += count
            else:
                aggregated_cycle_counts.append(count)
        # Save grid and output string from the first run only.
        if idx == 0:
            final_grid = grid
            final_output_string = out_str

    return aggregated_merged_list, aggregated_cycle_counts, final_grid, final_output_string


def simulate_single_sequence(
        seq: str,
        s_radius: float,
        aoe_radius: float,
        density: float,
        success_prob: float,
        deviation: float,
        mutation_rate: float,
        mutation_probabilities: Dict[str, float],
        output: str
) -> Tuple[List[Dict[str, Any]], List[int], cp.ndarray, str]:
    """
    Runs a single simulation for one sequence.
    The simulation:
      - Initializes a grid with available cells set to TYPE_B.
      - Places one point A at the center using the provided sequence.
      - Runs the amplification loop where each A cell converts a neighboring B cell into A.
    Returns:
      - A merged list (list of dicts with sequences and counts).
      - A list of cycle counts (active cell counts per cycle).
      - The final grid (point_type array).
      - An output string.
    """
    # Setup for a single sequence
    max_len = len(seq)
    encoded = encode_sequence(seq, max_len)
    # all_sequences starts with just one row (the encoded sequence)
    all_sequences = cp.expand_dims(encoded, axis=0)
    barcode_ids = cp.array([0], dtype=cp.int32)

    # Define resolution and grid size
    res = math.sqrt(1000)
    grid_size = int(round(2 * s_radius * res))
    center = s_radius * res

    # Create grid arrays on GPU: point_type and seq_id.
    point_type = cp.full((grid_size, grid_size), TYPE_UNAVAILABLE, dtype=cp.int8)
    seq_id = cp.full((grid_size, grid_size), -1, dtype=cp.int32)

    # Fill available cells with TYPE_B within a circular region.
    coords_i = cp.arange(grid_size)
    coords_j = cp.arange(grid_size)
    i_grid, j_grid = cp.meshgrid(coords_i, coords_j, indexing='ij')
    dist_to_center = cp.sqrt((i_grid - center) ** 2 + (j_grid - center) ** 2)
    within_circle = dist_to_center < (s_radius * res)
    fill_mask = (cp.random.rand(grid_size, grid_size) < (density / 100))
    combined_mask = within_circle & fill_mask
    point_type = cp.where(combined_mask, TYPE_B, point_type)

    # Place a single point A at the center with the provided sequence.
    i_center = int(round(center))
    j_center = int(round(center))
    point_type[i_center, j_center] = TYPE_A
    seq_id[i_center, j_center] = barcode_ids[0]
    active_i = cp.array([i_center], dtype=cp.int32)
    active_j = cp.array([j_center], dtype=cp.int32)
    active_seq_ids = cp.array([barcode_ids[0]], dtype=cp.int32)

    effective_aoe = (aoe_radius / 100) * s_radius * res
    cycle_counts = []
    max_cycles = 1000

    def vector_convert_parents_to_targets(
            parent_i: cp.ndarray,
            parent_j: cp.ndarray,
            parent_seq_ids: cp.ndarray,
            parent_type_val: int,
            target_type_val: int,
            all_sequences: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        For each parent cell in (parent_i, parent_j), attempts to convert one target cell of type `target_type_val`
        (e.g., TYPE_B) into a new cell of type `parent_type_val` (e.g., TYPE_A) within the effective area of effect.
        This implementation:
          1. Restricts the target search to a bounding box around the parents.
          2. Uses chunked computation to avoid large temporary arrays.
          3. Uses cp.float16 for reduced precision in distance calculations.
        Returns:
          (child_i, child_j, child_seq_ids, updated_all_sequences)
        """
        # Return empty arrays if there are no parents.
        if parent_i.size == 0:
            return (cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    all_sequences)

        # Gather target coordinates from the grid.
        target_coords = cp.argwhere(point_type == target_type_val)
        if target_coords.size == 0:
            return (cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    all_sequences)

        # Separate target coordinates.
        ti_all = target_coords[:, 0]
        tj_all = target_coords[:, 1]

        # 2. Restrict search region: compute a bounding box around the parent cells with margin = effective_aoe.
        parent_min_i = cp.min(parent_i)
        parent_max_i = cp.max(parent_i)
        parent_min_j = cp.min(parent_j)
        parent_max_j = cp.max(parent_j)
        margin = effective_aoe  # effective_aoe should be defined in your simulation.
        region_mask = (ti_all >= parent_min_i - margin) & (ti_all <= parent_max_i + margin) & \
                      (tj_all >= parent_min_j - margin) & (tj_all <= parent_max_j + margin)
        ti = ti_all[region_mask]
        tj = tj_all[region_mask]
        if ti.size == 0:
            return (cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    all_sequences)

        # 1. & 3. Chunk the distance computation using reduced precision.
        chunk_size = 1000  # Adjust this value according to available memory.
        num_targets = ti.shape[0]
        num_parents = parent_i.shape[0]

        # For each parent, we'll track the best candidate (target index) and its distance.
        candidate_indices = -cp.ones(num_parents, dtype=cp.int32)
        candidate_distances = cp.full((num_parents,), effective_aoe + 1, dtype=cp.float16)

        # Cast parent's coordinates to float16.
        parent_i_f16 = parent_i.astype(cp.float16)
        parent_j_f16 = parent_j.astype(cp.float16)

        for start in range(0, num_targets, chunk_size):
            end = min(start + chunk_size, num_targets)
            # Cast the chunk of target coordinates to float16.
            chunk_ti = ti[start:end].astype(cp.float16)
            chunk_tj = tj[start:end].astype(cp.float16)
            # Compute the distance matrix for the current chunk.
            # Shape: (num_parents, current_chunk_size)
            di_chunk = parent_i_f16[:, None] - chunk_ti[None, :]
            dj_chunk = parent_j_f16[:, None] - chunk_tj[None, :]
            dist_chunk = cp.sqrt(di_chunk * di_chunk + dj_chunk * dj_chunk)

            # For each parent, find the minimum distance in this chunk.
            min_dist_chunk = cp.min(dist_chunk, axis=1)
            argmin_chunk = cp.argmin(dist_chunk, axis=1)

            # Update the candidate if the found distance is lower than the current candidate.
            update_mask = min_dist_chunk < candidate_distances
            candidate_distances[update_mask] = min_dist_chunk[update_mask]
            candidate_indices[update_mask] = start + argmin_chunk[update_mask]

        # Determine which parents found a valid candidate within effective_aoe.
        valid_parents_mask = candidate_distances <= effective_aoe
        valid_parent_indices = cp.argwhere(valid_parents_mask).ravel()
        if valid_parent_indices.size == 0:
            return (cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    cp.array([], dtype=cp.int32),
                    all_sequences)

        # For these parents, retrieve the corresponding target coordinates.
        chosen_target_indices = candidate_indices[valid_parent_indices]
        child_i = ti[chosen_target_indices]
        child_j = tj[chosen_target_indices]
        child_parents_seq = parent_seq_ids[valid_parent_indices]

        # Apply success probability with deviation.
        random_dev = cp.random.uniform(-deviation, deviation, size=child_i.shape)
        eff_success_probs = cp.clip(success_prob * (1 + random_dev), 0, 1)
        success_draws = cp.random.rand(child_i.size)
        success_mask = success_draws < eff_success_probs
        child_i = child_i[success_mask]
        child_j = child_j[success_mask]
        child_parents_seq = child_parents_seq[success_mask]

        if child_i.size == 0:
            return (child_i, child_j, child_parents_seq, all_sequences)

        # Process mutations and update the sequence layer.
        updated_all_sequences, new_child_seq_ids = batch_mutate_sequences(
            child_parents_seq,
            all_sequences,
            mutation_rate,
            mutation_probabilities
        )

        # Mark the grid cells as converted.
        point_type[child_i, child_j] = parent_type_val
        seq_id[child_i, child_j] = new_child_seq_ids

        return (child_i, child_j, new_child_seq_ids, updated_all_sequences)

    # Main simulation loop: each cycle, convert neighboring TYPE_B cells to TYPE_A.
    for cycle in range(max_cycles):
        if active_i.size == 0:
            break

        child_a_i, child_a_j, child_a_seq_ids, all_sequences = vector_convert_parents_to_targets(
            active_i, active_j, active_seq_ids,
            TYPE_A,
            TYPE_B,
            all_sequences
        )


        total_active = active_i.size + child_a_i.size
        cycle_counts.append(int(total_active))
        if child_a_i.size == 0:
            break
        active_i = cp.concatenate([active_i, child_a_i])
        active_j = cp.concatenate([active_j, child_a_j])
        active_seq_ids = cp.concatenate([active_seq_ids, child_a_seq_ids])

    # Merge duplicates from TYPE_A cells.
    is_a_mask = (point_type == TYPE_A)
    a_coords = cp.argwhere(is_a_mask)
    a_seq_ids = seq_id[a_coords[:, 0], a_coords[:, 1]]
    if a_seq_ids.size == 0:
        return [], [0], point_type, output

    uniq_ids, counts = cp.unique(a_seq_ids, return_counts=True)
    uniq_ids_host = uniq_ids.get()
    counts_host = counts.get()

    merged_list = []
    for uid, cnt in zip(uniq_ids_host, counts_host):
        if uid < 0 or uid >= all_sequences.shape[0]:
            continue
        seq_array = all_sequences[uid]
        seq_str = decode_sequence(seq_array)
        merged_list.append({"sequence": seq_str, "N0": int(cnt)})

    polonies_output = f"results/polonies_{output}"
    return merged_list, cycle_counts, point_type, polonies_output


# -----------------------------------------------------------------------------
# Explanation:
#
# This implementation uses a NumPy 3D array to represent the spatial grid.
#
# 1. The grid’s first two dimensions represent a plane of size
#    (2 * s_radius * res) x (2 * s_radius * res), where the resolution factor
#    res = √1000 ≈ 31.62. For example:
#       - If s_radius = 5, grid dimensions ≈ round(2*5*31.62) = 316 x 316,
#         and the number of available spots (cells inside the working circle)
#         is approximately π*(5*31.62)² ≈ 78,500.
#       - If s_radius = 10, available spots ≈ π*(10*31.62)² ≈ 314,000.
#
# 2. The working area is defined as the circle centered at (s_radius, s_radius)
#    in physical units, which translates to a center at (s_radius*res, s_radius*res)
#    in grid coordinates. A cell (i, j) is available if its Euclidean distance from
#    the center is less than s_radius * res.
#
# 3. The grid has a third “dimension” of size 2:
#       - Field 0 stores the point type:
#             "B" or "D" for initial points (with an empty string in the sequence field),
#             "A" or "C" for converted points (with their sequence), or np.nan for unavailable cells.
#       - Field 1 stores the sequence data.
#
# 4. Available cells (those inside the working circle) are filled with B/D points
#    randomly, with the percentage of filled cells defined by the density parameter.
#    (Density must be ≤ 100; higher values trigger an error.)
#
# 5. Initial A/C points are placed by replacing some of the already assigned B/D points.
#    The number of initial A/C points is determined by the number of barcodes from the CSV.
#    For each barcode, a random available cell is chosen:
#       - If that cell originally held a "B", it is converted to an "A" point.
#       - If it held a "D", it is converted to a "C" point.
#
# 6. In each simulation cycle:
#       - Every active point (A or C) searches within its neighborhood for target cells:
#             A points only convert neighboring B cells,
#             C points only convert neighboring D cells.
#       - The effective aoe (in grid cells) is computed as (aoe_radius/100) * s_radius * res.
#       - For each candidate cell found within the aoe (using Euclidean distance),
#         a conversion attempt is made using an effective success probability (success_prob
#         adjusted by a random deviation ±deviation). On success, process_mutation is applied
#         to the parent's sequence, and the target cell is converted to the parent's type (A or C)
#         with the new mutated sequence.
#       - Converted cells become new active points, while the parent remains active.
#
# 7. The simulation continues until no active point can convert any additional neighbors.
#
# 8. Finally, the grid is scanned to merge duplicate sequences (from cells of type A/C),
#    summing their counts into an "N0" field. The function returns:
#         - A merged list of sequence dictionaries (with keys "sequence" and "N0"),
#         - A history of active point counts per cycle,
#         - The final grid,
#         - And an output file name string.
# -----------------------------------------------------------------------------


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
    return new_a_points_array

"""
(
        seq: str,
        s_radius: float,
        aoe_radius: float,
        density: float,
        success_prob: float,
        deviation: float,
        mutation_rate: float,
        mutation_probabilities: Dict[str, float],
        output: str
) -> Tuple[List[Dict[str, Any]], List[int], cp.ndarray, str]:
"""


def polonies_amplification(s_radius: float,
                            density: float,
                            sequences: list,
                            aoe_radius: float,
                            success_prob: float,
                            deviation: float,
                            simulate: bool,
                            mutation_rate: float,
                            mutation_probabilities: Dict[str,float],
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
      is accepted (collisions resolved randomly), processed via process_mutation, and the a point is
      moved to the next cycle. Active a points that at the beginning of a cycle have no p point
      in their aoe are removed (their sequence is saved for CSV output).
    - The simulation repeats cycles until no active a points remain.
    - Finally, a CSV file is written with header: sequence,N0, where duplicate sequences are
      collapsed with N0 counts summed.
    """
    # Generate initial points.
    p_points = generate_points(s_radius, density) # np.array
    a_points_active = generate_a_points(sequences, s_radius) # np.array

    # Dictionary to collect sequences from a points that are removed (cleared from memory).
    cleared_sequences = {}  # key: sequence, value: cumulative N0

    # Define the folder where the supporting files will be saved
    folder_path = 'supporting_folder'

    # Define the output file for concatenated results
    output_file_path = 'cleared_sequences.csv'

    cycle_num = 0
    while True:
        cycle_num += 1
        print(f"Cycle {cycle_num}: {len(a_points_active)} active a points, {len(p_points)} remaining p points")

        # Two fail safe if statements:
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


            # SWITCH IT TO SUPPORTING METHOD
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


