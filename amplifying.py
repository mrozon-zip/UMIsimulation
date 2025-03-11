import logging
from support import compute_global_p, process_mutation
import os
import concurrent.futures
import cupy as cp
import math, random
from typing import List, Dict, Tuple, Any

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


# -----------------------------------------------------------------------------
# Explanation:
#
# This implementation uses a NumPy 3D array to represent the spatial grid.
#
# 1. The grid’s first two dimensions represent a plane of size
#    (2 * S_radius * res) x (2 * S_radius * res), where the resolution factor
#    res = √1000 ≈ 31.62. For example:
#       - If S_radius = 5, grid dimensions ≈ round(2*5*31.62) = 316 x 316,
#         and the number of available spots (cells inside the working circle)
#         is approximately π*(5*31.62)² ≈ 78,500.
#       - If S_radius = 10, available spots ≈ π*(10*31.62)² ≈ 314,000.
#
# 2. The working area is defined as the circle centered at (S_radius, S_radius)
#    in physical units, which translates to a center at (S_radius*res, S_radius*res)
#    in grid coordinates. A cell (i, j) is available if its Euclidean distance from
#    the center is less than S_radius * res.
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
#       - The effective AOE (in grid cells) is computed as (aoe_radius/100) * S_radius * res.
#       - For each candidate cell found within the AOE (using Euclidean distance),
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


def encode_sequence(seq: str, max_len: int) -> cp.ndarray:
    """
    Convert a string DNA sequence to a CuPy array of length `max_len`,
    integer-encoded. If seq is shorter than max_len, pad with "A" (0).
    """
    arr = [base_to_int[ch] for ch in seq]
    # Pad if needed
    arr += [0] * (max_len - len(arr))
    return cp.array(arr, dtype=cp.int8)


def decode_sequence(seq_array: cp.ndarray) -> str:
    """
    Convert a CuPy array of integers back to a DNA string (no trimming).
    """
    return "".join(int_to_base[int(x)] for x in seq_array)


def batch_mutate_sequences(
        parent_ids: cp.ndarray,
        all_sequences: cp.ndarray,
        mutation_rate: float,
        mutation_probabilities: Dict[str, float]
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Vectorized mutation of many sequences at once on the GPU.
    - parent_ids: shape (N,) array of indices into `all_sequences`.
    - all_sequences: shape (num_sequences, seq_len).
    - mutation_rate: probability per base to mutate.
    - mutation_probabilities: e.g. {"substitution": 0.4, "deletion": 0.3, "insertion": 0.3}
      (Here we only implement substitution for simplicity).

    Returns:
      updated_all_sequences: possibly expanded version of all_sequences with new child sequences appended
      new_seq_ids: shape (N,) array of new sequence IDs in updated_all_sequences
    """
    # For simplicity, we'll do only random substitutions with probability = mutation_rate.
    parent_batch = all_sequences[parent_ids]  # shape (N, seq_len)

    # Create a random uniform matrix for deciding which bases to mutate.
    mut_mask = cp.random.rand(*parent_batch.shape) < mutation_rate

    # For mutated positions, choose a random base from [0..3].
    random_bases = cp.random.randint(0, 4, size=parent_batch.shape, dtype=cp.int8)

    # child_batch picks random_bases where mut_mask is True, else original parent base
    child_batch = cp.where(mut_mask, random_bases, parent_batch)

    # Append child_batch to the end of all_sequences
    new_seq_start = all_sequences.shape[0]
    updated_all_sequences = cp.concatenate([all_sequences, child_batch], axis=0)
    new_seq_ids = cp.arange(new_seq_start, new_seq_start + parent_batch.shape[0], dtype=cp.int32)

    return updated_all_sequences, new_seq_ids


def polonies_amplification(
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
    Demonstration of a GPU-accelerated polonies amplification using CuPy.
    Encodes sequences as integer arrays, uses vectorized neighbor search,
    and batch mutation on the GPU. No 'nonlocal' usage for clarity.
    """

    # 1) Convert input list of dicts into a plain list of sequences
    input_seqs = [seq_dict['sequence'] for seq_dict in sequences]
    max_len = max(len(s) for s in input_seqs)

    # 2) Create a big "all_sequences" array storing each input sequence as a row
    encoded_list = [encode_sequence(seq, max_len) for seq in input_seqs]
    all_sequences = cp.stack(encoded_list, axis=0)  # shape (num_input, max_len)

    # 3) We'll store (row index) for each input sequence
    barcode_ids = cp.arange(len(input_seqs), dtype=cp.int32)

    # 4) Define resolution and grid size
    res = math.sqrt(1000)
    grid_size = int(round(2 * s_radius * res))
    center = s_radius * res

    # 5) Create two 2D arrays on GPU
    #   - point_type: (grid_size, grid_size), int8
    #   - seq_id:     (grid_size, grid_size), int32
    point_type = cp.full((grid_size, grid_size), TYPE_UNAVAILABLE, dtype=cp.int8)
    seq_id = cp.full((grid_size, grid_size), -1, dtype=cp.int32)

    # Fill available cells with B or D according to density
    coords_i = cp.arange(grid_size)
    coords_j = cp.arange(grid_size)
    i_grid, j_grid = cp.meshgrid(coords_i, coords_j, indexing='ij')  # shape (grid_size, grid_size)

    # Distance from center
    dist_to_center = cp.sqrt((i_grid - center) ** 2 + (j_grid - center) ** 2)
    within_circle = dist_to_center < (s_radius * res)

    fill_mask = (cp.random.rand(grid_size, grid_size) < (density / 100))
    combined_mask = within_circle & fill_mask

    # Randomly choose B or D for those cells
    bd_rand = cp.random.rand(grid_size, grid_size)
    type_b_or_d = cp.where(bd_rand < 0.5, TYPE_B, TYPE_D)
    point_type = cp.where(combined_mask, type_b_or_d, point_type)

    # 6) Place initial A/C points by replacing some of B/D cells
    # Find coords of B or D
    b_or_d_mask = (point_type == TYPE_B) | (point_type == TYPE_D)
    valid_coords = cp.argwhere(b_or_d_mask)  # shape (#cells, 2)
    if valid_coords.shape[0] == 0:
        # Edge case: no cells at all
        return [], [], point_type, output

    n_barcodes = len(input_seqs)
    chosen_indices = cp.random.choice(
        valid_coords.shape[0],
        size=n_barcodes,
        replace=(valid_coords.shape[0] < n_barcodes)
    )
    chosen_positions = valid_coords[chosen_indices]  # shape (n_barcodes, 2)

    chosen_i = chosen_positions[:, 0]
    chosen_j = chosen_positions[:, 1]
    old_types = point_type[chosen_i, chosen_j]
    new_types = cp.where(old_types == TYPE_B, TYPE_A, TYPE_C)
    point_type[chosen_i, chosen_j] = new_types
    seq_id[chosen_i, chosen_j] = barcode_ids

    # Active points (i,j,seq_id)
    active_i = chosen_i
    active_j = chosen_j
    active_seq_ids = barcode_ids

    # 7) Main simulation
    effective_aoe = (aoe_radius / 100) * s_radius * res
    cycle_counts = []
    max_cycles = 1000

    for cycle in range(max_cycles):
        if active_i.size == 0:
            break

        parent_ptypes = point_type[active_i, active_j]
        # Partition parents into two groups: A or C
        is_a_mask = (parent_ptypes == TYPE_A)
        is_c_mask = (parent_ptypes == TYPE_C)

        # We'll define a helper function that performs vectorized conversion attempts
        def vector_convert_parents_to_targets(
                parent_i: cp.ndarray,
                parent_j: cp.ndarray,
                parent_seq_ids: cp.ndarray,
                parent_type_val: int,
                target_type_val: int,
                all_sequences: cp.ndarray
        ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
            """
            For each parent in (parent_i, parent_j), attempt to convert exactly one
            target cell of 'target_type_val' within the AOE. Returns:
              (child_i, child_j, child_seq_ids, updated_all_sequences)
            """

            if parent_i.size == 0:
                return (cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        all_sequences)

            # gather target coords
            target_coords = cp.argwhere(point_type == target_type_val)
            if target_coords.size == 0:
                return (cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        all_sequences)

            ti = target_coords[:, 0]  # shape (Y,)
            tj = target_coords[:, 1]
            X = parent_i.size
            Y = target_coords.shape[0]

            # Compute pairwise distance
            pi_col = parent_i[:, None]  # shape (X,1)
            pj_col = parent_j[:, None]
            di = pi_col - ti[None, :]  # shape (X,Y)
            dj = pj_col - tj[None, :]
            dist_matrix = cp.sqrt(di * di + dj * dj)

            in_range = dist_matrix <= effective_aoe
            any_valid = cp.any(in_range, axis=1)  # shape (X,)
            valid_targets_idx = cp.argmax(in_range, axis=1)  # picks first True for each row
            valid_targets_idx = cp.where(any_valid, valid_targets_idx, -1)

            parent_indices = cp.argwhere(valid_targets_idx != -1).ravel()
            if parent_indices.size == 0:
                return (cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        cp.array([], dtype=cp.int32),
                        all_sequences)

            chosen_targets = valid_targets_idx[parent_indices]
            child_i = ti[chosen_targets]
            child_j = tj[chosen_targets]
            child_parents_seq = parent_seq_ids[parent_indices]

            # success probability with deviation
            random_dev = cp.random.uniform(-deviation, deviation, size=child_i.shape)
            eff_success_probs = cp.clip(success_prob * (1 + random_dev), 0, 1)
            success_draws = cp.random.rand(child_i.size)
            success_mask = success_draws < eff_success_probs

            child_i = child_i[success_mask]
            child_j = child_j[success_mask]
            child_parents_seq = child_parents_seq[success_mask]

            if child_i.size == 0:
                return (child_i, child_j, child_parents_seq, all_sequences)

            # batch mutation
            updated_all_sequences, new_child_seq_ids = batch_mutate_sequences(
                child_parents_seq,
                all_sequences,
                mutation_rate,
                mutation_probabilities
            )

            # Mark them in the grid
            point_type[child_i, child_j] = parent_type_val
            seq_id[child_i, child_j] = new_child_seq_ids

            return (child_i, child_j, new_child_seq_ids, updated_all_sequences)

        # Convert A -> B
        a_i = active_i[is_a_mask]
        a_j = active_j[is_a_mask]
        a_seq = active_seq_ids[is_a_mask]
        child_a_i, child_a_j, child_a_seq_ids, all_sequences = vector_convert_parents_to_targets(
            a_i, a_j, a_seq,
            TYPE_A,  # parent's type
            TYPE_B,  # target type to convert
            all_sequences
        )

        # Convert C -> D
        c_i = active_i[is_c_mask]
        c_j = active_j[is_c_mask]
        c_seq = active_seq_ids[is_c_mask]
        child_c_i, child_c_j, child_c_seq_ids, all_sequences = vector_convert_parents_to_targets(
            c_i, c_j, c_seq,
            TYPE_C,
            TYPE_D,
            all_sequences
        )

        # Combine new children
        new_i = cp.concatenate([child_a_i, child_c_i])
        new_j = cp.concatenate([child_a_j, child_c_j])
        new_seq = cp.concatenate([child_a_seq_ids, child_c_seq_ids])

        total_active = active_i.size + new_i.size
        cycle_counts.append(int(total_active))

        if new_i.size == 0:
            break

        # Update active
        active_i = cp.concatenate([active_i, new_i])
        active_j = cp.concatenate([active_j, new_j])
        active_seq_ids = cp.concatenate([active_seq_ids, new_seq])

    # 8) Merge duplicates from A/C cells
    is_ac_mask = (point_type == TYPE_A) | (point_type == TYPE_C)
    ac_coords = cp.argwhere(is_ac_mask)
    ac_seq_ids = seq_id[ac_coords[:, 0], ac_coords[:, 1]]
    if ac_seq_ids.size == 0:
        return [], [0], point_type, output

    uniq_ids, counts = cp.unique(ac_seq_ids, return_counts=True)
    uniq_ids_host = uniq_ids.get()
    counts_host = counts.get()

    merged_list = []
    for uid, cnt in zip(uniq_ids_host, counts_host):
        if uid < 0 or uid >= all_sequences.shape[0]:
            continue
        seq_array = all_sequences[uid]  # shape (max_len,)
        # For demonstration, we won't handle trailing padding
        seq_str = decode_sequence(seq_array)
        merged_list.append({"sequence": seq_str, "N0": int(cnt)})

    polonies_output = f"results/polonies_{output}"
    cycle_counts_host = [int(x) for x in cycle_counts]

    return merged_list, cycle_counts_host, point_type, polonies_output
