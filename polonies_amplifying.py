from support import compute_global_p, process_mutation, decode, encode
import math
import random
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.spatial import cKDTree
import shutil
import glob
import csv
import os

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def generate_points(s_radius: float, density: float) -> np.ndarray:
    """
    Generate p_points as a numpy array of (x, y) coordinates uniformly distributed within
    a circle of radius s_radius. The total number is given by density * (π * s_radius^2).
    """
    total_points = int(density * math.pi * (s_radius ** 2))
    r = s_radius * np.sqrt(np.random.rand(total_points))
    theta = 2 * np.pi * np.random.rand(total_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dtype = np.dtype([('x', np.float64),
                      ('y', np.float64),
                      ('id', 'O'),
                      ('born', 'O'),
                      ('active', 'O')])
    points = np.empty(total_points, dtype=dtype)
    points['x'] = x
    points['y'] = y
    return points


def generate_a_points(sequences: list, s_radius: float) -> np.ndarray:
    """
    Generate a numpy structured array for active a points.
    Each entry gets a random coordinate within the circle of radius s_radius and
    carries the sequence and its N0 value.
    """
    dtype = np.dtype([('x', np.float64),
                      ('y', np.float64),
                      ('sequence', 'O'),
                      ('id', 'O'),
                      ('parent_id', 'O'),
                      ('born', 'O'),
                      ('active', 'O')
                      ])
    n = len(sequences)
    a_points = np.empty(n, dtype=dtype)

    # Generate random coordinates in circle
    r = s_radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    a_points['x'] = r * np.cos(theta)
    a_points['y'] = r * np.sin(theta)

    # Assign sequence and a unique id
    for i, seq_dict in enumerate(sequences):
        a_points['sequence'][i] = encode(seq_dict["sequence"])
        a_points['born'][i] = 1
        a_points['active'][i] = 1
    return a_points


def generate_new_a_points(new_a_points_list: list) -> np.ndarray:
    """
    Generate a numpy structured array for active a points.
    Each entry gets a random coordinate within the circle of radius s_radius and
    carries the sequence and its N0 value.
    """
    dtype = np.dtype([('x', np.float64),
                      ('y', np.float64),
                      ('sequence', 'O'),
                      ('id', 'O'),
                      ('parent_id','O'),
                      ('born', 'O'),
                      ('active', 'O')])
    n = len(new_a_points_list)
    new_a_points_array = np.empty(n, dtype=dtype)

    for i, seq_dict in enumerate(new_a_points_list):
        new_a_points_array['sequence'][i] = seq_dict["sequence"]
        new_a_points_array['x'][i] = seq_dict['x']
        new_a_points_array['y'][i] = seq_dict['y']
        new_a_points_array['id'][i] = seq_dict["id"]
        new_a_points_array['born'][i] = seq_dict["born"]
        new_a_points_array['active'][i] = seq_dict["active"]
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
        writer.writerow(['sequence', 'N0', 'id', 'parent_id', 'x', 'y', 'born', 'active'])

        # Write the data (sequence, N0, id) from list of dictionaries
        for row in cleared_sequences:
            writer.writerow([row['sequence'],
                             row['N0'],
                             row['id'],
                             row['parent_id'],
                             row['x'],
                             row['y'],
                             row['born'],
                             row['active']])


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
    bd_points = generate_points(s_radius, density)  # np.array
    ac_points = generate_a_points(sequences, s_radius)  # np.array
    inactive_updates = {}

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
    a_copy = a_points_active
    c_copy = c_points_active

    b_points = bd_points[indices2[:mid2]]
    d_points = bd_points[indices2[mid2:]]

    # Set id and parent_id fields for a_points_active and c_points_active.
    for i in range(len(a_points_active)):
        a_points_active['id'][i] = available_ids.pop()

    for i in range(len(c_points_active)):
        c_points_active['id'][i] = available_ids.pop()

    for i in range(len(b_points)):
        b_points['id'][i] = available_ids.pop()

    for i in range(len(d_points)):
        d_points['id'][i] = available_ids.pop()

    base_folder = "/Users/krzysztofmrozik/Desktop/SciLifeLab/Projects/PCR simulation/"
    # Dictionary to collect sequences from a points that are removed (cleared from memory).
      # key: sequence, value: cumulative N0
    folder_name = f"{base_folder}results_amplified/helping_folder"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Define the folder where the supporting files will be saved
    base, ext = os.path.splitext(output)
    helping_files = f"{base_folder}results_amplified/helping_folder/c_supporting_{base}"
    orphan_file = f"{base_folder}results_amplified/helping_folder/orphan.csv"
    aoe_radius = s_radius * aoe_radius/100
    # Define the output file for concatenated results
    output_file_path = f"{base_folder}results_amplified/helping_folder/cleared_sequences_{base}.csv"

    cycle_num = 0
    while True:
        cycle_num += 1
        print(f"Cycle {cycle_num}: {len(a_points_active)} active a points, {len(b_points)} remaining b points")
        print(f"Cycle {cycle_num}: {len(c_points_active)} active a points, {len(d_points)} remaining d points")

        # Two fail-safe if statements:
        # Termination: if no active a points remain, check if any b points or d points exist and then break.
        if len(a_points_active) == 0 and len(c_points_active) == 0:
            if len(b_points) > 0 or len(d_points) > 0:
                orphans = np.concatenate((b_points, d_points))
            break

        # --- Determine pending p points ---
        # Build a KDTree on p_points.
        b_tree = cKDTree(np.column_stack((b_points['x'], b_points['y'])))
        d_tree = cKDTree(np.column_stack((d_points['x'], d_points['y'])))
        # Get coordinates of active a points.
        coords_a = np.column_stack((a_points_active['x'], a_points_active['y'])) # A points
        coords_c = np.column_stack((c_points_active['x'], c_points_active['y']))
        # For each active a point, get indices of p_points within its aoe.
        query_results_b = b_tree.query_ball_point(coords_a, r=aoe_radius) # B points within AOE
        query_results_d = d_tree.query_ball_point(coords_c, r=aoe_radius)

        # Identify a_points that do NOT have any p point within their aoe.
        no_pending_mask_b = np.array([len(indices) == 0 for indices in query_results_b]) # A points that don't have b point in AOE
        no_pending_mask_d = np.array([len(indices) == 0 for indices in query_results_d])

        if np.any(no_pending_mask_b): # What to do with A points with no b points within AOE
            inactive_a = a_points_active[no_pending_mask_b]
            for pt in inactive_a:
                inactive_updates[pt['id']] = pt['active']
            a_points_active = a_points_active[~no_pending_mask_b] # A points that have b points in AOE are the active pool
            coords_a = np.column_stack((a_points_active['x'], a_points_active['y']))
            query_results_b = b_tree.query_ball_point(coords_a, r=aoe_radius) # So after removing inactive A points, we remake active A points array


        if np.any(no_pending_mask_d): # the same as above but for c points.
            inactive_c = c_points_active[no_pending_mask_d]
            for pt in inactive_c:
                inactive_updates[pt['id']] = pt['active']
            c_points_active = c_points_active[~no_pending_mask_d]
            coords_c = np.column_stack((c_points_active['x'], c_points_active['y']))
            query_results_d = d_tree.query_ball_point(coords_c, r=aoe_radius)


        # Increment 'active' attribute only if pending points are found in the AOE
        for idx, indices in enumerate(query_results_b):
            if len(indices) > 0:  # pending b point exists in AOE
                a_points_active['active'][idx] += 1

        for idx, indices in enumerate(query_results_d):
            if len(indices) > 0:  # pending d point exists in the AOE
                c_points_active['active'][idx] += 1

        # Create a boolean mask for b_points that are within any active a point's aoe.
        pending_mask_b = np.zeros(len(b_points), dtype=bool)
        for indices in query_results_b:
            if indices:
                pending_mask_b[indices] = True # Basically all b points that are within a points AOE are marked as available
        # Extract pending points.
        pending_b_points = b_points[pending_mask_b] # This variable marks which b points we take into consideration
        # These that are not within AOE are deleted
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
        new_points = []
        if cycle_num == 1:
            for point in a_points_active:
                new_points.append({
                    "sequence": point['sequence'],
                    "N0": 1,
                    "id": point['id'],
                    "parent_id": point["parent_id"],
                    'x': point["x"],
                    "y": point["y"],
                    "born": point["born"],
                    "active": point["active"]
                })
            for point in c_points_active:
                new_points.append({
                    "sequence": point['sequence'],
                    "N0": 1,
                    "id": point['id'],
                    "parent_id": point["parent_id"],
                    "x": point["x"],
                    "y": point["y"],
                    "born": point["born"],
                    "active": point["active"],
                })
        while (pending_b_points.size > 0 and
               a_points_active.size > 0 and
               pending_d_points.size > 0 and
               c_points_active.size > 0):            # Build a KDTree for the current pending p points.
            pending_tree_b = cKDTree(np.column_stack((pending_b_points['x'], pending_b_points['y'])))
            pending_tree_d = cKDTree(np.column_stack((pending_d_points['x'], pending_d_points['y'])))

            # Record connection attempts:
            connection_attempts_a = {}
            for i, point in enumerate(a_points_active):
                pt = [point['x'], point['y']]
                candidate_indices = pending_tree_b.query_ball_point(pt, r=aoe_radius)
                if candidate_indices:
                    chosen = random.choice(candidate_indices)
                    connection_attempts_a.setdefault(chosen, []).append(i)

            # Record connection attempts:
            connection_attempts_c = {}
            for i, point in enumerate(c_points_active):
                pt = [point['x'], point['y']]
                candidate_indices = pending_tree_d.query_ball_point(pt, r=aoe_radius)
                if candidate_indices:
                    chosen = random.choice(candidate_indices)
                    connection_attempts_c.setdefault(chosen, []).append(i)
            # If no a point found any pending p point, break out.
            if not connection_attempts_c and not connection_attempts_a:
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
                chosen_c_index = random.choice(c_indices)
                if random.random() < success_prob:
                    successful_connections_cd.append((chosen_c_index, pending_idx))

            if not successful_connections_cd and not successful_connections_ab:
                break

            indices_to_remove_a = set()
            pending_indices_to_remove_b = set()
            for a_idx, pending_idx in successful_connections_ab:
                a_point = a_points_active[a_idx]
                b_point = pending_b_points[pending_idx]
                new_seq, mutated = process_mutation(a_point['sequence'], mutation_rate, mutation_probabilities)
                # Create a new point with the p_point's coordinates and assign its id.
                new_point = {"x": b_point['x'],
                             "y": b_point['y'],
                             "sequence": new_seq,
                             "id": b_point['id'],
                             "parent_id": a_point['id'],
                             "born": cycle_num,
                             "active": 0
                             }
                new_points.append({
                    "sequence": new_point['sequence'],
                    "N0": 1,
                    "id": new_point['id'],
                    "parent_id": new_point['parent_id'],
                    "x": new_point["x"],
                    "y": new_point["y"],
                    "born": new_point["born"],
                    "active": new_point["active"]
                })
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
                # Create a new point with the p_point's coordinates and assign its id.
                new_point = {"x": d_point['x'],
                             "y": d_point['y'],
                             "sequence": new_seq,
                             "id": d_point['id'],
                             "parent_id": c_point["id"],
                             "born": cycle_num,
                             "active": 0
                             }
                new_points.append({
                    "sequence": new_point['sequence'],
                    "N0": 1,
                    "id": new_point['id'],
                    "parent_id": new_point['parent_id'],
                    "x": new_point["x"],
                    "y": new_point["y"],
                    "born": new_point["born"],
                    "active": new_point["active"]
                })
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

        save_a_points_to_file(helping_files, cycle_num, new_points)
            # (Remaining a_points_active will try again if there are still pending p points.)

        # *** Merge leftover pending points back into the main pool ***
        # Any pending p_points that were not used are returned to p_points.
        b_points = np.concatenate((b_points, pending_b_points))
        print(f"After merging, b_points has {len(b_points)} points.")

        d_points = np.concatenate((d_points, pending_d_points))
        print(f"After merging, d_points has {len(d_points)} points.")

        # End of connection attempts for this cycle.
        if next_active_a_points_list:
            a_points_active = generate_new_a_points(next_active_a_points_list)
        else:
            a_points_active = np.empty(0, dtype=a_points_active.dtype)

        # End of connection attempts for this cycle.
        if next_active_c_points_list:
            c_points_active = generate_new_a_points(next_active_c_points_list)
        else:
            c_points_active = np.empty(0, dtype=c_points_active.dtype)

    # Update CSV files in helping_files folder with inactive_updates values
    for csv_file in glob.glob(os.path.join(helping_files, '*.csv')):
        updated_rows = []
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    point_id = int(row['id'])
                except ValueError:
                    point_id = row['id']
                if point_id in inactive_updates:
                    row['active'] = str(inactive_updates[point_id])
                updated_rows.append(row)
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['sequence', 'N0', 'id', 'parent_id', 'x', 'y', 'born', 'active']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in updated_rows:
                writer.writerow(row)
    all_rows = conjugate_files(helping_files)

    # Write the combined rows into the final 'cleared_sequences.csv' file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['sequence', 'N0', 'id', 'parent_id', "x", "y", 'born', 'active'])
        # Write the data from the combined rows list
        for row in all_rows:
            writer.writerow(row)

    #####
    # Read the existing cleared_sequences CSV file into a list of dictionaries
    cleared_points = []
    with open(output_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert x and y to float; skip rows that cannot be converted
            try:
                row["x"] = float(row["x"])
                row["y"] = float(row["y"])
            except ValueError:
                continue
            cleared_points.append(row)
    new_points_orphans = []
    # For every entry in orphans (if any), select a random point from cleared_points within its area of effect (aoe_radius)
    if 'orphans' in locals() and orphans.size > 0:
        for orphan in orphans:
            ox, oy = orphan['x'], orphan['y']
            # Find candidates within the orphan's AOE
            candidates = []
            for point in cleared_points:
                dx = ox - point["x"]
                dy = oy - point["y"]
                if (dx * dx + dy * dy) ** 0.5 <= aoe_radius:
                    candidates.append(point)
            if candidates:
                selected_point = random.choice(candidates)
                new_seq, mutated = process_mutation(int(selected_point['sequence']), mutation_rate,
                                                    mutation_probabilities)
                new_point = {"x": orphan['x'],
                             "y": orphan['y'],
                             "sequence": new_seq,
                             "id": orphan['id'],
                             "parent_id": selected_point["id"],
                             "born": cycle_num+1,
                             "active": 0}
                new_points_orphans.append({
                    "sequence": new_point['sequence'],
                    "N0": 1,
                    "id": new_point["id"],
                    "parent_id": new_point["parent_id"],
                    "x": new_point["x"],
                    "y": new_point["y"],
                    "born": new_point["active"],
                    "active": new_point["active"]
                })
    # Append the orphan rows to the existing data
    cleared_points.extend(new_points_orphans)

    # Here
    with open(output_file_path, mode='w', newline='') as file:
        fieldnames = ['sequence', 'N0', 'id', 'parent_id', 'x', 'y', 'born', 'active']
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        for row in cleared_points:
            writer.writerow(row)
    #####

    # Delete the folder after the files are combined and deleted
    shutil.rmtree(helping_files)

    print("Simulation ended.")

    dedup = {}

    with open(output_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["sequence"] == "sequence":
                continue
            encoded_seq = row["sequence"]
            try:
                n0 = int(row["N0"])
            except ValueError:
                continue
            id_val = int(row["id"])
            born = int(row["born"])
            active = int(row["active"])
            if encoded_seq in dedup:
                dedup[encoded_seq]["N0"] += n0
                dedup[encoded_seq]["id"].append(id_val)
                dedup[encoded_seq]["born"].append(int(born))
                dedup[encoded_seq]["active"].append(active)
            else:
                dedup[encoded_seq] = {"N0": n0,
                                      "id": [id_val],
                                      "parent_id": row["parent_id"],
                                      "x": row["x"],
                                      "y": row["y"],
                                      "born": [born],
                                      "active": [active]}

    # Build the final list of deduplicated and decoded rows.
    desired_length = len(sequences[0]['sequence'])
    decoded_sequences = {encoded_seq: decode(encoded_seq) for encoded_seq in dedup}
    sequences_polony_amp = []
    for encoded_seq, data in dedup.items():
        decoded_seq = decoded_sequences[encoded_seq]
        if len(decoded_seq) > desired_length:
            decoded_seq = decoded_seq[:desired_length]
        elif len(decoded_seq) < desired_length:
            decoded_seq = decoded_seq + ''.join(random.choices(NUCLEOTIDES, k=desired_length - len(decoded_seq)))
        sequences_polony_amp.append({
            "sequence": decoded_seq,
            "N0": data["N0"],
            "id": data["id"],
            "parent_id": data["parent_id"],
            "x": data["x"],
            "y": data["y"],
            "born": data["born"],
            "active": data["active"]
        })

    polonies_output = f"{base_folder}results_amplified/polonies_{base}{ext}"


    return sequences_polony_amp, polonies_output
