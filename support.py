import math
import cupy as cp
import random
from typing import Dict, Tuple, List

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


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


def batch_mutate_sequences(
        parent_ids: cp.ndarray,
        all_sequences: cp.ndarray,
        mutation_rate: float,
        mutation_probabilities: Dict[str, float],
        pad_value: int = -1
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Process mutation over a batch of sequences with possible substitutions, deletions, and insertions.
    Each nucleotide in each parent sequence is checked for mutation.

    If a mutation occurs at a nucleotide:
      - substitution: Replace the nucleotide with a different one.
      - deletion: Remove the nucleotide.
      - insertion: Insert a new nucleotide before the current one.

    Sequences are represented as fixed-length arrays (e.g., with nucleotides as integers 0–3),
    and the output mutated sequences are padded with pad_value (default -1) to form a 2D array.
    The effective (true) sequence length is tracked separately.

    Additionally, if the mutated sequences become longer than the original sequence length,
    the original all_sequences is padded to ensure both arrays have the same number of columns.

    Args:
      parent_ids: CuPy array of shape (N,) with indices into `all_sequences` (selecting the parent sequences).
      all_sequences: CuPy array of shape (num_sequences, seq_len) holding sequences.
      mutation_rate: Per-base probability to mutate.
      mutation_probabilities: Dictionary with weights for each mutation type, e.g.
                              {"substitution": 0.4, "deletion": 0.3, "insertion": 0.3}.
      pad_value: The integer used to pad sequences (default -1).

    Returns:
      updated_all_sequences: CuPy array with the new mutated sequences appended (each row padded to the new maximum length).
      new_seq_ids: CuPy array of new sequence IDs (indices in updated_all_sequences).
      effective_lengths: CuPy array (shape (N,)) holding the effective (true) length of each mutated sequence.
    """
    # Extract the parent sequences (shape: N x original_length)
    parent_batch = all_sequences[parent_ids]
    N, original_length = parent_batch.shape

    mutated_sequences: List[List[int]] = []
    effective_lengths: List[int] = []

    # Process each sequence individually for variable-length mutations.
    for idx in range(N):
        # Convert the CuPy array to a Python list.
        seq = parent_batch[idx].get().tolist()
        mutated = False
        new_seq: List[int] = []
        j = 0
        while j < len(seq):
            if random.random() < mutation_rate:
                mutated = True
                mutation_type = random.choices(
                    population=['substitution', 'deletion', 'insertion'],
                    weights=[
                        mutation_probabilities.get('substitution', 0),
                        mutation_probabilities.get('deletion', 0),
                        mutation_probabilities.get('insertion', 0)
                    ],
                    k=1
                )[0]
                if mutation_type == 'substitution':
                    current_nuc = seq[j]
                    options = [n for n in [0, 1, 2, 3] if n != current_nuc]
                    new_seq.append(random.choice(options))
                    j += 1
                elif mutation_type == 'deletion':
                    j += 1  # Skip this nucleotide.
                elif mutation_type == 'insertion':
                    inserted = random.choice([0, 1, 2, 3])
                    new_seq.append(inserted)
                    new_seq.append(seq[j])
                    j += 1
            else:
                new_seq.append(seq[j])
                j += 1

        # If no mutation occurred, keep the original sequence.
        if not mutated:
            new_seq = seq

        mutated_sequences.append(new_seq)
        effective_lengths.append(len(new_seq))

    # Determine the maximum effective length among mutated sequences.
    max_mutated_length = max(effective_lengths)

    # New maximum length is the larger of the original sequence length or the maximum mutated length.
    new_max_length = max(original_length, max_mutated_length)

    # Pad the original all_sequences if needed.
    if new_max_length > original_length:
        pad_width = ((0, 0), (0, new_max_length - original_length))
        all_sequences = cp.pad(all_sequences, pad_width, mode='constant', constant_values=pad_value)

    # Pad each mutated sequence to the new maximum length.
    padded_mutated_sequences = [
        seq + [pad_value] * (new_max_length - len(seq)) for seq in mutated_sequences
    ]
    mutated_array = cp.array(padded_mutated_sequences, dtype=all_sequences.dtype)

    # Append the mutated sequences to the original all_sequences.
    new_seq_start = all_sequences.shape[0]
    updated_all_sequences = cp.concatenate([all_sequences, mutated_array], axis=0)
    new_seq_ids = cp.arange(new_seq_start, new_seq_start + len(mutated_sequences), dtype=cp.int32)
    effective_lengths_cp = cp.array(effective_lengths, dtype=cp.int32)

    return updated_all_sequences, new_seq_ids