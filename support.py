import math
import random
from typing import Dict, Tuple, List, Union

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']

def encode(sequence: str) -> int:
    """
    Encodes a DNA sequence (composed of A, G, C, T) to an integer,
    using mapping: A->1, G->2, C->3, T->4.
    """
    mapping = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
    result = 0
    for char in sequence:
        result = result * 10 + mapping[char]
    return result


def decode(number: int) -> str:
    """
    Decodes an integer (encoded via the above encode function) back to a DNA sequence.
    """
    mapping = {'1': 'A', '2': 'G', '3': 'C', '4': 'T'}
    s = str(number)
    return ''.join(mapping[digit] for digit in s)

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


def process_mutation(sequence: Union[str, int], mutation_rate: float,
                     mutation_probabilities: Dict[str, float]) -> Tuple[Union[str, int], bool]:
    """
    Process mutation over a sequence replication event.
    Each nucleotide is checked with probability `mutation_rate` for mutation.
    If a mutation occurs, one of three types is chosen:
      - substitution: replace nucleotide with a different one.
      - deletion: remove the nucleotide.
      - insertion: insert a new nucleotide before the current one.
    Returns a tuple of (possibly mutated sequence, mutation_occurred flag).
    """
    if isinstance(sequence, int):
        decoded = decode(sequence)
        seq_list = list(decoded)
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
        int_sequence = encode(''.join(seq_list))
        return int_sequence, mutated
    else:
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