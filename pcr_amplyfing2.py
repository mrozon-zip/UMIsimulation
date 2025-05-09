import random
import csv
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SequenceRecord:
    seq: str
    p_amp: float


def generate_sequences(seq_length: int, count: int) -> List[str]:
    """Generate `count` random DNA sequences of length `seq_length`."""
    bases = ['A', 'C', 'G', 'T']
    return [''.join(random.choices(bases, k=seq_length)) for _ in range(count)]


def amplify_sequences(
        sequences: List[str],
        cycles: int,
        amp_range: Tuple[float, float],
        error_prob: float
) -> List[str]:
    """
    Amplify a list of sequences over a number of PCR cycles.

    Each sequence has an assigned amplification probability drawn uniformly
    from amp_range. In each cycle, sequences passing a random check are copied,
    introducing insertion/deletion/substitution errors per base at probability
    error_prob.
    """
    bases = ['A', 'C', 'G', 'T']
    records = [SequenceRecord(seq, random.uniform(*amp_range)) for seq in sequences]

    for _ in range(cycles):
        original = list(records)
        new_records: List[SequenceRecord] = []

        for record in original:
            if random.random() <= record.p_amp:
                seq_list = list(record.seq)
                i = 0
                while i < len(seq_list):
                    if random.random() <= error_prob:
                        error = random.choice(['substitution', 'deletion', 'insertion'])
                        if error == 'substitution':
                            orig = seq_list[i]
                            choices = [b for b in bases if b != orig]
                            seq_list[i] = random.choice(choices)
                            i += 1
                        elif error == 'deletion':
                            seq_list.pop(i)
                            # i stays the same
                        else:  # insertion
                            ins = random.choice(bases)
                            seq_list.insert(i + 1, ins)
                            i += 2
                    else:
                        i += 1

                new_seq = ''.join(seq_list)
                new_pamp = random.uniform(*amp_range)
                new_records.append(SequenceRecord(new_seq, new_pamp))

        records.extend(new_records)

    return [r.seq for r in records]


def trim_sequences(sequences: List[str], target_length: int) -> List[str]:
    """
    Trim or pad sequences to exactly target_length.
    Longer sequences are cut; shorter sequences get random bases appended.
    """
    bases = ['A', 'C', 'G', 'T']
    trimmed: List[str] = []
    for seq in sequences:
        if len(seq) > target_length:
            trimmed.append(seq[:target_length])
        elif len(seq) < target_length:
            padding = ''.join(random.choices(bases, k=target_length - len(seq)))
            trimmed.append(seq + padding)
        else:
            trimmed.append(seq)
    return trimmed


def introduce_sequencing_errors(sequences: List[str], seq_error_prob: float) -> List[str]:
    """
    Introduce random substitution errors across all sequences simulating
    sequencing noise.
    """
    bases = ['A', 'C', 'G', 'T']
    total_nucs = sum(len(seq) for seq in sequences)
    num_errors = int(total_nucs * seq_error_prob)

    seqs = [list(seq) for seq in sequences]
    errors_done = 0
    seen = set()
    while errors_done < num_errors:
        idx = random.randrange(len(seqs))
        pos = random.randrange(len(seqs[idx]))
        if (idx, pos) in seen:
            continue
        seen.add((idx, pos))
        orig = seqs[idx][pos]
        choices = [b for b in bases if b != orig]
        seqs[idx][pos] = random.choice(choices)
        errors_done += 1

    return [''.join(seq) for seq in seqs]


def subsample_sequences(sequences: List[str], depth: int) -> List[str]:
    """
    Randomly sample `depth` sequences without replacement.
    If depth >= number of sequences, returns all sequences.
    """
    if depth >= len(sequences):
        return sequences.copy()
    return random.sample(sequences, depth)


def write_csv(sequences: List[str], filename: str = 'results1/pcr_amplified.csv') -> None:
    """
    Write sequences to a CSV file with two columns: sequence, N0 (filled with 1).
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'N0'])
        for seq in sequences:
            writer.writerow([seq, 1])


def pcr_pipeline(
        seq_length: int,
        initial_count: int,
        cycles: int,
        amp_range: Tuple[float, float],
        error_prob: float,
        seq_error_prob: float,
        depth: int,
        seed: int = None
) -> None:
    """
    Full PCR simulation pipeline:
      1. Generate initial sequences
      2. Amplify over cycles with PCR errors
      3. Trim/pad to original length
      4. Introduce sequencing errors
      5. Subsample to desired depth
      6. Write output CSV
    """
    if seed is not None:
        random.seed(seed)

    seqs = generate_sequences(seq_length, initial_count)
    write_csv(seqs, filename='dump/results1/true_barcodes_pcr.csv')
    amp_seqs = amplify_sequences(seqs, cycles, amp_range, error_prob)
    trimmed = trim_sequences(amp_seqs, seq_length)
    errored = introduce_sequencing_errors(trimmed, seq_error_prob)
    sampled = subsample_sequences(errored, depth)
    write_csv(sampled)


# Example usage:
if __name__ == "__main__":
    pcr_pipeline(
        seq_length=10,
        initial_count=20,
        cycles=20,
        amp_range=(0.2, 1.0),
        error_prob=0.001,
        seq_error_prob=0.001,
        depth=200,
        seed=42
    )