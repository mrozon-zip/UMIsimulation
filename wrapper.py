import csv
import pysam


def csv_to_sam(input, output):
    with open(input, "r") as csvfile, open(output, "w") as samfile:
        # Write a minimal SAM header. Here we define one reference sequence.
        samfile.write("@HD\tVN:1.0\tSO:coordinate\n")
        samfile.write("@SQ\tSN:chr1\tLN:1000000\n")

        reader = csv.DictReader(csvfile)
        first_row = next(reader)
        sequence_length = len(first_row["sequence"].strip())
        cigar_val = f"{sequence_length}M"
        read_id = 1

        # Process the first row
        umi = first_row["sequence"].strip()
        count = int(first_row["N0"])
        for i in range(count):
            sam_line = (
                f"read{read_id}\t0\tchr1\t1000\t255\t{cigar_val}\t*\t0\t0\t*\t*\tUB:Z:{umi}\n"
            )
            samfile.write(sam_line)
            read_id += 1

        # Process remaining rows
        for row in reader:
            umi = row["sequence"].strip()
            count = int(row["N0"])
            for i in range(count):
                sam_line = (
                    f"read{read_id}\t0\tchr1\t1000\t255\t{cigar_val}\t*\t0\t0\t*\t*\tUB:Z:{umi}\n"
                )
                samfile.write(sam_line)
                read_id += 1

def bam_to_csv(input, output):
    # Dictionary to count occurrences of each UMI sequence
    umi_counts = {}

    # Open the BAM file for reading in binary mode.
    with pysam.AlignmentFile(input, "rb") as bam:
        # Iterate through every alignment record in the BAM file.
        for read in bam.fetch(until_eof=True):
            try:
                # Extract the UMI from the "UB" tag.
                umi = read.get_tag("UB")
            except KeyError:
                # If there is no UB tag, skip the record.
                continue

            # Increment the count for this UMI
            if umi in umi_counts:
                umi_counts[umi] += 1
            else:
                umi_counts[umi] = 1

    # Write the collected UMI counts to a CSV file.
    with open(output, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["sequence", "N0"])

        # Write each UMI and its count.
        for umi, count in umi_counts.items():
            writer.writerow([umi, count])

    print(f"CSV file has been written to {output} with {len(umi_counts)} unique UMI sequences.")
