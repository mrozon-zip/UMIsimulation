import argparse
import csv
import logging
import matplotlib.pyplot as plt
import math
from polonies_amplifying import polonies_amplification
from pcr_amplifying import pcr_amplification
import subprocess
from generate import generate_sequences
from analysis import confusion_matrix
from support import denoise
from wrapper import csv_to_sam, bam_to_csv
import os

base_folder = "/Users/krzysztofmrozik/Desktop/SciLifeLab/Projects/PCR_simulation/"

def main():
    parser = argparse.ArgumentParser(description="DNA Amplification Simulation Tool")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: generate
    generate_parser = subparsers.add_parser('generate', help="Generate true barcode sequences")
    generate_parser.add_argument('--num', type=int, required=True, help="Number of sequences to generate")
    generate_parser.add_argument('--length', type=int, required=True, help="Length of each sequence")
    generate_parser.add_argument('--unique', action='store_true', help="Ensure sequences are unique")
    generate_parser.add_argument('--output', type=str, default='true_barcodes.csv', help="Output CSV filename")

    # Subcommand: amplify
    amplify_parser = subparsers.add_parser('amplify', help="Amplify sequences using PCR and/or Bridge amplification")
    amplify_parser.add_argument('--method', type=str, choices=['pcr', 'bridge', 'polonies_amplification', 'both_12',
                                                               'both_13'], required=True,
                                help="Amplification method to use")
    amplify_parser.add_argument('--cycles', type=int, default=30, help="Number of amplification cycles")
    amplify_parser.add_argument('--mutation_rate', type=float, default=0.01,
                                help="Mutation rate per nucleotide per replication event")
    amplify_parser.add_argument('--substitution_prob', type=float, default=0.4,
                                help="Probability of substitution mutation")
    amplify_parser.add_argument('--deletion_prob', type=float, default=0.3, help="Probability of deletion mutation")
    amplify_parser.add_argument('--insertion_prob', type=float, default=0.3, help="Probability of insertion mutation")
    amplify_parser.add_argument('--substrate_capacity', type=int, default=(2 ** 18),
                                help="Initial substrate capacity")
    amplify_parser.add_argument('--S', type=int, default=9_000_000, help="Threshold S parameter")
    amplify_parser.add_argument('--C', type=float, default=1e-9, help="Sharpness parameter C")
    amplify_parser.add_argument('--input', type=str, default='true_barcodes.csv',
                                help="Input CSV filename with true barcodes")
    amplify_parser.add_argument('--plot', dest='plot', action='store_true', help="Enable plotting (default)",
                                default=True)
    amplify_parser.add_argument('--output', type=str, default='amplified.csv')
    amplify_parser.add_argument('--no_plot', dest='plot', action='store_false', help="Disable plotting")

    # Bridge amplification specific parsers:
    amplify_parser.add_argument('--S_radius', type=float, default=10,
                                help="Radius of S area where points are generated")
    amplify_parser.add_argument('--AOE_radius', type=float, default=10, help="Radius of AOE of every active A point")
    amplify_parser.add_argument('--simulate', action='store_true', dest='simulate', help="Simulate amplification")
    amplify_parser.add_argument('--no_simulate', action='store_false', dest='simulate', help="Do not simulate amplification")
    amplify_parser.add_argument('--density', type=float, default=10, help="Density parameter for Bridge amplification")
    amplify_parser.add_argument('--success_prob', type=float, default=0.85,
                                help="Success probability for Bridge amplification")
    amplify_parser.add_argument('--deviation', type=float, default=0.1,
                                help="Deviation for Bridge amplification parameters (e.g., 0.1 for 10%)")

    # Subcommand: denoise
    denoise_parser = subparsers.add_parser('denoise', help="Denoise amplified sequences")
    denoise_parser.add_argument('--input', type=str, required=True, help="Input CSV filename for denoising")
    denoise_parser.add_argument('--type', type=int, required=True, help="type of denoising")
    #denoise_parser.add_argument('--output', type=str, default='denoised.csv',
    #                            help="Output CSV filename for denoised sequences")

    # Subcommand: analyse
    analyse_parser = subparsers.add_parser('analyse', help='Analysis of results')
    analyse_parser.add_argument('--true_barcodes', type=str, default='true_barcodes.csv')
    analyse_parser.add_argument('--denoised', type=str, help='pcr denoising output file')
    analyse_parser.add_argument('--amplified', type=str, help='pcr amplified output file')
    analyse_parser.add_argument('--plot', type=bool, default=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    if args.command == 'generate':
        generate_sequences(args.num, args.length, args.unique, args.output)

    elif args.command == 'amplify':
        # Load true barcodes from CSV.
        sequences = []
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sequences.append({'sequence': row['sequence'], 'N0': int(row['N0'])})
        mutation_probabilities = {
            'substitution': args.substitution_prob,
            'deletion': args.deletion_prob,
            'insertion': args.insertion_prob
        }
        total_prob = sum(mutation_probabilities.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-2):
            logging.error("Cumulative mutation probabilities must equal 1.0")

        if args.method in ['pcr', 'both_12', 'both_13']:
            sequences_pcr = [dict(seq) for seq in sequences]
            logging.info("Starting PCR amplification...")
            sequences_pcr, history_pcr, pcr_output = pcr_amplification(
                sequences_pcr,
                cycles=args.cycles,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                substrate_capacity_initial=args.substrate_capacity,
                s=args.S,
                c=args.C,
                output=args.output,
            )
            with open(pcr_output, 'w', newline='') as f:
                fieldnames = ['sequence', 'N0', 'id', 'parent_id', 'active', 'born']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for seq in sequences_pcr:
                    writer.writerow(seq)
            logging.info(f"PCR amplification complete. Results saved to {pcr_output}.")
            if args.plot:
                plt.plot(range(1, len(history_pcr) + 1), history_pcr)
                plt.xlabel("Cycle")
                plt.ylabel("Value")
                plt.title("PCR Amplification History")
                plt.grid(True)
                plt.show()
                print(history_pcr)

        if args.method in ['polonies_amplification', 'both_13']:
            sequences_bridge = [dict(seq) for seq in sequences]
            logging.info("Starting polonies_amplification...")
            sequences_polony_amp, polonies_output = polonies_amplification(
                s_radius=args.S_radius,
                density=args.density,
                sequences=sequences_bridge,
                aoe_radius=args.AOE_radius,
                success_prob=args.success_prob,
                deviation=args.deviation,
                simulate=args.simulate,
                mutation_rate=args.mutation_rate,
                mutation_probabilities=mutation_probabilities,
                output=args.output,
            )
            with open(polonies_output, 'w', newline='') as csvfile:
                fieldnames = ['sequence', 'N0', 'id', 'parent_id', "x", "y", "born", "active"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for seq_dict in sequences_polony_amp:
                    writer.writerow(seq_dict)
            logging.info(f"Generated {len(sequences_polony_amp)} sequences and saved to {polonies_output}")

        if args.plot and args.method in ['both_12', 'both_13', 'bridge', 'polonies_amplification', 'pcr']:
            # (Plotting code as before)
            plt.show()

    # Suppose this is part of your command handling
    if args.command == 'denoise':

        if args.type == 1:
            input_file = args.input
            input_file_simple = os.path.basename(args.input)
            root, _ = os.path.splitext(input_file_simple)

            # File names for intermediate and final outputs.
            output_sam = f"{root}.sam"  # Used by csv_to_sam()
            output_csv = f"{root}_denoised.csv"  # Final CSV output

            # Convert CSV input to a SAM file.
            csv_to_sam(input_file, output_sam)

            # Create a multi-line bash command string.
            # - The first line converts SAM to BAM using samtools.
            # - The second line sorts the BAM file.
            # - The third line deduplicates the sorted BAM using umi_tools.
            bash_command = f"""
            samtools view -S -b {root}.sam > {root}.bam
            samtools sort {root}.bam -o sorted.bam
            samtools index sorted.bam
            umi_tools dedup -I sorted.bam -S deduped.bam --output-stats=deduplicated --extract-umi-method=tag --umi-tag=UB --method=directional
            """

            # Run the bash command.
            # Using shell=True allows you to run a multi-line command.
            # check=True will raise an error if any command fails.
            subprocess.run(bash_command, shell=True, check=True)

            # Use the deduplicated BAM file as input for the CSV conversion.
            input_bam = f"{base_folder}deduped.bam"
            bam_to_csv(input_bam, output_csv)
        elif args.type == 2:
            denoise(args.input, 2)


    elif args.command == 'analyse':
        con_mat = confusion_matrix(denoised_file=args.denoised,
                                    amplified=args.amplified,
                                    true_umis_file=args.true_barcodes,
                                    plot=args.plot)
        # Write confusion matrix metrics to CSV file named after amplified file
        amplified_file = args.amplified
        # Extract the filename stem without directory or extension
        file_stem = os.path.splitext(os.path.basename(amplified_file))[0]
        params_stem = file_stem.replace('amplified', '')
        metrics_stem = file_stem.replace('amplified', 'metrics')

        # Determine amplification type and parse parameters
        parts = params_stem.split('_')
        print(parts)
        extra_fields = {}
        if parts[0] == 'pcr':
            extra_fields['amplification_type'] = 'pcr'
            mut_idx = parts.index('mut')
            extra_fields['mutation_rate'] = parts[mut_idx + 1]
            extra_fields['cycles'] = parts[mut_idx + 2]
        elif parts[0] == 'polonies':
            extra_fields['amplification_type'] = 'polonies'
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    extra_fields[parts[i]] = parts[i + 1]

        # Combine metrics with parameters
        row = {**con_mat, **extra_fields}
        print(row)
        print(extra_fields)

        # Write header and row with dynamic fields
        fieldnames = ["TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "specificity"] + list(extra_fields.keys())
        metrics_filename = 'results_metrics/' + metrics_stem + '_metrics.csv'
        with open(metrics_filename, 'w', newline='') as mf:
            writer = csv.DictWriter(mf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        logging.info(f"Confusion matrix metrics saved to {metrics_filename}")



if __name__ == "__main__":
    # Optional: on Windows, call freeze_support() if needed:
    # from multiprocessing import freeze_support
    # freeze_support()
    main()