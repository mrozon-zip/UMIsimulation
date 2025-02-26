import argparse
import csv
import logging
import matplotlib.pyplot as plt
import math
from denoise import Denoiser
from amplifying import pcr_amplification, bridge_amplification, polonies_amplification
from generate import generate_sequences
import os

os.makedirs("results", exist_ok=True)

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
amplify_parser.add_argument('--substrate_capacity', type=float, default=(2 ** 18),
                            help="Initial substrate capacity")
amplify_parser.add_argument('--S', type=float, default=9_000_000, help="Threshold S parameter")
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
amplify_parser.add_argument('--AOE_radius', type=float, default=1, help="Radius of AOE of every active A point")
amplify_parser.add_argument('--simulate', action='store_true', help="Number of amplification cycles")
# TO DO: include --no_simulate with 'store_false'
amplify_parser.add_argument('--density', type=float, default=10, help="Density parameter for Bridge amplification")
amplify_parser.add_argument('--success_prob', type=float, default=0.85,
                            help="Success probability for Bridge amplification")
amplify_parser.add_argument('--deviation', type=float, default=0.1,
                            help="Deviation for Bridge amplification parameters (e.g., 0.1 for 10%)")


# Subcommand: denoise
denoise_parser = subparsers.add_parser('denoise', help="Denoise amplified sequences")
denoise_parser.add_argument('--input', type=str, required=True, help="Input CSV filename for denoising")
denoise_parser.add_argument('--method', type=str, choices=['simple', 'directional'], required=True,
                            help="Denoising method to use")
denoise_parser.add_argument('--threshold', type=int, default=300, help="Threshold for simple denoising")
denoise_parser.add_argument('--output', type=str, default='denoised.csv',
                            help="Output CSV filename for denoised sequences")
denoise_parser.add_argument("--show", type=int, default=3, help='Determine what plots to be shown')
denoise_parser.add_argument('--input_true_barcodes', type=str, default='true_barcodes.csv')

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
            fieldnames = ['sequence', 'N0']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seq in sequences_pcr:
                writer.writerow(seq)
        logging.info(f"PCR amplification complete. Results saved to {pcr_output}.")

    if args.method in ['bridge', 'both_12']:
        sequences_bridge = [dict(seq) for seq in sequences]
        logging.info("Starting Bridge amplification...")
        sequences_bridge_amp, history_bridge, bridge_output = bridge_amplification(
            sequences_bridge,
            mutation_rate=args.mutation_rate,
            mutation_probabilities=mutation_probabilities,
            simulate=args.simulate,
            s_radius=args.S_radius,
            aoe_radius=args.AOE_radius,
            density=args.density,
            success_prob=args.success_prob,
            deviation=args.deviation,
            output=args.output,
        )
        with open(bridge_output, 'w', newline='') as csvfile:
            fieldnames = ['sequence', 'N0']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seq_dict in sequences_bridge_amp:
                writer.writerow(seq_dict)
        logging.info(f"Generated {len(sequences_bridge_amp)} sequences and saved to {bridge_output}")

    if args.method in ['polonies_amplification', 'both_13']:
        sequences_bridge = [dict(seq) for seq in sequences]
        logging.info("Starting Bridge amplification...")
        sequences_polony_amp, hist_polony_amp, polonies_output = polonies_amplification(
            sequences_bridge,
            mutation_rate=args.mutation_rate,
            mutation_probabilities=mutation_probabilities,
            simulate=args.simulate,
            s_radius=args.S_radius,
            aoe_radius=args.AOE_radius,
            density=args.density,
            success_prob=args.success_prob,
            deviation=args.deviation,
            output=args.output,
        )
        with open(polonies_output, 'w', newline='') as csvfile:
            fieldnames = ['sequence', 'N0']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seq_dict in sequences_polony_amp:
                writer.writerow(seq_dict)
        logging.info(f"Generated {len(sequences_polony_amp)} sequences and saved to {polonies_output}")

    if args.plot and args.method in ['both_12', 'both_13', 'bridge', 'polonies_amplification', 'pcr']:
        # Build a configuration dictionary that maps each method to its plotting parameters.
        # Note: history_pcr is only defined for 'pcr', 'both_12', and 'both_13';
        # history_bridge is defined for 'both_12' and 'bridge';
        # hist_polony_amp is defined for 'both_13' and 'polonies_amplification'.
        plot_configs = {
            'both_12': {
                'datasets': [
                    {'data': globals().get('history_pcr'), 'marker': 'o', 'label': 'PCR Amplification'},
                    {'data': globals().get('history_bridge'), 'marker': 's', 'label': 'Bridge Amplification'}
                ],
                'xlabel': "Cycle Number",
                'ylabel': "Total Unique Sequences",
                'title': "Total Sequences per Cycle Comparison"
            },
            'both_13': {
                'datasets': [
                    {'data': globals().get('history_pcr'), 'marker': 'o', 'label': 'PCR Amplification'},
                    {'data': globals().get('hist_polony_amp'), 'marker': 's', 'label': 'Bridge Amplification'}
                ],
                'xlabel': "Cycle Number",
                'ylabel': "Total Unique Sequences",
                'title': "Total Sequences per Cycle Comparison"
            },
            'bridge': {
                'datasets': [
                    {'data': globals().get('history_bridge'), 'marker': 's', 'label': 'Bridge Amplification'}
                ],
                'xlabel': "Cycle Number",
                'ylabel': "Total Unique Sequences",
                'title': "Total Sequences per Cycle Comparison"
            },
            'polonies_amplification': {
                'datasets': [
                    {'data': globals().get('hist_polony_amp'), 'marker': 's', 'label': 'Bridge Amplification'}
                ],
                'xlabel': "Cycle Number",
                'ylabel': "Total Unique Sequences",
                'title': "Total Sequences per Cycle Comparison"
            },
            'pcr': {
                'datasets': [
                    {'data': globals().get('history_pcr'), 'marker': 'o', 'label': None}
                ],
                'xlabel': "Cycle Number",
                'ylabel': "Total Number of Unique Sequences",
                'title': "PCR Amplification: Total Sequences per Cycle"
            }
        }

        config = plot_configs[args.method]
        plt.figure()
        for dataset in config['datasets']:
            data = dataset['data']
            # If the data is not defined (i.e. globals().get returned None), warn and skip plotting that dataset.
            if data is None:
                print(f"Warning: Data for {dataset.get('label') or args.method} is not defined. Skipping.")
                continue
            plt.plot(range(1, len(data) + 1), data, marker=dataset['marker'], label=dataset.get('label'))
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        # Only add a legend if at least one dataset provided a label.
        if any(ds.get('label') and ds.get('data') is not None for ds in config['datasets']):
            plt.legend()
        plt.grid(True)
        plt.show()

elif args.command == 'denoise':
    # Perform denoising
    denoiser = Denoiser(args.input)
    filename = 'results.csv'
    if args.method == "directional":
        before_graph, after_graph, unique_molecules = denoiser.directional_networks(show=args.show)
        central_nodes_data = denoiser.networks_resolver(after_graph, toggle="central_node")
        denoiser.node_probe(after_graph)
        prefix = 'directional'
        # Split the filename into base and extension
        base, ext = os.path.splitext(filename)
        denoised_file = f"{prefix}_{base}{ext}"
        # Save the results to a CSV file
        with open(denoised_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=central_nodes_data[0].keys())
            writer.writeheader()
            writer.writerows(central_nodes_data)

        print(f"Central nodes saved to '{denoised_file}' with {len(central_nodes_data)} rows.")
    elif args.method == "simple":
        threshold_value = args.threshold
        valid_sequences = denoiser.simple(threshold_value)

        prefix = 'simple'
        # Use the input filename from the arguments (not a hardcoded variable)
        base, ext = os.path.splitext(args.input)
        denoised_file = f"{prefix}_{base}{ext}"

        # Write the filtered data to the computed output CSV file
        with open(denoised_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=valid_sequences[0].keys())
            writer.writeheader()
            writer.writerows(valid_sequences)

        print(f"Enhanced results saved to '{denoised_file}' with {len(valid_sequences)} rows.")
    else:
        print("Invalid input in method choice")

    true_barcodes = args.input_true_barcodes
    denoiser.analysis(denoised_file, true_barcodes)