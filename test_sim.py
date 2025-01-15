from classes3 import SimPcr

do_simulation = True

if do_simulation == True:
    # Example usage
    simulator = SimPcr(length=12, number_of_rows=100)
    simulator.create_true_UMIs()
    error_types = {
        'substitution': 0.6,  # 60% chance of substitution
        'deletion': 0.2,      # 20% chance of deletion
        'insertion': 0.2      # 20% chance of insertion
    }
    simulator.amplify_with_errors(amplification_probability=0.9, error_rate=0.004, error_types=error_types, amplification_cycles=6)
elif do_simulation == False:
    print("Omiting simulation")

# File paths for the data
true_umis_file = "true_UMIs.csv"
