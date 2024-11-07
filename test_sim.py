from classes import simPCR

# Example usage
simulator = simPCR(length=12, number_of_rows=100)
simulator.create_true_UMIs()
simulator.true_UMIs_analyze()
error_types = {
    'substitution': 0.6,  # 60% chance of substitution
    'deletion': 0.2,      # 20% chance of deletion
    'insertion': 0.2      # 20% chance of insertion
}
simulator.amplify_with_errors(amplification_probability=0.9, error_rate=0.02, error_types=error_types, amplification_cycles=6)
simulator.PCR_analyze()
