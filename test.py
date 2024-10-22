from classes import simPCR

# Example usage
simulator = simPCR(length=24, number_of_rows=100)
simulator.create_true_UMIs()
simulator.true_umis_analyze()
simulator.amplify_with_errors(amplification_probability=0.9, error_rate=0.01, amplification_cycles=4)
simulator.pcr_analyze()
