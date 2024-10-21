import pandas as pd
from functions import amplify_with_errors

# Step 1: Load the initial DataFrame (from file)
df = pd.read_csv('true_UMIs.csv')

# Step 2: Set amplification parameters
amplification_probability = 0.9  # 90% chance of amplification
error_rate = 0.01  # Error rate for polymerase
amplification_cycles = 4  # Number of amplification cycles

# Step 3: Call the amplification function
amplify_with_errors(df, amplification_probability, error_rate, amplification_cycles, output_filename='amplified_UMIs.csv')
