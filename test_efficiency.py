import numpy as np
import pandas as pd
import scipy as sp

df = pd.read_csv("amplified_UMIs.csv")

n_cycles = 12
efficiency = 0.9
error_rate = 0.0001
length = 24
numerator = (1 + efficiency*np.exp(-error_rate*length))**n_cycles
denominator = (1+efficiency)**n_cycles
proportion_no_error = numerator/denominator
print(proportion_no_error)
mu = error_rate # set your value
G = length  # set your value
lmbda = efficiency # set your value
n = n_cycles  # set your value
mutation_list = range(5)  # number of mutations

for m in mutation_list:
    # Analytical expression for every mutation
    part1 = (mu*G)**m
    part2 = (1 + lmbda * np.exp(-mu*G))**n
    part3 = sp.special.factorial(m) * ((1 + lmbda)**n)
    part4 = (n * lmbda * np.exp(-mu*G) / (lmbda * np.exp(-mu*G) + 1))**m  # Expectation of Binomial distribution
    # Combine all parts
    result = (part1 * part2 / part3) * part4
    print(f"Fraction of sequences that exhibit {m} mutations:", result)