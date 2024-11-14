from classes import Denoiser
import pandas as pd

denoiser = Denoiser(csv_file='amplified_UMIs.csv')

denoiser_results = pd.read_csv("central_nodes.csv")
print(denoiser_results.iloc[:, 0])
true_UMIs = pd.read_csv("true_UMIs.csv")
print(true_UMIs.iloc[:, 2])

denoiser.analysis(denoiser_results, true_UMIs, col1=0, col2=2)
