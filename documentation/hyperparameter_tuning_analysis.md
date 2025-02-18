# Overview 

This Jupyter Notebook analyzes the results of hyperparameter tuning experiments to identify the best-performing hyperparameter combination for a generative model.

# Workflow

## Load Results:
Reads .pkl files generated by the hyperparameter tuning script (`hyperparameter_tuning_fit.py`). Extracts key metrics (e.g., LLPD, log-likelihood, etc.) and stores them in a DataFrame for further analysis.

## Filter and Process Data:
Groups results by unique hyperparameter combinations (`λ`, `ϑ`, `k`) to compute:

Mean LLPD across multiple runs.

Standard deviation of LLPD to measure variability.
Identifies the hyperparameter combination with the highest mean LLPD.

## Visualize Results:
Plots key metrics (e.g., LLPD) to compare the performance of different hyperparameter combinations. Uses visualizations like bar charts, scatter plots, or parallel coordinates to highlight trends.

## Output:
Summarizes the best-performing hyperparameter combination and exports results (e.g., as a CSV or image) for documentation or further use.
