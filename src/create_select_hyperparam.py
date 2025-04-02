import pandas as pd
import pickle

# Create a DataFrame similar to the final selected hyperparameter setting
final_setting = pd.DataFrame({
    'k': [100],
    'lambda': [0.021644],
    'upsilon': [0.050383],
    'uid': [30],
})

# Save using pickle exactly like in the notebook
output_path = "selected_hyperparam"
with open(output_path, "wb") as f:
    pickle.dump(final_setting, f)

print(f"✅ selected_hyperparam file created at: {output_path}")
