from evo_science.entities.metrics import WAPE

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Assuming the WAPE metric is in a module named metrics.py

    # Create a sample DataFrame with true values and predicted values
    data = {"y_true": [100, 150, 200, 250, 300], "y_pred": [110, 140, 190, 260, 310]}

    df = pd.DataFrame(data)

    # Convert DataFrame columns to numpy arrays
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    # Initialize WAPE metric
    wape_metric = WAPE()

    # Calculate WAPE
    wape_value = wape_metric._calculate_np(y_true, y_pred)

    # Print the result
    print(f"WAPE: {wape_value:.2f}%")
