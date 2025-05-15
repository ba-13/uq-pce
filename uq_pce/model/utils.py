import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_input_vs_output(
    samples_array: np.ndarray,
    outputs: np.ndarray,
    dimension_name_map: dict[int, str],
    n_cols: int = 3,
):
    """
    samples_array: shape (n_samples, n_dimensions)
    outputs: shape (n_samples,)
    dimension_name_map: maps dimension index to parameter name
    """
    n_samples, n_dims = samples_array.shape

    # Convert to DataFrame using dimension_name_map
    df = pd.DataFrame(
        {dimension_name_map[i]: samples_array[:, i] for i in range(n_dims)}
    )
    df["Output"] = outputs

    # Plotting
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    n = 0
    for i, (param, ax) in enumerate(zip(dimension_name_map.values(), axes)):
        sns.scatterplot(x=df[param], y=df["Output"], ax=ax, alpha=0.6)
        ax.set_xlabel(param)
        ax.set_ylabel("Model Output")
        ax.set_title(f"Output vs {param}")
        ax.grid(True)
        n += 1

    # Hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
