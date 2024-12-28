import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def missing_values_proportions(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    missing_proportions = (missing_values / len(df)) * 100
    missing_proportions = missing_proportions.round(2)

    return pd.DataFrame(
        {"Missing Values": missing_values, "Proportion (%)": missing_proportions}
    )


def handle_outliers(df, columns, plot_box=False, replace_with="boundaries"):
    """Detect and handle outliers in specified columns of a DataFrame."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if replace_with == "boundaries":
            # Replace outliers with boundaries
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif replace_with == "mean":
            # Replace outliers with the mean
            mean = df[col].mean()
            df[col] = np.where(
                (df[col] < lower_bound) | (df[col] > upper_bound), mean, df[col]
            )
        else:
            raise ValueError("replace_with must be either 'boundaries' or 'mean'")

        if plot_box:
            plt.figure(figsize=(6, 0.3))
            sns.boxplot(x=df[col])
            plt.title(f"Box Plot for {col} replaced with {replace_with} ")
            plt.show()

    return df
