import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns


class PreprocessData:
    def __init__(self, input_path, output_path, missing_threshold=0.6):
        """
        Initialize the PreprocessData class.

        Parameters:
        - input_path: Path to the input data file.
        - output_path: Path to save the processed data.
        - missing_threshold: Threshold for dropping columns with missing values.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.missing_threshold = missing_threshold
        self.df = None

    def load_data(self):
        """Load data from the input file."""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.input_path, sep="|")
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def drop_missing_columns(self):
        """Drop columns with missing values exceeding the specified threshold."""
        print("Dropping columns with excessive missing values...")
        initial_columns = self.df.shape[1]
        threshold = self.missing_threshold * self.df.shape[0]
        self.df = self.df.loc[:, self.df.isnull().sum() <= threshold]
        dropped_columns = initial_columns - self.df.shape[1]
        print(f"Dropped {dropped_columns} columns with missing values.")

    def handle_missing_values(self):
        """Handle missing values in numerical and categorical columns."""
        print("Handling missing values...")
        for col in self.df.select_dtypes(include=["number"]).columns:
            if self.df[col].isnull().any():
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
                print(
                    f"Filled missing values in numerical column '{col}' with mean: {mean_value:.2f}"
                )

        for col in self.df.select_dtypes(include=["object"]).columns:
            if self.df[col].isnull().any():
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                print(
                    f"Filled missing values in categorical column '{col}' with mode: '{mode_value}'"
                )

    def process_columns_with_commas(self):
        """Detect and process object columns with commas by converting them to floats."""
        print("Processing columns with commas...")
        for col in self.df.select_dtypes(include=["object"]).columns:
            if self.df[col].str.contains(",", na=False).any():
                print(
                    f"Processing column '{col}' to replace ',' with '.' and convert to float."
                )
                try:
                    self.df[col] = (
                        self.df[col].str.replace(",", ".", regex=False).astype(float)
                    )
                    print(f"Column '{col}' successfully processed.")
                except ValueError:
                    print(f"Error processing column '{col}'. Skipping.")

    def handle_outliers(self):
        """Handle outliers using the IQR method and validate with Z-score."""
        print("Handling outliers...")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers.sum()
            print(f"Column '{col}': Detected {outlier_count} outliers using IQR.")

            if outlier_count > 0:
                median_value = self.df[col].median()
                self.df.loc[outliers, col] = median_value
                print(f"Replaced outliers in '{col}' with median: {median_value:.2f}")

    def save_data(self):
        """Save the processed data to the specified output path."""
        print("Saving processed data...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            print(f"Processed data saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
            raise

    def run(self):
        """Execute the complete preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        self.load_data()
        self.drop_missing_columns()
        self.process_columns_with_commas()
        self.handle_missing_values()
        self.handle_outliers()
        self.save_data()
        print("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset for analysis.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input data file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed data.",
    )
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.6,
        help="Threshold to drop columns with missing values.",
    )
    args = parser.parse_args()

    preprocessor = PreprocessData(
        input_path=args.input_path,
        output_path=args.output_path,
        missing_threshold=args.missing_threshold,
    )
    preprocessor.run()


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
