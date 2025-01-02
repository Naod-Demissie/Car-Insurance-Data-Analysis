import argparse
import itertools
import json
import random
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from importlib import import_module
import os


class ModelTuner:
    def __init__(
        self,
        df,
        config_file,
        results_path,
        model_output_path,
    ):
        """
        Initialize the tuner with a dataset and configuration file.

        Parameters:
        - df: DataFrame containing the dataset.
        - config_file: Path to the tuner.json file defining models and parameter grids.
        - results_path: Path to save tuning results.
        - model_output_path: Path to save the best model.
        """
        self.df = df.iloc[:1000]
        self.config = self.load_config(config_file)
        self.results_path = results_path
        self.model_output_path = model_output_path

    def prepare_data(self):
        print("[INFO] Preparing the data for model training...")
        # Ensure 'TransactionMonth' is in datetime format
        self.df["TransactionMonth"] = pd.to_datetime(self.df["TransactionMonth"])

        # Extract 'Year' and 'Month' into separate columns
        self.df["Year"] = self.df["TransactionMonth"].dt.year.astype("object")
        self.df["Month"] = self.df["TransactionMonth"].dt.month.astype("object")

        # Define categorical and numerical columns
        self.categorical_columns = [
            "Year",
            "Month",
            "UnderwrittenCoverID",
            "PostalCode",
            "PolicyID",
            "IsVATRegistered",
            "Citizenship",
            "LegalType",
            "Title",
            "Language",
            "Bank",
            "AccountType",
            "MaritalStatus",
            "Gender",
            "Country",
            "Province",
            "MainCrestaZone",
            "SubCrestaZone",
            "ItemType",
            "VehicleType",
            "make",
            "Model",
            "bodytype",
            "VehicleIntroDate",
            "AlarmImmobiliser",
            "TrackingDevice",
            "NewVehicle",
            "WrittenOff",
            "Rebuilt",
            "Converted",
            "CrossBorder",
            "TermFrequency",
            "ExcessSelected",
            "CoverCategory",
            "CoverType",
            "CoverGroup",
            "Section",
            "Product",
            "StatutoryClass",
            "StatutoryRiskType",
        ]
        self.numerical_columns = [
            "mmcode",
            "RegistrationYear",
            "Cylinders",
            "cubiccapacity",
            "kilowatts",
            "NumberOfDoors",
            "CapitalOutstanding",
            "NumberOfVehiclesInFleet",
            "SumInsured",
            "CalculatedPremiumPerTerm",
            "TotalPremium",
            "TotalClaims",
        ]

        # Drop 'TransactionMonth' column
        self.df.drop("TransactionMonth", axis=1, inplace=True)

        # Separate features and targets
        X = self.df.drop(columns=["TotalPremium", "TotalClaims"], errors="ignore")
        y_total_premium = self.df["TotalPremium"]
        y_total_claims = self.df["TotalClaims"]
        self.df.drop(columns=["TotalPremium", "TotalClaims"], inplace=True)

        # Filter columns based on availability
        self.numerical_columns = [
            col for col in self.numerical_columns if col in X.columns
        ]
        self.categorical_columns = [
            col for col in self.categorical_columns if col in X.columns
        ]

        # Preprocess features using ColumnTransformer
        self.data_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numerical_columns),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_columns,
                ),
            ]
        )

        print("[INFO] Applying preprocessing transformations...")
        X_transformed = self.data_preprocessor.fit_transform(X)

        # Split data for both targets (train, val, test)
        (
            X_temp_premium,
            self.X_test_premium,
            y_temp_premium,
            self.y_test_premium,
        ) = train_test_split(
            X_transformed, y_total_premium, test_size=0.2, random_state=42
        )
        (
            self.X_train_premium,
            self.X_val_premium,
            self.y_train_premium,
            self.y_val_premium,
        ) = train_test_split(
            X_temp_premium, y_temp_premium, test_size=0.25, random_state=42
        )

        (
            X_temp_claims,
            self.X_test_claims,
            y_temp_claims,
            self.y_test_claims,
        ) = train_test_split(
            X_transformed, y_total_claims, test_size=0.2, random_state=42
        )
        (
            self.X_train_claims,
            self.X_val_claims,
            self.y_train_claims,
            self.y_val_claims,
        ) = train_test_split(
            X_temp_claims, y_temp_claims, test_size=0.25, random_state=42
        )

        print("[INFO] Data preparation complete.")

    def load_model(self, model_path):
        module_name, class_name = model_path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)

    def load_config(self, config_file):
        print(f"[INFO] Loading configuration from {config_file}...")
        with open(config_file, "r") as f:
            config = json.load(f)
        for model_name, model_details in config.items():
            model_path = model_details["model"]
            config[model_name]["model"] = self.load_model(
                model_path
            )()  # Instantiate the model
        return config

    def tune(self, n_iter=10):
        """
        Perform hyperparameter tuning for both targets: premium and claims using a predefined validation set.

        Parameters:
        - n_iter: Total number of parameter samples to evaluate (default: 10).
        """
        all_results = {"models": {}, "targets": ["TotalPremium", "TotalClaims"]}
        best_model_configs = {
            "models": {}
        }  # Initialize structure to store best configurations

        for target in ["TotalPremium", "TotalClaims"]:
            print(f"[INFO] Starting hyperparameter tuning for {target}...")

            # Select the appropriate datasets
            if target == "TotalPremium":
                X_train, y_train = self.X_train_premium, self.y_train_premium
                X_val, y_val = self.X_val_premium, self.y_val_premium
            elif target == "TotalClaims":
                X_train, y_train = self.X_train_claims, self.y_train_claims
                X_val, y_val = self.X_val_claims, self.y_val_claims

            # Initialize an empty dictionary for this target
            all_results["models"][target] = {}
            best_model_configs["models"][target] = {}

            for model_name, model_config in self.config.items():
                print(f"[INFO] Tuning {model_name} for {target}...")

                model = model_config["model"]
                param_grid = model_config["params"]

                # Generate the full search space
                search_space = [
                    dict(zip(param_grid.keys(), values))
                    for values in itertools.product(*param_grid.values())
                ]

                # Determine the actual number of iterations
                actual_n_iter = min(len(search_space), n_iter)

                # Randomly sample from the search space
                sampled_combinations = random.sample(search_space, actual_n_iter)

                best_model = None
                best_score = float("-inf")
                best_params = None
                best_rmse = None

                # Store all iterations results for this model
                model_results = []

                for i, sampled_params in enumerate(sampled_combinations):
                    print(f"[INFO] Iteration {i + 1} of {actual_n_iter}...")

                    # Initialize model with sampled parameters
                    model.set_params(**sampled_params)

                    # Fit the model on training data
                    model.fit(X_train, y_train)

                    # Ensure X_val is 2D before prediction
                    if len(X_val.shape) == 1:
                        X_val = X_val.reshape(1, -1)

                    # Evaluate the model on the validation set
                    val_predictions = model.predict(X_val)
                    val_score_r2 = model.score(X_val, y_val)  # R² Score
                    val_score_rmse = mean_squared_error(y_val, val_predictions)  # RMSE

                    # Store this iteration's result
                    model_results.append(
                        {
                            "params": sampled_params,
                            "r2_score": val_score_r2,
                            "rmse": val_score_rmse,
                        }
                    )

                    if val_score_r2 > best_score:
                        best_model = model
                        best_score = val_score_r2
                        best_params = sampled_params
                        best_rmse = val_score_rmse

                all_results["models"][target][model_name] = {
                    "best": {
                        "model_type": best_model.__class__.__name__,
                        "params": best_params,
                        "best_r2": best_score,
                        "best_rmse": best_rmse,
                    },
                    "all_results": model_results,  # Store all combinations
                }

                # Store the best configuration for each model in best_model_configs
                best_model_configs["models"][target][model_name] = {
                    "model_type": best_model.__class__.__name__,
                    "params": best_params,
                }

        # Save all results to the results_path
        results_dir = os.path.dirname(self.results_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as f:
                json.dump({}, f)  # Create an empty file if it doesn't exist
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"[INFO] All results saved to {self.results_path}")

        # Ensure the output directory exists for model output
        output_dir = os.path.dirname(self.model_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(self.model_output_path):
            with open(self.model_output_path, "w") as f:
                json.dump({}, f)  # Create an empty file if it doesn't exist
        with open(self.model_output_path, "w") as f:
            json.dump(best_model_configs, f, indent=4)
        print(f"[INFO] Best model configurations saved to {self.model_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DVC Model Tuning")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the tuner JSON config file"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="tuner_results.yml",
        help="Path to save tuning results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_model.json",
        help="Path to save the best model JSON file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    tuner = ModelTuner(
        df, args.config, results_path=args.results, model_output_path=args.output
    )
    tuner.prepare_data()
    tuner.tune(n_iter=5)
