import argparse
import json
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
import os

sys.path.append(os.path.abspath(os.path.pardir))


class ModelTrainer:
    def __init__(self, df, config_file, result_file, checkpoint_dir, n_splits=2):
        # Load model configuration and parameters
        self.df = df.iloc[:1000]
        self.config_file = config_file
        self.result_file = result_file
        self.checkpoint_dir = checkpoint_dir
        self.n_splits = n_splits  # Number of splits for cross-validation

        # Load configuration
        self.config = self.load_config(config_file)

        # Prepare data (Preprocessing steps)
        self.prepare_data()

    def load_config(self, config_file):
        print(f"Loading configuration from {config_file}...")
        with open(config_file, "r") as f:
            return json.load(f)

    def prepare_data(self):
        print("[INFO] Preparing the data for model training...")
        # Ensure 'TransactionMonth' is in datetime format
        self.df["TransactionMonth"] = pd.to_datetime(self.df["TransactionMonth"])

        # Extract 'Year' and 'Month' into separate columns
        self.df["Year"] = self.df["TransactionMonth"].dt.year.astype("object")
        self.df["Month"] = self.df["TransactionMonth"].dt.month.astype("object")

        # Define categorical columns
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

        print(f"[INFO] Numerical columns: {self.numerical_columns}")
        print(f"[INFO] Categorical columns: {self.categorical_columns}")

        # Drop 'TransactionMonth' column
        self.df.drop("TransactionMonth", axis=1, inplace=True)

        # Separate features and targets
        X = self.df.drop(columns=["TotalPremium", "TotalClaims"])
        y_total_premium = self.df["TotalPremium"]
        y_total_claims = self.df["TotalClaims"]
        self.df.drop(columns=["TotalPremium", "TotalClaims"], inplace=True)

        # Filter columns based on availability in the dataframe
        self.numerical_columns = [
            col for col in self.numerical_columns if col in self.df.columns
        ]
        self.categorical_columns = [
            col for col in self.categorical_columns if col in self.df.columns
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
        self.X_transformed = self.data_preprocessor.fit_transform(X)

        # Split data for both targets (TotalPremium and TotalClaims)
        (
            self.X_train_premium,
            self.X_test_premium,
            self.y_train_premium,
            self.y_test_premium,
        ) = train_test_split(
            self.X_transformed, y_total_premium, test_size=0.3, random_state=42
        )
        (
            self.X_train_claims,
            self.X_test_claims,
            self.y_train_claims,
            self.y_test_claims,
        ) = train_test_split(
            self.X_transformed, y_total_claims, test_size=0.3, random_state=42
        )
        # Save the splits into npz arrays
        np.savez(
            "C:/dev/Side-Projects/10 Acadamy/W3 Challenge/Car-Insurance-Data-Analysis/data/processed/data_splits.npz",
            X_train_premium=self.X_train_premium,
            X_test_premium=self.X_test_premium,
            y_train_premium=self.y_train_premium,
            y_test_premium=self.y_test_premium,
            X_train_claims=self.X_train_claims,
            X_test_claims=self.X_test_claims,
            y_train_claims=self.y_train_claims,
            y_test_claims=self.y_test_claims,
        )
        print("[INFO] Data splits saved to npz file.")

        print("[INFO] Data preparation complete. Proceeding to model training.")

    def train_and_evaluate(self):
        print(
            "/nStarting model training and evaluation with 5-fold cross-validation.../n"
        )

        # Initialize result dictionary
        results = {}

        # Create KFold cross-validation split
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Loop over each target ('TotalPremium', 'TotalClaims')
        for target, models in self.config["models"].items():
            print(f"/nTarget: {target}")

            # Loop over each model for the current target
            for model_name, model_config in models.items():
                print(f"/nTraining model: {model_name} for target: {target}...")

                model_type = model_config["model_type"]
                model_params = model_config["params"]

                # Initialize model
                if model_type == "RandomForestRegressor":
                    print(f"Initializing {model_name} with parameters: {model_params}")
                    model = RandomForestRegressor(**model_params)
                elif model_type == "LinearRegression":
                    print(f"Initializing {model_name} with parameters: {model_params}")
                    model = LinearRegression(**model_params)
                elif model_type == "XGBRegressor":
                    print(f"Initializing {model_name} with parameters: {model_params}")
                    model = XGBRegressor(**model_params)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Determine the appropriate features and target variables
                if target == "TotalPremium":
                    X_train, y_train = self.X_train_premium, self.y_train_premium
                elif target == "TotalClaims":
                    X_train, y_train = self.X_train_claims, self.y_train_claims
                else:
                    raise ValueError(f"Unknown target: {target}")

                # Cross-validation for the current target
                print(
                    f"/nTraining {model_name} for '{target}' with cross-validation..."
                )
                target_results = self.cross_validate(model, X_train, y_train)
                print(
                    f"Cross-validation complete for '{target}'. Mean MSE: {target_results['mean_mse']}, Mean R/u00b2: {target_results['mean_r2']}"
                )

                # Save model checkpoint
                print(f"/nSaving {model_name} model checkpoint for target: {target}...")
                self.save_model(model, f"{model_name}_{target}")

                # Store results along with model configuration
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][target] = {
                    "model_config": model_config,
                    "results": target_results,
                }

        # Ensure the directory for saving results exists
        output_file = "logs/result.yml"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save results in YAML file
        print("/nSaving results to result.yml...")
        with open(output_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False)
        print("/nModel training and evaluation complete!")

    def cross_validate(self, model, X, y):
        results = {"mse": [], "r2": []}
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_index, val_index in kfold.split(X):
            # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Fit the model
            model.fit(X_train, y_train)

            # Predict on the validation set
            predictions = model.predict(X_val)

            # Calculate performance metrics
            mse = mean_squared_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            results["mse"].append(mse)
            results["r2"].append(r2)

        # Convert metrics to standard Python floats
        return {
            "mean_mse": float(np.mean(results["mse"])),
            "mean_r2": float(np.mean(results["r2"])),
        }

    def save_model(self, model, model_name):
        # Save model to a pickle file
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_model.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model checkpoint for {model_name} saved at {checkpoint_path}")

    def save_results(self, results):
        # Save the results as a YAML file
        with open(self.result_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to {self.result_file}")


def main():
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(
        description="Train and evaluate multiple models with 5-fold cross-validation."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the input data CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration JSON file.",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to save the results YAML file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save the model checkpoints.",
    )
    args = parser.parse_args()

    # Load the data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Instantiate and train the models
    trainer = ModelTrainer(
        df,
        config_file=args.config,
        result_file=args.result_file,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
