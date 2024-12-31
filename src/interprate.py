import argparse
import pickle
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import yaml
import os


class ModelInterpreter:
    def __init__(self, model_path, data_path, output_path, target_column):
        """
        Initialize the ModelInterpreter.

        Parameters:
        - model_path: Path to the trained model file (pickle file).
        - data_path: Path to the test data file (NPZ file containing test split).
        - output_path: Path to save the interpretation results.
        - target_column: Target column name in the dataset.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.target_column = target_column
        self.model = None
        self.X = None
        self.y = None

    def load_model_and_data(self):
        """Load the model and dataset."""
        # Load model from pickle
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)
        
        # Load dataset from NPZ file
        data = np.load(self.data_path)
        self.X = data['X_test']
        self.y = data['y_test']

    def interpret_with_shap(self):
        """Interpret the model using SHAP and save results."""
        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)

        # Save SHAP summary plot
        shap_output_path = os.path.join(os.path.dirname(self.output_path), "shap_summary.png")
        shap.summary_plot(shap_values, self.X, show=False)
        shap.plt.savefig(shap_output_path)

        # Save SHAP values to YAML
        shap_yaml_path = os.path.join(os.path.dirname(self.output_path), "shap_values.yaml")
        with open(shap_yaml_path, "w") as file:
            yaml.dump({"shap_values": shap_values.values.tolist()}, file)

        return {"summary_plot": shap_output_path, "values_file": shap_yaml_path}

    def interpret_with_lime(self):
        """Interpret the model using LIME and save results."""
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X,
            feature_names=[f'Feature {i+1}' for i in range(self.X.shape[1])],
            class_names=["Prediction"],
            mode="regression"
        )
        lime_output = []
        for i in range(min(5, len(self.X))):  # Limit to first 5 rows for brevity
            exp = explainer.explain_instance(self.X[i], self.model.predict, num_features=10)
            lime_output.append({"row": i, "explanation": exp.as_list()})

        # Save LIME explanations to YAML
        lime_yaml_path = os.path.join(os.path.dirname(self.output_path), "lime_explanations.yaml")
        with open(lime_yaml_path, "w") as file:
            yaml.dump({"lime_explanations": lime_output}, file)

        return {"explanations_file": lime_yaml_path}

    def save_summary(self, shap_results, lime_results):
        """Save a summary of interpretation results."""
        summary = {
            "shap": shap_results,
            "lime": lime_results,
        }
        with open(self.output_path, "w") as file:
            yaml.dump(summary, file)

    def run(self):
        """Run the interpretation process."""
        self.load_model_and_data()
        shap_results = self.interpret_with_shap()
        lime_results = self.interpret_with_lime()
        self.save_summary(shap_results, lime_results)
        print(f"Interpretation saved to {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret a trained model with SHAP and LIME")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (pickle file)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file (NPZ file containing test split)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the interpretation results")
    parser.add_argument("--target_column", type=str, required=True, help="Target column in the dataset")

    args = parser.parse_args()

    interpreter = ModelInterpreter(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        target_column=args.target_column,
    )
    interpreter.run()
