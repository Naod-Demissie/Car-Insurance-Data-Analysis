import pandas as pd
from scipy.stats import f_oneway, ttest_ind


class ABHypothesisTester:
    def __init__(self, data):
        self.data = data

    def _anova_analysis(self, category_col, measure_col):
        """
        Conduct ANOVA for specified category and measure columns.
        """
        groups = [
            self.data[self.data[category_col] == group][measure_col]
            for group in self.data[category_col].unique()
        ]
        result = f_oneway(*groups)
        return {
            "Test Type": "ANOVA",
            "Category": category_col,
            "Measure": measure_col,
            "F-Statistic": float(result.statistic),
            "p-Value": float(result.pvalue),
            "Reject Null": bool(result.pvalue < 0.05),
        }

    def _ttest_analysis(self, category_col, measure_col, group_one, group_two):
        """
        Conduct T-Test between two groups of a category column.
        """
        data_one = self.data[self.data[category_col] == group_one][measure_col]
        data_two = self.data[self.data[category_col] == group_two][measure_col]
        result = ttest_ind(data_one, data_two, equal_var=False)
        return {
            "Test Type": "T-Test",
            "Category": category_col,
            "Measure": measure_col,
            "Groups Compared": (group_one, group_two),
            "T-Statistic": float(result.statistic),
            "p-Value": float(result.pvalue),
            "Reject Null": bool(result.pvalue < 0.05),
        }

    def analyze_province_risk(self):
        """
        Evaluate risk variations across provinces based on total claims.
        """
        return self._anova_analysis(category_col="Province", measure_col="TotalClaims")

    def analyze_zipcode_risk(self):
        """
        Evaluate risk variations across zip codes based on total claims.
        """
        return self._anova_analysis(
            category_col="PostalCode", measure_col="TotalClaims"
        )

    def analyze_zipcode_margin(self):
        """
        Evaluate profit margin variations across zip codes.
        """
        self.data["Margin"] = self.data["TotalPremium"] - self.data["TotalClaims"]
        return self._anova_analysis(category_col="PostalCode", measure_col="Margin")

    def analyze_gender_risk(self):
        """
        Evaluate risk variations between genders based on total claims.
        """
        return self._ttest_analysis(
            category_col="Gender",
            measure_col="TotalClaims",
            group_one="Male",
            group_two="Female",
        )
