# Car Insurance Data Analysis
The project focuses on analyzing car insurance data using EDA, hypothesis testing, and statistical modeling to uncover patterns and build predictive models for better decision-making.

## Project Structure

```
├── checkpoints/
│   ├── LinearRegression_TotalClaims_model.pkl      
│   ├── LinearRegression_TotalPremium_model.pkl     
│   ├── RandomForest_TotalClaims_model.pkl          
│   ├── RandomForest_TotalPremium_model.pkl         
│   ├── XGBoost_TotalClaims_model.pkl               
│   ├── XGBoost_TotalPremium_model.pkl              
│
├── configs/
│   ├── best_tuned_model.json                       
│   ├── tuner.json                                  
│
├── data/
│   ├── processed/                                  
│   ├── raw/                                        
│
├── logs/
│   ├── result.yml                                  
│   ├── tuner_results.yml                           
│
├── notebooks/
│   ├── 1.0-insurance-data-exploration.ipynb        
│   ├── 3.0-ab_hypothesis_testing.ipynb            
│   ├── 4.0-statistical-modeling.ipynb             
│   ├── README.md                                   
│   ├── __init__.py                                 
│
├── scripts/
│   ├── README.md                                   
│   ├── __init__.py                                 
│
├── src/
│   ├── interprate.py                               
│   ├── preprocess.py                               
│   ├── test_hypothesis.py                          
│   ├── train.py                                    
│   ├── tune.py                                     
│   ├── __init__.py                                 
│
├── tests/
│   ├── __init__.py                                 
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <project_directory>
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv/Scripts/activate`
   pip install -r requirements.txt
   ```

## Contribution

Feel free to fork the repository, make improvements, and submit pull requests.
