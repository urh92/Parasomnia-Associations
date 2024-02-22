from helper_functions import *
from data_processing import *
import pandas as pd
import numpy as np
import statsmodels.api as sm

def preprocess_data():
    """
    Load and prepare the data.
    """
    respiratory_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Respiratory.csv'
    stages_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Stages.csv'
    spo2_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/SpO2.csv'
    arousals_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Arousals.csv'

    # Create an instance of the DataProcessor and run the processing
    processor = DataProcessor(respiratory_path, stages_path, spo2_path, arousals_path)
    processor.run()
    df_meta = processor.data
    return df_meta

def handle_covariates(df, covariates):
    """
    Process covariates within the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        covariates (list): List of covariates to process.
    
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    if 'AHI_Delta' in covariates:
        df = calc_ahi_features(df, 'AHI_Delta')
    if 'Percentage_REM_Ratio' in covariates:
        df = calc_ahi_features(df, 'Percentage_REM_Ratio')
    for var in covariates:
        df = clean_data(dataframe=df, column=var)
    df = df.dropna(subset=covariates)
    return df

def perform_statistical_analysis(df, y, covariates, co_var=False):
    """
    Perform statistical analysis on the dataframe for a given y column.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze.
        y (str): The dependent variable column name.
        covariates (list): List of covariate column names.
    
    Returns:
        pd.DataFrame: A dataframe with the analysis results.
    """
    results = []  # This will store the results before creating a DataFrame
    columns = [col for col in df.columns if col not in psg_cols + ahi_features]
    
    for col in columns:
        if col in y_cols or (col in covariates and co_var):
            continue
        df_temp = df.copy().dropna(subset=[col])
        df_temp = clean_data(dataframe=df_temp, column=col)
        
        if co_var:
            formula = y + " ~ " + col + ' + ' + ' + '.join(covariates)
        else:
            formula = y + " ~ " + col
        model = sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=df_temp)
        result = model.fit()
        
        # Process and store the results for each column
        process_and_store_results(result, col, df_temp, y, results)
    
    # Convert results to DataFrame and return
    return pd.DataFrame(results)

def process_and_store_results(result, col, df_temp, y, results):
    """
    Process the statistical analysis results and store them in the results list.
    
    Args:
        result (ModelResults): The result of the statistical model fitting.
        col (str): The independent variable column name.
        df_temp (pd.DataFrame): The temporary dataframe used for the analysis.
        y (str): The dependent variable column name.
        results (list): The list to store the results dictionaries.
    """
    p = result.pvalues
    beta = result.params
    std_err = result.bse
    intercept = beta.pop('Intercept')
    p = p.drop(index=['Intercept'])
    std_err = std_err.drop(index=['Intercept'])
    
    if co_var:
        beta = beta.drop(index=co_variates)
        p = p.drop(index=co_variates)
        std_err = std_err.drop(index=co_variates)
    
    col = p.index.item()
    intercept = intercept.item()
    p_value = p.item()
    beta = beta.item()
    odds_r = np.exp(beta.item())
    ci1 = np.exp(beta.item() - 1.96 * std_err.item())
    ci2 = np.exp(beta.item() + 1.96 * std_err.item())
    std_error = std_err.item()
        
    df0 = df_temp[df_temp[y] == 0]
    df1 = df_temp[df_temp[y] == 1]
        
    mean1 = df0[col].mean()
    std1 = df0[col].std()
    mean2 = df1[col].mean()
    std2 = df1[col].std()
    n1 = len(df0[col])
    n2 = len(df1[col])
    
    results.append({'Column': col, 'Intercept': intercept, 'Beta': beta, 'Odds Ratio': odds_r, 'CI5': ci1, 'CI95': ci2,
                    'P value': p_value, 'Mean1': mean1, 'Std1': std1, 'Mean2': mean2, 'Std2': stds2, 'N1': n1, 'N2': n2})

def main():
    # Call a function to preprocess the data and load it into df_meta
    df_meta = preprocess_data()
    
    # Iterate through each variable of interest defined in y_cols
    for y in y_cols:
        print(f'Processing: {y}')  # Print the variable currently being processed to the console
        
        # Copy the preprocessed DataFrame and drop rows where the variable of interest is NaN
        df = df_meta.copy().dropna(subset=[y])
        
        # Replace values in the variable column according to a predefined dictionary (others_dict)
        df[y] = df[y].replace(others_dict)
        
        # If covariates are to be considered in the analysis
        if co_var:
            # Call a function to handle the covariates, which may involve filtering or adjusting the DataFrame
            df = handle_covariates(df, co_variates)
        
        # Perform statistical analysis on the DataFrame for the current variable of interest
        # This may involve regression, correlation analysis, etc., depending on the implementation
        df_p = perform_statistical_analysis(df, y, co_variates)
        
        # Define the output path for the results CSV file, incorporating the variable name for clarity
        output_path = f'C:/Users/UmaerHANIF/Documents/Univariate Analyses/Run 1/{y}_Other.csv'
        
        # Sort the results DataFrame by p-values and Beta coefficients, in ascending order of p-values and descending order of Betas
        # This highlights the most statistically significant findings at the top
        df_p.sort_values(by=['P values', 'Beta'], key=abs, ascending=[True, False]).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
