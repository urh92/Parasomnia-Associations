# Import necessary libaries and packages
from helper_functions import *
import pandas as pd
import numpy as np


class DataProcessor:
    """
    Processes sleep study data, including cleaning, feature calculation, and exporting results.
    """

    def __init__(self, respiratory_path, stages_path, spo2_path, arousals_path):
        """
        Initializes the SleepStudyProcessor with the paths to the necessary CSV files.
        """
        self.respiratory_path = respiratory_path
        self.stages_path = stages_path
        self.spo2_path = spo2_path
        self.arousals_path = arousals_path
        self.data = pd.DataFrame()

    def load_and_merge_data(self):
        """
        Loads and merges respiratory, stages, SpO2, and arousals data into a single DataFrame.
        """
        # Load each dataset
        respiratory_df = pd.read_csv(self.respiratory_path)
        stages_df = pd.read_csv(self.stages_path)
        spo2_df = pd.read_csv(self.spo2_path)
        arousals_df = pd.read_csv(self.arousals_path)

        # Merge datasets on 'ID'
        self.data = respiratory_df.merge(stages_df, on='ID').merge(spo2_df, on='ID').merge(arousals_df, on='ID')
        print("Data loaded and merged.")

    def preprocess_data(self):
        """
        Preprocesses the data by handling missing values and calculating new features.
        """
        self.data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values
        self.data = self.data.drop(columns=['AHI', 'AHI_NREM', 'AHI_REM', 'TST_x', 'WASO_x', 'Percentage_Stage1_x', 
                                            'Percentage_Stage2_x', 'Percentage_SWS_x', 'Percentage_NREM_x', 'Percentage_REM_x', 
                                            'Latency_REM_x', 'ROL', 'Duration_NREM_x', 'Duration_REM_x', 'SME'])

        self.data = self.data.rename(columns={'DisplayGender': 'Gender', 'SO': 'SOL', 'TST_y': 'TST', 'Percentage_NREM_y': 'Percentage_NREM',
                                              'Percentage_REM_y': 'Percentage_REM', 'Latency_REM_y': 'Latency_REM', 
                                              'Percentage_Stage1_y': 'Percentage_Stage1', 'Percentage_Stage2_y': 'Percentage_Stage2',
                                              'Percentage_SWS_y': 'Percentage_SWS', 'WASO_y': 'WASO', 'Duration_NREM_y': 'Duration_NREM', 
                                              'Duration_REM_y': 'Duration_REM'})
        
        self.data[['Percentage_Stage1', 'Percentage_Stage2', 'Percentage_SWS', 'Percentage_REM']] = self.data[['Percentage_Stage1', 'Percentage_Stage2', 'Percentage_SWS', 'Percentage_REM']].apply(fill_nan, axis=1)

        print("Data preprocessed.")

    def calculate_features(self):
        """
        Calculates new features from the existing data and adds them to the DataFrame.
        """
        self.data = insomnia_score(self.data)
        self.data = compute_score(df=self.data, cols=allergy_cols, label='Allergy_Score')
        self.data = compute_score(df=self.data, cols=muscoskeletal_cols, label='Muscoskeletal_Pain_Score')
        self.data = compute_score(df=self.data, cols=reflux_cols, label='Reflux_Score')
        self.data = compute_score(df=self.data, cols=cardiopulmonary_cols, label='Cardiopulmonary_Score')
        self.data = compute_score(df=self.data, cols=other_cols, label='Other_Score')
        self.data = compute_score(df=self.data, cols=anxiety_cols, label='Anxiety_Score')
        self.data['PH3_Sleep_Hours'] = self.data['PH3_Sleep_Hours'].apply(convert_sleep_hours)
        self.data['Percentage_Stage1_2'] = self.data['Percentage_Stage1'] + self.data['Percentage_Stage2']
        self.data['OA_HYP_Duration'] = (self.data['Durations_HI_Average_Duration'] * self.data['Numbers_HI_Total'] + self.data['Durations_OA_Average_Duration'] * self.data['Numbers_OA_Total'])/(self.data['Numbers_OA_Total'] + self.data['Numbers_HI_Total'])
        
        conditions = [
        (self.data['PH2_Shift_2nd'] + self.data['PH2_Shift_3rd'] > 0),
        (self.data['PH2_Shift_1st'] == 1) & (self.data['PH2_Shift_2nd'] == 0) & (self.data['PH2_Shift_3rd'] == 0),
        (self.data['PH2_Shift_1st'] == 0) & (self.data['PH2_Shift_2nd'] == 0) & (self.data['PH2_Shift_3rd'] == 0)
        ]

        choices = [1, 0, 0]

        self.data['PH2_Shiftworker'] = np.select(conditions, choices, default=np.nan)
        print("Features calculated.")

    def run(self):
        """
        Executes the entire data processing workflow.
        """
        self.load_and_merge_data()
        self.preprocess_data()
        self.calculate_features()


if __name__ == "__main__":
    respiratory_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Respiratory.csv'
    stages_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Stages.csv'
    spo2_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/SpO2.csv'
    arousals_path = 'C:/Users/UmaerHANIF/Documents/Morpheus_SQL/Arousals.csv'

    # Create an instance of the DataProcessor and run the processing
    processor = DataProcessor(respiratory_path, stages_path, spo2_path, arousals_path)
    processor.run()