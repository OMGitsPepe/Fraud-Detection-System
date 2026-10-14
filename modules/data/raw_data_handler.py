import os
import random
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split


class RawDataHandler:
    def __init__(self, storage_path: str, save_path: str):
        """
        Handles extraction, transformation and description of raw data for
        machine learning preprocessing.
        :param storage_path: Path where the raw files are stored
        :param save_path: Path where the cleaned data will be saved
        """
        self.storage_path = storage_path  # Path for data files
        self.save_path = save_path  # Path to save data

    def extract(self, customer_information_filename: str,
                transaction_filename: str, fraud_information_filename: str):
        """
        Reads raw data files and returns them as DataFrames.
        :param customer_information_filename: CSV file containing customers information
        :param transaction_filename: Parquet file containing transaction data
        :param fraud_information_filename: JSON file containing fraud labels
        :return: Tuple of the three created Pandas DataFrames
        """
        # Get paths for customers, transactions and fraud
        customer_path = os.path.join(self.storage_path, customer_information_filename)
        transaction_path = os.path.join(self.storage_path, transaction_filename)
        fraud_path = os.path.join(self.storage_path, fraud_information_filename)

        # Load data
        customer_information = pd.read_csv(customer_path)
        transaction_information = pd.read_parquet(transaction_path)
        fraud_information = pd.read_json(fraud_path, orient="index")
        fraud_information.reset_index(inplace=True)
        fraud_information.columns = ['trans_num', 'is_fraud']

        return customer_information, transaction_information, fraud_information

    def convert_dates(self, transaction_data_frame: pd.DataFrame):
        """
        Convert the trans_date_trans_time column into several distinct columns
        and ensures it's a datetime object.c
        :param transaction_data_frame: Transaction DataFrame with trans_date_trans_time column
        :return: Modified DataFrame with new date and time columns.
        """
        # Convert to datetime
        transaction_data_frame['trans_date_trans_time'] = pd.to_datetime(transaction_data_frame['trans_date_trans_time'], errors='coerce')

        # Drop rows where date conversion failed, if any
        transaction_data_frame.dropna(subset=['trans_date_trans_time'], inplace=True)

        # Extract features
        transaction_data_frame['day_of_week'] = transaction_data_frame['trans_date_trans_time'].dt.day_name()
        transaction_data_frame['hour'] = transaction_data_frame['trans_date_trans_time'].dt.hour
        transaction_data_frame['minute'] = transaction_data_frame['trans_date_trans_time'].dt.minute
        transaction_data_frame['seconds'] = transaction_data_frame['trans_date_trans_time'].dt.second
        transaction_data_frame['day_date'] = transaction_data_frame['trans_date_trans_time'].dt.day
        transaction_data_frame['month_date'] = transaction_data_frame['trans_date_trans_time'].dt.month_name()
        transaction_data_frame['month_num'] = transaction_data_frame['trans_date_trans_time'].dt.month
        transaction_data_frame['year_date'] = transaction_data_frame['trans_date_trans_time'].dt.year

        return transaction_data_frame

    def transform(self, customer_information: pd.DataFrame, transaction_information: pd.DataFrame, fraud_information: pd.DataFrame):
        """
        Prepares and cleans data by merging, dropping missing values and
        duplicates.
        :param customer_information: DataFrame of customer information
        :param transaction_information: DataFrame of transaction information
        :param fraud_information: DataFrame of fraud information
        :return: Cleaned and transformed DataFrame
        """
        # Merge transaction and fraud information on transaction number
        merge_data = transaction_information.merge(fraud_information, on='trans_num', how='left')

        # Merge with customer information on customer ID
        merge_data = merge_data.merge(customer_information, on='cc_num', how='left')

        # Drop duplicate rows
        merge_data = merge_data.drop_duplicates()

        # Drop rows with missing values in is_fraud column
        merge_data = merge_data.dropna(subset=['is_fraud'])

        # Fill missing values in unix_time, category and merchant
        merge_data = merge_data.fillna({
            'unix_time': -1,
            'category': 'Unknown',
            'merchant': 'Unknown'
        })

        # Convert all values in the category column to lowercase
        merge_data['category'] = merge_data['category'].str.lower()

        return merge_data

    def feature_engineer(self, df: pd.DataFrame):
        """
        Creates additional features based on the provided analysis.
        Requires 'trans_date_trans_time' to be datetime object.
        :param df: DataFrame after initial transformation and date conversion.
        :return: DataFrame with new engineered features.
        """
        # Ensure data is sorted for time difference calculation
        df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])

        # Create Time Segment based on hour
        df['time_segment'] = np.where((df['hour'] <= 3) | (df['hour'] >= 22), 'late_night', 'daytime')

        # Cyclical encoding for month using month_num
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

        # Compute time difference between consecutive transactions per customer
        df['time_diff'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 60
        df['time_diff'] = df['time_diff'].fillna(0) # Fill NaN for first transaction
        df['log_time_diff'] = np.log(df['time_diff'] + 1) # Add 1 to avoid log(0)

        # Compute transaction amount z-score within each purchase category
        df['amt_mean'] = df.groupby('category')['amt'].transform('mean')
        df['amt_std'] = df.groupby('category')['amt'].transform('std')
        # Calculate z-score; handle division by zero or NaN std
        df['amt_zscore'] = (df['amt'] - df['amt_mean']) / df['amt_std'].replace(0, np.nan)
        df['amt_zscore'] = df['amt_zscore'].fillna(0) # Fill NaNs arising from std=0 or missing groups

        # One-hot encode time_segment and day_of_week
        df = pd.get_dummies(df, columns=['time_segment', 'day_of_week'], drop_first=False, dummy_na=False) # Avoid NA dummies

        return df

    def remove_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes features that are not used by the fraud detection model.
        This method drops columns that are irrelevant or were used solely for feature engineering.
        :param df: DataFrame *after* partitioning and *before* saving the final dataset.
        :return: DataFrame with unused features removed.
        """
        # List of columns to drop
        columns_to_drop = [
            'trans_num', 'trans_date_trans_time', 'first', 'last', 'sex',
            'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob',
            'minute', 'seconds', 'day_date', 'month_date', 'month_num', 'unix_time', 'year_date',
            'amt_mean', 'amt_std', 'merchant', 'category', 'time_diff', 'merch_lat', 'merch_long',
            'amt', 'hour']

        # Find columns that actually exist in the dataframe to avoid errors
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        # Handle potential duplicate index columns
        potential_merge_indices = ['index_x', 'index_y', 'index']
        for idx_col in potential_merge_indices:
             if idx_col in df.columns and idx_col not in existing_columns_to_drop:
                 existing_columns_to_drop.append(idx_col)

        df_cleaned = df.drop(columns=existing_columns_to_drop)
        print(f"Removed columns: {existing_columns_to_drop}")
        print(f"Remaining columns: {df_cleaned.columns.tolist()}")
        return df_cleaned

    def partition(self, df: pd.DataFrame, test_size_customers: float = 0.2, time_split_quantile: float = 0.8, random_state: int = None):
        """
        Partitions the dataset ensuring the test customers are different
        from training customers. Training data is strictly before a time cutoff.
        Test data is at or after the time cutoff.
        :param df: DataFrame with features, must include 'cc_num' and 'trans_date_trans_time'.
        :param test_size_customers: Proportion of unique customers for the test set.
        :param time_split_quantile: The quantile of transaction time used to define the cutoff.
        :param random_state: Seed for customer split reproducibility.
        :return: Tuple (train_df, test_df)
        """
        if 'trans_date_trans_time' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
             raise ValueError("DataFrame must contain 'trans_date_trans_time' column as datetime objects.")
        if 'cc_num' not in df.columns:
            raise ValueError("DataFrame must contain 'cc_num' column.")

        # Determine time cutoff
        cutoff_time = df['trans_date_trans_time'].quantile(time_split_quantile)
        print(f"Using time cutoff: {cutoff_time} (Quantile: {time_split_quantile})")

        # Split customers
        unique_customers = df['cc_num'].unique()
        train_customers, test_customers = train_test_split(unique_customers, test_size=test_size_customers, random_state=random_state)
        train_customer_set = set(train_customers)
        test_customer_set = set(test_customers)

        # Output total of train and testing
        print(f"Total unique customers: {len(unique_customers)}")
        print(f"Train customers: {len(train_customer_set)}")
        print(f"Test customers: {len(test_customer_set)}")

        # Filter data based on customer sets
        train_df_all_time = df[df['cc_num'].isin(train_customer_set)]
        test_df_all_time = df[df['cc_num'].isin(test_customer_set)]

        # Apply time filtering
        train_df = train_df_all_time[train_df_all_time['trans_date_trans_time'] < cutoff_time].copy()
        test_df = test_df_all_time[test_df_all_time['trans_date_trans_time'] >= cutoff_time].copy()

        print(f"Initial train records (selected customers, all time): {len(train_df_all_time)}")
        print(f"Final train records (before cutoff {cutoff_time}): {len(train_df)}")
        print(f"Initial test records (selected customers, all time): {len(test_df_all_time)}")
        print(f"Final test records (at or after cutoff {cutoff_time}): {len(test_df)}")

        # Verification
        if not set(train_df['cc_num'].unique()).isdisjoint(set(test_df['cc_num'].unique())):
             print("Warning: Customer overlap detected between train and test sets after filtering!")
        if not train_df.empty and not test_df.empty:
            if train_df['trans_date_trans_time'].max() >= test_df['trans_date_trans_time'].min():
                print(f"Warning: Time overlap detected! Max train time: {train_df['trans_date_trans_time'].max()}, Min test time: {test_df['trans_date_trans_time'].min()}")

        return train_df, test_df

    def create_dataset(self, df: pd.DataFrame, partition_type: str = "train", nonfraud_ratio: int = 5) -> str:
        """
        Removes unused features, creates a smaller dataset by sampling non-fraudulent transactions,
        and saves it as a CSV file.
        :param df: The DataFrame for this partition before removing unused features.
                   Must contain 'is_fraud' and all features needed by the model.
        :param partition_type: The partition type train" or "test". Default is "train".
        :param nonfraud_ratio: The ratio of non-fraudulent records to include per fraudulent record.
                               If None, keeps all records.
        :return: The file path of the saved dataset.
        """
        print(f"\nCreating {partition_type} dataset...")
        # Remove unused features
        df_cleaned = self.remove_features(df.copy()) # Use copy to avoid modifying original df slice

        # Create samplings
        if nonfraud_ratio is not None:
            print(f"Applying non-fraud ratio: {nonfraud_ratio}")
            # Separate fraudulent and non-fraudulent transactions.
            fraud_df = df_cleaned[df_cleaned['is_fraud'] == 1]
            nonfraud_df = df_cleaned[df_cleaned['is_fraud'] == 0]

            fraud_count = len(fraud_df)
            print(f"Fraud records: {fraud_count}")
            print(f"Non-fraud records: {len(nonfraud_df)}")

            if fraud_count == 0 and nonfraud_ratio is not None:
                print("Warning: No fraud records found, cannot apply ratio. Using all non-fraud records.")
                sampled_df = nonfraud_df.copy()
            elif len(nonfraud_df) == 0:
                 print("Warning: No non-fraud records found.")
                 sampled_df = fraud_df.copy()
            else:
                sample_size = min(nonfraud_ratio * fraud_count, len(nonfraud_df))
                print(f"Sampling {sample_size} non-fraud records.")
                random_seed = random.randint(1, 1000) # Use a different seed or manage seeds globally
                sampled_nonfraud = nonfraud_df.sample(n=sample_size, random_state=random_seed)
                # Combine fraud and sampled non-fraud records.
                sampled_df = pd.concat([fraud_df, sampled_nonfraud])
        else:
            print("No non-fraud ratio applied. Using all records.")
            sampled_df = df_cleaned.copy()


        # Shuffle dataset
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Final {partition_type} dataset size: {len(sampled_df)}")
        if 'is_fraud' in sampled_df.columns:
             print(f"Fraud prevalence in final {partition_type} set: {sampled_df['is_fraud'].mean():.4f}")


        # Save datasets
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # ISO-like format
        dataset_dir = os.path.join(self.save_path, "datasets")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        file_name = f"dataset_{partition_type}_{timestamp}.csv"
        file_path = os.path.join(dataset_dir, file_name)
        sampled_df.to_csv(file_path, index=False)
        print(f"{partition_type.capitalize()} dataset saved to: {file_path}")
        return file_path

    def create_train_test_datasets(self, df: pd.DataFrame, test_size_customers: float = 0.2,
                                   time_split_quantile: float = 0.8, nonfraud_ratio: int = 5,
                                   random_state: int = None):
        """
        Partitions the full dataset by customer AND time, removes unused features,
        applies non-fraud sampling, and creates separate CSV files for training and testing.
        :param df: The full DataFrame *before* feature removal but *after* feature engineering.
                   Must include 'cc_num', 'trans_date_trans_time', and 'is_fraud'.
        :param test_size_customers: Proportion of unique customers for the test set.
        :param time_split_quantile: Quantile for the time split cutoff.
        :param nonfraud_ratio: Ratio of non-fraudulent to fraudulent records for sampling. Can be None.
        :param random_state: Seed for reproducibility in customer partitioning.
        :return: Tuple of file paths (train_csv, test_csv)
        """
        print("--- Starting Train/Test Dataset Creation ---")
        # Partition by customer and time using the new method
        train_df, test_df = self.partition(df, test_size_customers=test_size_customers,
                                           time_split_quantile=time_split_quantile,
                                           random_state=random_state)

        # Create final training and testing sets
        train_csv = self.create_dataset(train_df, partition_type="train", nonfraud_ratio=nonfraud_ratio)
        test_csv = self.create_dataset(test_df, partition_type="test", nonfraud_ratio=nonfraud_ratio)

        return train_csv, test_csv

    def describe(self, df: pd.DataFrame):
        """
        Creates a dictionary of quality metrics for the data.
        :param df: DataFrame to describe
        :return: Summary of quality metrics
        """
        description = {
            "Number of records": df.shape[0],
            "Number of columns": df.shape[1],
            "Feature names": df.columns.tolist(),
            "Missing values count": int(df.isnull().sum().sum()),
            "Memory usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            # "Column data types": df.dtypes.astype(str).to_dict()
        }

        if 'is_fraud' in df.columns:
             description["Fraud prevalence"] = f"{df['is_fraud'].mean():.4f}"
        if 'trans_date_trans_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
             description["Min Transaction Time"] = str(df['trans_date_trans_time'].min())
             description["Max Transaction Time"] = str(df['trans_date_trans_time'].max())
        return description


if __name__ == '__main__':
    # Configuration
    STORAGE_PATH = "../../storage"
    SAVE_PATH = "../../storage"
    CUSTOMER_FILE = "customer_release.csv"
    TRANSACTION_FILE = "transactions_release.parquet"
    FRAUD_FILE = "fraud_release.json"
    TEST_CUSTOMER_FRACTION = 0.2 # Proportion of customers for test set
    TIME_SPLIT_QUANTILE = 0.8 # Use 80th percentile of time as cutoff
    NONFRAUD_SAMPLE_RATIO = 5 # Ratio for sampling non-fraudulent transactions (set to None to disable)
    RANDOM_SEED = 42 # For reproducibility

    # Initialize handler
    handler = RawDataHandler(storage_path=STORAGE_PATH, save_path=SAVE_PATH)

    # Extract
    print("Extracting data...")
    customer_info, transaction_info, fraud_info = handler.extract(CUSTOMER_FILE, TRANSACTION_FILE, FRAUD_FILE)
    print("Extraction complete.")

    # Transform (Merge & Basic Clean)
    print("\nTransforming data...")
    merged_data = handler.transform(customer_info, transaction_info, fraud_info)
    print(f"Merged data shape: {merged_data.shape}")
    print("Transformation complete.")

    # Convert Dates
    print("\nConverting dates...")
    merged_data = handler.convert_dates(merged_data)
    print("Date conversion complete.")

    # Save intermediate merged data for inspection
    # merged_data.to_csv(os.path.join(SAVE_PATH, "temp_merged_data.csv"), index=False)

    # Feature Engineering
    print("\nEngineering features...")
    engineered_data = handler.feature_engineer(merged_data)
    print(f"Engineered data shape: {engineered_data.shape}")
    print("Feature engineering complete.")

    # Save engineered data before splitting
    # engineered_data.to_csv(os.path.join(SAVE_PATH, "temp_engineered_data.csv"), index=False)

    # Describe Engineered Data (before splitting/sampling)
    print("\n--- Data Description (Engineered Data) ---")
    engineered_description = handler.describe(engineered_data)
    for key, value in engineered_description.items():
        print(f"{key}: {value}")
    print("-" * 50)


    # Create Train/Test Datasets (Includes Partitioning, Feature Removal, Sampling)
    print("\nCreating Train/Test datasets...")
    train_csv_path, test_csv_path = handler.create_train_test_datasets(
        engineered_data, test_size_customers=TEST_CUSTOMER_FRACTION, time_split_quantile=TIME_SPLIT_QUANTILE,
        nonfraud_ratio=NONFRAUD_SAMPLE_RATIO, random_state=RANDOM_SEED)

    print("\n--- Dataset Creation Summary ---")
    print(f"Training dataset saved to: {train_csv_path}")
    print(f"Testing dataset saved to: {test_csv_path}")
    print("-" * 50)

