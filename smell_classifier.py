import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class SmellClassifier:
    def __init__(self, dataset_path='Temi_Sensor_Data/newSensor_training.csv'):
        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        self.dataset = self.process_dataframe(dataset)
        # Separate features and target
        X = self.dataset.drop('class', axis=1)
        y = self.dataset['class']
        # Initialize and train the KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X, y)
        # Store the feature names used during training
        self.feature_names = X.columns.tolist()

    channel_map = [999, 999, 11, 11, 11, 3, 3, 3, 2, 2, 2, 1, 1, 1, 999, 999,
                   999, 999, 999, 999, 999, 9, 9, 9, 6, 6, 6, 5, 5, 5, 999, 999,
                   999, 999, 14, 14, 14, 10, 10, 10, 7, 7, 7, 4, 4, 4, 999, 999,
                   999, 999, 15, 15, 15, 13, 13, 13, 12, 12, 12, 8, 8, 8, 999, 999,
                   16, 17, 0]

    def process_dataframe(self, df):
        # Drop channels that are not active
        drop_list = []
        for idx, x in enumerate(self.channel_map):
            if x == 999 and idx < len(df.columns):  # Ensure index is valid
                colname = df.columns[idx]
                drop_list.append(colname)

        df = df.drop(columns=drop_list, errors='ignore')  # Ignore missing columns

        num_cols = df.shape[1]
        # The last three columns (temp, humidity, and class) remain unchanged
        num_groups = (num_cols - 3) // 3 if 'class' in df.columns else (num_cols - 2) // 3
        averaged_data = []
        col_names = []

        for i in range(num_groups):
            cols = df.iloc[:, i * 3:(i + 1) * 3]
            averaged_col = cols.mean(axis=1)
            averaged_data.append(averaged_col)
            col_names.append(f'channel_{i + 1}_avg')

        # Add the last columns unchanged
        if 'temperature' in df.columns:
            averaged_data.append(df['temperature'])
            col_names.append('temperature')
        if 'humidity' in df.columns:
            averaged_data.append(df['humidity'])
            col_names.append('humidity')
        if 'class' in df.columns:
            averaged_data.append(df['class'])
            col_names.append('class')

        # Combine into a new DataFrame
        result_df = pd.concat(averaged_data, axis=1)
        result_df.columns = col_names

        return result_df

    def classify_sensor_data(self, formatted_values):
        # Validate input length
        if len(formatted_values) != 17:
            raise ValueError(f"Expected 17 sensor values, got {len(formatted_values)}")

        new_data_df = pd.DataFrame([formatted_values], columns=self.feature_names)
        # Reorder columns to match training data
        new_data_df = new_data_df[self.feature_names]

        # Predict using the KNN model
        # prediction = self.knn.predict(new_data_df)[0]
        prediction = 'ambient'   # Hardcoded ambient smell for now (Demo)
        return prediction