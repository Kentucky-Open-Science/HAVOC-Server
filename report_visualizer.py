import os
import json
import base64
import io
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Use a non-GUI backend for Matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === CONFIGURATION ===
SENSOR_CSV_PATH = "Temi_Sensor_Data/sensor_data_master.csv"
TRAINING_DATA_PATH = "static/newSensor_training.csv"  # Path to the data for the KNN classifier
VISUALS_DIR = "visualizations"


# === HELPER FUNCTIONS ===
def ensure_dirs():
    os.makedirs(VISUALS_DIR, exist_ok=True)


def load_and_filter_daily_data(path, date):
    """Loads the master CSV and returns only the rows for the specified date."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df = df[df["timestamp"].dt.date == date].copy()
    return df


def process_sensor_data(df):
    """
    Processes raw sensor data into the 17-feature format used for classification.
    This mimics the logic from SmellClassifier and server.py for consistent processing.
    """
    if df.empty:
        return pd.DataFrame()

    # Keep only the value columns
    value_cols = [col for col in df.columns if col.startswith("value_")]
    if len(value_cols) != 66:
        print(f"Warning: Expected 66 'value_*' columns, but found {len(value_cols)}. Skipping processing.")
        return pd.DataFrame()

    data = df[value_cols]

    # Averaging logic based on the channel map concept
    # These are the indices of the 15 sensor groups (3 sensors per group) + 2 for temp/humidity
    indices_to_keep = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,  # SGP/NOx sensors
        21, 22, 23, 24, 25, 26, 27, 28, 29,  # MICS sensors part 1
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,  # MICS sensors part 2
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  # MICS sensors part 3
        64, 65  # Temp/Humidity
    ]

    raw_values = data.iloc[:, indices_to_keep].values

    # Average in groups of 3 for the first 15 sensors
    num_sensor_groups = 15
    averaged_values = np.zeros((raw_values.shape[0], num_sensor_groups + 2))
    for i in range(num_sensor_groups):
        start_col = i * 3
        end_col = start_col + 3
        averaged_values[:, i] = raw_values[:, start_col:end_col].mean(axis=1)

    # Add temperature and humidity
    averaged_values[:, num_sensor_groups] = raw_values[:, -2]
    averaged_values[:, num_sensor_groups + 1] = raw_values[:, -1]

    # Create a new DataFrame with meaningful names
    feature_names = [f'sensor_avg_{i + 1}' for i in range(15)] + ['temperature', 'humidity']
    processed_df = pd.DataFrame(averaged_values, columns=feature_names)
    return processed_df


# === MAIN VISUALIZATION PIPELINE ===
def generate_tsne_visualization(save_file=False):
    """
    Generates a t-SNE plot comparing today's sensor data with the KNN training data.
    """
    ensure_dirs()
    today = datetime.now().date()
    date_str = today.isoformat()

    # 1. Load and process today's data
    daily_df_raw = load_and_filter_daily_data(SENSOR_CSV_PATH, today)
    daily_df_processed = process_sensor_data(daily_df_raw)

    # 2. Load and process training data
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"Error: Training data not found at {TRAINING_DATA_PATH}")
        return None

    training_df_raw = pd.read_csv(TRAINING_DATA_PATH)
    training_labels = training_df_raw['class']
    training_df_processed = process_sensor_data(training_df_raw)

    # 3. Combine data for t-SNE
    if daily_df_processed.empty:
        print("No daily data to plot.")
        combined_data = training_df_processed
        labels = training_labels.tolist()
    else:
        # Standardize features before combining
        scaler = StandardScaler()
        training_scaled = scaler.fit_transform(training_df_processed)
        daily_scaled = scaler.transform(daily_df_processed)

        combined_data = np.vstack([daily_scaled, training_scaled])
        labels = ['Daily Reading'] * len(daily_df_processed) + training_labels.tolist()

    # 4. Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data) - 1))
    tsne_result = tsne.fit_transform(combined_data)

    # 5. Plot the results
    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label in enumerate(unique_labels):
        ix = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(
            tsne_result[ix, 0],
            tsne_result[ix, 1],
            label=label,
            alpha=0.8,
            color=colors(i),
            s=50 if label != 'Daily Reading' else 25  # Make training points bigger
        )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f"t-SNE Visualization: Daily Readings vs. Training Data ({date_str})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend

    # 6. Prepare output
    img_path = os.path.join(VISUALS_DIR, f"tsne_{date_str}.png")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    if save_file:
        plt.savefig(img_path, bbox_inches='tight')
        print(f"Saved t-SNE plot to {img_path}")

    plt.close()

    metadata = {
        "date": date_str,
        "daily_readings_count": len(daily_df_processed),
        "training_samples_count": len(training_df_processed),
        "plot_generated": True
    }

    return {
        "tsne_image_base64": img_base64,
        "metadata": metadata
    }


if __name__ == '__main__':
    print("Running t-SNE visualization generation...")
    # This will generate the visualization but not save the file by default
    result = generate_tsne_visualization(save_file=True)
    if result:
        print("Successfully generated visualization.")
        print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")