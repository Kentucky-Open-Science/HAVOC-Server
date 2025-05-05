import os
import json
import threading
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from daily_reports import increment, metrics

# === CONFIGURATION ===
load_dotenv()
SENSOR_CSV_PATH = "Temi_Sensor_Data/sensor_data_master.csv"
TARGET_CSV_PATH = "Temi_Sensor_Data/target_smell.csv"
MODEL_PATH = "models/autoencoder_latest.pt"
EMBEDDINGS_DIR = "embeddings"
VISUALS_DIR = "visualizations"
MASTER_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "master_embeddings.npy")
REPORT_TIME = os.getenv("TEMI_REPORT_TIME", "20:00")

# === HELPER FUNCTIONS ===
def ensure_dirs():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

def load_csv_subset(path, date):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df = df[(df["timestamp"].dt.date == date) &
            (df["timestamp"].dt.hour >= 8) & (df["timestamp"].dt.hour < 17)]
    df.dropna(inplace=True)
    return df

def clean_data(df):
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.loc[:, df.columns != "timestamp"]

    value_cols = [col for col in df.columns if col.startswith("value_")]
    df = df[value_cols]

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    return df

class Autoencoder(nn.Module):
    def __init__(self, input_dim=66, embedding_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# === MAIN PIPELINE ===
def run_embedding_pipeline(test_mode=False, skip_save=False):
    ensure_dirs()
    today = datetime.now().date()

    target_df = clean_data(pd.read_csv(TARGET_CSV_PATH))
    full_df = clean_data(pd.read_csv(SENSOR_CSV_PATH))

    input_dim = full_df.shape[1]
    embedding_dim = 10
    model = Autoencoder(input_dim=input_dim, embedding_dim=embedding_dim)
    scaler = StandardScaler()

    if os.path.exists(MODEL_PATH) and not skip_save:
        today_df = clean_data(load_csv_subset(SENSOR_CSV_PATH, today))
        if today_df.empty:
            print("No ambient data for today. Skipping fine-tuning.")
            return
        ambient_scaled = scaler.fit_transform(today_df)
        target_scaled = scaler.transform(target_df)
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded previous model, continuing fine-tuning.")
        num_epochs = 15
    else:
        ambient_scaled = scaler.fit_transform(full_df)
        target_scaled = scaler.transform(target_df)
        print("Training on full dataset.")
        num_epochs = 75

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(ambient_scaled, dtype=torch.float32)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()

    # Skip saving if flag is set
    if not skip_save:
        torch.save(model.state_dict(), MODEL_PATH)

    model.eval()
    with torch.no_grad():
        ambient_embeds = model.encoder(X_tensor).numpy()
        target_embeds = model.encoder(torch.tensor(target_scaled, dtype=torch.float32)).numpy()

    today_str = datetime.now().date().isoformat()
    if not skip_save:
        np.save(os.path.join(EMBEDDINGS_DIR, f"{today_str}.npy"), ambient_embeds)
        if os.path.exists(MASTER_EMBEDDINGS_PATH):
            master_embeds = np.load(MASTER_EMBEDDINGS_PATH)
            master_embeds = np.concatenate([master_embeds, ambient_embeds], axis=0)
        else:
            master_embeds = ambient_embeds
        np.save(MASTER_EMBEDDINGS_PATH, master_embeds)

    tsne = TSNE(n_components=2, random_state=42)
    combined_embeds = np.vstack([ambient_embeds, target_embeds])
    tsne_result = tsne.fit_transform(combined_embeds)
    labels = ["ambient"] * len(ambient_embeds) + ["target"] * len(target_embeds)

    plt.figure(figsize=(10, 6))
    for label in set(labels):
        ix = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne_result[ix, 0], tsne_result[ix, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title(f"t-SNE Visualization: {today_str}")

    if test_mode:
        print("\n--- TEST MODE OUTPUT ---")
        print(f"Ambient Rows: {len(ambient_embeds)}")
        print(f"Target Rows: {len(target_embeds)}")
        print(f"Silhouette Score: {silhouette_score(combined_embeds, labels):.4f}")
        print(f"Loss: {float(loss.item()):.6f}")
        plt.show()
    else:
        if not skip_save:
            plt.savefig(os.path.join(VISUALS_DIR, f"tsne_{today_str}.png"))
        plt.close()

    silhouette = silhouette_score(combined_embeds, labels)
    if not skip_save:
        metadata = {
            "date": str(today_str),
            "ambient_rows": int(len(ambient_embeds)),
            "target_rows": int(len(target_embeds)),
            "silhouette_score": float(silhouette),  # force native float
            "loss": float(loss.item())              # also force native float
        }

        with open(os.path.join(EMBEDDINGS_DIR, f"{today_str}.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
    # REPORT: increment every training round
    increment("training_rounds_today")

    print(f"Training complete for {today_str}. Silhouette: {silhouette:.4f}")

# === SCHEDULER ENTRYPOINT ===
def schedule_embedding_pipeline():
    now = datetime.now()
    hour, minute = map(int, REPORT_TIME.split(":"))
    run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if run_time < now:
        run_time += timedelta(days=1)

    delay = (run_time - now).total_seconds() - 300
    delay = max(0, delay)

    def run_and_log():
        print(f"⏱ Running embedding pipeline {round(delay / 60)} minutes before report...")
        run_embedding_pipeline()

    threading.Timer(delay, run_and_log).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run pipeline in test mode with local output")
    parser.add_argument("--skip-save", action="store_true", help="Run pipeline without saving model, embeddings, or files")
    args = parser.parse_args()

    run_embedding_pipeline(test_mode=args.test_mode, skip_save=args.skip_save)
