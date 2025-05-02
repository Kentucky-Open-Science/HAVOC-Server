import serial
import csv
import os
import time
import logging
from datetime import datetime

# === Configuration ===
DEV_PATH = 'COM3'            # Adjust this to your actual serial port
BAUD_RATE = 115200
SENSOR_CSV_DIR = "Local_Sensor_Data"
CSV_FILENAME = "sensor_data_local.csv"
LOG_FILENAME = "sensor_log.txt"
EXPECTED_LENGTH = 66         # Number of expected sensor values

# === Setup Logging ===
os.makedirs(SENSOR_CSV_DIR, exist_ok=True)
log_path = os.path.join(SENSOR_CSV_DIR, LOG_FILENAME)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

# === Initialize Serial Connection ===
try:
    ser = serial.Serial(DEV_PATH, BAUD_RATE, timeout=1)
    logging.info(f"Serial connection opened on {DEV_PATH} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    logging.error(f"Error opening serial port: {e}")
    exit(1)

csv_path = os.path.join(SENSOR_CSV_DIR, CSV_FILENAME)

# === Main Recording Loop ===
def record_loop():
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['timestamp'] + [f'value_{i}' for i in range(EXPECTED_LENGTH)])
            logging.info("Wrote CSV header.")

        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                logging.debug(f"Raw line: {line}")

                parts = line.split(';')

                # Remove the "start" prefix if present
                if parts[0].lower() == "start":
                    parts = parts[1:]

                if len(parts) != EXPECTED_LENGTH:
                    logging.warning(f"Invalid data length: {len(parts)} (expected {EXPECTED_LENGTH})")
                    continue

                try:
                    values = [float(val) for val in parts]
                except ValueError:
                    logging.warning("Non-numeric data encountered. Skipping line.")
                    continue

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp] + values)
                csvfile.flush()
                logging.info(f"Saved reading at {timestamp}")
                time.sleep(1.8)

        except KeyboardInterrupt:
            logging.info("Stopped by user (KeyboardInterrupt).")
        except Exception as e:
            logging.exception(f"Unexpected error during recording: {e}")
        finally:
            ser.close()
            logging.info("Serial connection closed.")

if __name__ == "__main__":
    record_loop()
