# data/download_data.py
import os
import pandas as pd

# Permanent OpenML source for French Motor Insurance Claims dataset (freMTPL2freq)
DATA_URL = "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.csv"
OUTPUT_PATH = os.path.join("data", "raw", "freMTPL2freq.csv")


def fetch_raw_data():
    print(f"[*] Extracting raw data from:\n    {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

        print(f"[+] Saved locally to: {OUTPUT_PATH}")
        print(f"[+] Record Count: {len(df)} rows x {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"[-] Download failed: {e}")


if __name__ == "__main__":
    fetch_raw_data()
    