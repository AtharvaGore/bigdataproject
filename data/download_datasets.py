"""
Dataset Downloader for Distributed Bias Mitigation
Downloads and prepares:
1. UCI Adult Census Income
2. COMPAS Recidivism (ProPublica)
3. Folktables (US Census ACS PUMS)
"""

import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_uci_adult():
    print("[1/3] Downloading UCI Adult dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    raw_path = os.path.join(DATA_DIR, "adult.raw.csv")
    out_path = os.path.join(DATA_DIR, "adult.csv")
    
    if os.path.exists(out_path):
        print(f"  UCI Adult already exists at {out_path}")
        return out_path
        
    urllib.request.urlretrieve(url, raw_path)
    
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(raw_path, names=columns, skipinitialspace=True)
    
    # Clean whitespace and define binary label & protected attribute
    # Favorable outcome: income > 50K (1), <= 50K (0)
    # Privileged: Male (1), Unprivileged: Female (0)
    df["label"] = (df["income"].str.contains(">50K")).astype(int)
    df["protected"] = (df["sex"] == "Male").astype(int)
    
    df.to_csv(out_path, index=False)
    if os.path.exists(raw_path):
        os.remove(raw_path)
    print(f"  Saved UCI Adult: {len(df)} rows to {out_path}")
    return out_path

def download_compas():
    print("[2/3] Downloading COMPAS (ProPublica) dataset...")
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    raw_path = os.path.join(DATA_DIR, "compas.raw.csv")
    out_path = os.path.join(DATA_DIR, "compas.csv")
    
    if os.path.exists(out_path):
        print(f"  COMPAS already exists at {out_path}")
        return out_path
        
    urllib.request.urlretrieve(url, raw_path)
    df = pd.read_csv(raw_path)
    
    # Standard ProPublica filtering criteria:
    # 1. days_b_screening_arrest between -30 and 30
    # 2. is_recid != -1
    # 3. c_charge_degree != 'O'
    # 4. race is African-American or Caucasian
    df_filtered = df[
        (df["days_b_screening_arrest"] >= -30) &
        (df["days_b_screening_arrest"] <= 30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["race"].isin(["African-American", "Caucasian"]))
    ].copy()
    
    # Standard fairness setup:
    # Favorable outcome: Non-recidivism (two_year_recid == 0 -> 1, recidivism == 1 -> 0)
    # Privileged group: Caucasian (1), Unprivileged: African-American (0)
    df_filtered["label"] = (df_filtered["two_year_recid"] == 0).astype(int)
    df_filtered["protected"] = (df_filtered["race"] == "Caucasian").astype(int)
    
    keep_cols = [
        "age", "c_charge_degree", "race", "sex", "priors_count",
        "decile_score", "two_year_recid", "label", "protected"
    ]
    df_clean = df_filtered[keep_cols]
    df_clean.to_csv(out_path, index=False)
    if os.path.exists(raw_path):
        os.remove(raw_path)
    print(f"  Saved COMPAS: {len(df_clean)} rows to {out_path}")
    return out_path

def download_folktables():
    print("[3/3] Downloading Folktables (US ACS PUMS - California & Texas)...")
    out_path = os.path.join(DATA_DIR, "folktables_acs.csv")
    if os.path.exists(out_path):
        print(f"  Folktables ACS already exists at {out_path}")
        return out_path
        
    from folktables import ACSDataSource, ACSIncome
    
    # We download 2018 1-Year ACS for CA (and TX if needed) to get hundreds of thousands of rows
    # Each state provides real census records
    states = ["CA", "TX"]
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=DATA_DIR)
    
    dfs = []
    for state in states:
        print(f"  Fetching ACS data for {state}...")
        try:
            acs_data = data_source.get_data(states=[state], download=True)
            features, label, group = ACSIncome.df_to_pandas(acs_data)
            features["label"] = label.astype(int)
            # ACSIncome default group is RAC1P (White = 1 vs Non-White = 0)
            # Or SEX (1: Male, 2: Female). Let's use White (RAC1P == 1) as privileged group
            features["protected"] = (group == 1).astype(int)
            dfs.append(features)
            print(f"    Loaded {len(features)} rows from {state}")
        except Exception as e:
            print(f"    Error downloading {state}: {e}")
            
    if not dfs:
        raise RuntimeError("Failed to download any Folktables state data.")
        
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  Saved Folktables: {len(combined)} rows to {out_path}")
    return out_path

if __name__ == "__main__":
    download_uci_adult()
    download_compas()
    download_folktables()
    print("\nAll datasets successfully prepared in data/ folder!")
