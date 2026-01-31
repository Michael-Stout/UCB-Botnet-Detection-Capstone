"""Data loading, cleaning, and feature engineering for botnet detection."""

import math
from collections import Counter

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS, COLS_TO_DROP, DUR_BINS, DUR_LABELS,
    IP_COLS, MAIN_FEATURES, NUMERIC_FEATURES,
)


def safe_divide(num, denom):
    """Divide num by denom, returning 0 when denom is zero."""
    return num / denom if denom != 0 else 0


def address_entropy(addr):
    """Compute Shannon entropy of the character distribution of an IP address string."""
    addr_str = str(addr)
    c_count = Counter(addr_str)
    total_chars = len(addr_str)
    if total_chars < 2:
        return 0
    entropy = 0.0
    for char_count in c_count.values():
        p = char_count / total_chars
        entropy -= p * math.log2(p)
    return entropy


def get_port_range(x):
    """Bucket a port number into WellKnown, Registered, Ephemeral, or Unknown."""
    try:
        port = int(x)
    except (ValueError, TypeError):
        return 'Unknown'
    if 0 <= port <= 1023:
        return 'WellKnown'
    elif 1024 <= port <= 49151:
        return 'Registered'
    elif 49152 <= port <= 65535:
        return 'Ephemeral'
    return 'Unknown'


def load_raw_data(file_path):
    """
    Load a CTU-13 CSV file, filter out background flows, and create the
    binary Botnet target column. Returns the DataFrame.
    """
    df = pd.read_csv(file_path)

    if 'Label' in df.columns:
        df = df[~df['Label'].str.contains('Background', case=False, na=False)].copy()
        df['Botnet'] = df['Label'].apply(lambda x: 1 if 'botnet' in str(x).lower() else 0)
        df.drop(columns=['Label'], inplace=True)

    return df


def engineer_features(df):
    """
    Add derived features to the DataFrame:
    BytesPerSecond, PktsPerSecond, SrcAddrEntropy, DstAddrEntropy,
    SportRange, DportRange, DurCategory, BytePktRatio.

    Drops columns listed in COLS_TO_DROP.
    Returns the modified DataFrame.
    """
    df = df.copy()

    # Drop unnecessary columns
    for col in COLS_TO_DROP:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Rate features
    if 'Dur' in df.columns and 'TotBytes' in df.columns:
        df['BytesPerSecond'] = df.apply(lambda r: safe_divide(r['TotBytes'], r['Dur']), axis=1)
        df['PktsPerSecond'] = df.apply(lambda r: safe_divide(r['TotPkts'], r['Dur']), axis=1)

    # Entropy features
    if 'SrcAddr' in df.columns:
        df['SrcAddrEntropy'] = df['SrcAddr'].apply(address_entropy)
    if 'DstAddr' in df.columns:
        df['DstAddrEntropy'] = df['DstAddr'].apply(address_entropy)

    # Port range bucketing
    if 'Sport' in df.columns:
        df['SportRange'] = df['Sport'].apply(get_port_range)
        df.drop(columns=['Sport'], inplace=True)
    if 'Dport' in df.columns:
        df['DportRange'] = df['Dport'].apply(get_port_range)
        df.drop(columns=['Dport'], inplace=True)

    # Duration binning
    if 'Dur' in df.columns:
        df['DurCategory'] = pd.cut(df['Dur'], bins=DUR_BINS, labels=DUR_LABELS, include_lowest=True)

    # Byte-to-packet ratio
    if 'TotBytes' in df.columns and 'TotPkts' in df.columns:
        df['BytePktRatio'] = df.apply(lambda r: safe_divide(r['TotBytes'], r['TotPkts']), axis=1)

    return df


def prepare_xy(df):
    """
    Select main features, separate into X and y, drop IP columns,
    and encode categoricals as numeric codes.

    Returns (X, y) as DataFrames.
    """
    available = [col for col in MAIN_FEATURES if col in df.columns]
    df = df[available].copy()

    if 'Botnet' not in df.columns:
        raise ValueError("No 'Botnet' column found in DataFrame.")

    y = df['Botnet']
    X = df.drop('Botnet', axis=1)

    # Drop IP columns
    for col in IP_COLS:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category').cat.codes
            elif str(X[col].dtype).startswith('category'):
                X[col] = X[col].cat.codes

    return X, y


def load_and_prepare_scenario(file_path):
    """
    Load a CTU-13 CSV and prepare numeric features for cross-scenario evaluation.
    Returns (X, y) with only NUMERIC_FEATURES columns.
    """
    df = load_raw_data(file_path)

    # Derive numeric features
    if 'Dur' in df.columns and 'TotBytes' in df.columns:
        df['BytesPerSecond'] = df.apply(lambda r: safe_divide(r['TotBytes'], r['Dur']), axis=1)
        df['PktsPerSecond'] = df.apply(lambda r: safe_divide(r['TotPkts'], r['Dur']), axis=1)
        df['BytePktRatio'] = df.apply(lambda r: safe_divide(r['TotBytes'], r['TotPkts']), axis=1)

    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[available].copy().fillna(0)
    y = df['Botnet']
    return X, y
