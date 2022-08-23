# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_data_exploratory.ipynb.

# %% auto 0
__all__ = ['load_data', 'data_summary', 'catego_unique_values', 'check_null_values', 'correlation_view']

# %% ../00_data_exploratory.ipynb 2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from nbdev.showdoc import *

# %% ../00_data_exploratory.ipynb 4
def load_data(path) -> pd.DataFrame:
    """Load a dataframe."""
    if isinstance(path, (str, Path)):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("DataFrame Is empty")
    else:
        raise ValueError("Check your data path")
    return df

# %% ../00_data_exploratory.ipynb 10
def data_summary(df) -> pd.DataFrame:
    """Summarize data"""
    print(df.info())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.describe())

# %% ../00_data_exploratory.ipynb 18
def catego_unique_values(df):
    """Visualize unique values."""
    # exclude date and title columns
    no_columns = ["StartDate","EndDate","CampaignTitle"]
    # select only catego variables
    for col in df.select_dtypes(["object","string"]):
        if col not in no_columns:
            print(f'{df[col].nunique()} unique values for {col:-<20}  {df[col].unique()}')
            matplotlib.rcParams['figure.figsize'] = (12,6)
            sns.displot(df[col])
            plt.show()

# %% ../00_data_exploratory.ipynb 29
def check_null_values(df)-> None:
    """check null values for each column."""
    df_nulls = pd.DataFrame(df.isnull().sum(),columns=["Empty values"])
    print(df_nulls)
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# %% ../00_data_exploratory.ipynb 33
def correlation_view(df):
    """visualize correlation."""
    correlation = df.corr()
    sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True)
    sns.pairplot(df)
    plt.show()
        
