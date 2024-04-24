#%%
import pandas as pd 
import numpy as np
from utils import CleanData

df = pd.read_excel('./data/full_data_set.xlsx')
len(df.columns)
#df.drop(['Name', 'Hometown', 'State', 'High School', 'NFL Draft Pick'], inplace=True, axis=1)
# %% drop cols with less than 10 values
len(df.columns)
#%%
cols_dropped = [col for col in df.columns if df[col].count() < 99]
df.drop(columns=cols_dropped, inplace=True)
len(cols_dropped)
#%%
#named_cols_to_drop = [col for col in df.columns if 'Avg' in col]
named_cols_to_drop = [col for col in df.columns if any(sub in col for sub in ['Avg', '100+', 'Y/G'])]

# Drop these columns
df.drop(columns=named_cols_to_drop, inplace=True)

#%%
len(named_cols_to_drop)
# %%
len(df.columns)
#%%
df.shape
#%%
len(df)*0.2
#%%
df.head()
# %%
df1 = df.dropna(thresh = 35)
#%%
df1.shape
#%%
df1.isnull().sum()
#%%
cleaner = CleanData()
df1 = cleaner.convert_draft_pick(df1)
#%%
df1.head()
df1["NFL Draft Pick"].value_counts()
#%%
df1.isnull().sum()
# %%
df1.to_excel('./data/cleaned_data_77.xlsx', index=False)
# %%
feature_cols = [col for col in df1.columns if col not in ['NFL Draft Pick', 'Name', 'Hometown', 'State', 'High School']]

df1[feature_cols].corr().to_excel('./data/correlation_matrix.xlsx')
# %%
df1[feature_cols].corr()
# %%
