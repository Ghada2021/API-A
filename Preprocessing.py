import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# Read dataset
df= pd.read_csv("dataset.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop nulls
df.drop(10642, axis=0, inplace=True)
df.reset_index(drop=True,inplace=True)

# Encoding
encoder = LabelEncoder()
df['authentification methode'] = encoder.fit_transform(df['authentification methode'])
df['authentification methode'] = df['authentification methode'].astype(str)
df['authentification methode'] = df['authentification methode'].str.strip()
df['authentification methode'] = df['authentification methode'].replace('3','nan')

# Imputing
column_index = 12
column_to_impute = df.iloc[:, column_index]
imputer = KNNImputer(n_neighbors=5)
column_imputed = imputer.fit_transform(column_to_impute.values.reshape(-1, 1))
df.iloc[:, column_index] = column_imputed.flatten()
df['authentification methode']=df['authentification methode'].astype(int)

# Encoding
df['source'] = df.apply(lambda row: 0 if 'E' in row['source'] else 1, axis=1)
encoder = LabelEncoder()
df['statusCode'] = encoder.fit_transform((df['statusCode']).astype(str))
df['risk'] = df.apply(lambda row: 0 if 'low' in row['risk'] else (1 if 'moderate' in row['risk'] else 2), axis=1)

df['num_unique_apis']= df['num_unique_apis'].astype(int)

# Normalization
scaler = RobustScaler()
df['inter_api_access_duration(sec)'] = scaler.fit_transform(df[['inter_api_access_duration(sec)']])
scaler = RobustScaler()
df['sequence_length(count)'] = scaler.fit_transform(df[['sequence_length(count)']])
scaler = RobustScaler()
df['vsession_duration(min)'] = scaler.fit_transform(df[['vsession_duration(min)']])

# Drop useless variables
df.drop(['_id','api_access_uniqueness','num_sessions','num_users','ip_type'],axis=1,inplace=True)



# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')


# Save csv
train.to_csv(train_path, index=False)
