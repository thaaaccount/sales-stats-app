import pickle
import pandas as pd
from pygam import GAM, s
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Mexico_Input.csv')
df = df[df['UnitPrice'] >= 0]
df = df[df['UnitPrice'] != 315.93]
    
df['UnitPrice'] = df['UnitPrice'].astype(float)
df['Quantity'] = df['Quantity'].astype(float)
df['UnitCost'] = df['UnitCost'].astype(float)
df['Revenue'] = df['UnitPrice'] * df['Quantity']
df['OrderDate_Year'] = df['OrderDate_Year'].astype(str)

label_encoder = LabelEncoder()
df['LongItem_encoded'] = label_encoder.fit_transform(df['LongItem'])

df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['UnitCost'] = pd.to_numeric(df['UnitCost'], errors='coerce')

df = df.dropna(subset=['UnitPrice', 'Quantity'])

import pickle
import pandas as pd
from pygam import GAM, s

# Assume df is your DataFrame
df_agg = df.groupby(['LongItem', 'UnitPrice']).agg({
    'Quantity': 'sum',
    'Revenue': 'sum'
}).reset_index()

# Train and save the models
for item in df_agg['LongItem'].unique():
    subset = df_agg[df_agg['LongItem'] == item]
    X = subset[['UnitPrice']].to_numpy()  # Ensure X is a dense numpy array
    y = subset['Revenue']
    gam = GAM(s(0)).fit(X, y)
    with open(f'{item}_gam_model.pkl', 'wb') as f:
        pickle.dump(gam, f)
