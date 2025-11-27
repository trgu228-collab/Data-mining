import pandas as pd

def load_and_clean_data(filepath):
df = pd.read_csv(filepath, low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week

df = df.dropna(subset=['accident_severity'])
df = df.drop(columns=['location_easting_osgr', 'location_northing_osgr', 'date'])

df = df[df['accident_severity'].isin([1, 2, 3])]
df['accident_severity'] = df['accident_severity'].astype(int)

return df

def select_features_and_target(df):
X = df.drop(columns=['accident_severity'])
y = df['accident_severity']
return X, y
