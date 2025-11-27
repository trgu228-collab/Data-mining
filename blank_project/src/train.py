from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.baseline import get_basic_models
from src.advanced import get_advanced_models

def split_and_scale(X, y):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
return X_train_scaled, X_test_scaled, y_train, y_test

def train_all_models(X_train, y_train):
models = get_basic_models()
models.update(get_advanced_models())
trained = {}
for name, model in models.items():
model.fit(X_train, y_train)
trained[name] = model
return trained
