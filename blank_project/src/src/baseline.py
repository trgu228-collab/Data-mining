import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    # 用户之后会把数据放到 data/processed/train.csv
    return pd.read_csv("data/processed/train.csv")

def train_baseline():
    df = load_data()

    X = df.drop("queue_length", axis=1)
    y = df["queue_length"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("Baseline MSE:", mse)

    joblib.dump(model, "baseline_model.pkl")

if __name__ == "__main__":
    train_baseline()
