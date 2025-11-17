import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train import LSTMModel

def evaluate():
    # 用户后续会把 data/processed/sequence_test.csv 放好
    df = pd.read_csv("data/processed/sequence_test.csv")

    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = df.iloc[:, -1].values

    model = LSTMModel(input_dim=X.shape[1])
    model.load_state_dict(torch.load("lstm_model.pth"))
    model.eval()

    with torch.no_grad():
        pred = model(X).squeeze().numpy()

    print("MAE:", mean_absolute_error(y, pred))
    print("MSE:", mean_squared_error(y, pred))

if __name__ == "__main__":
    evaluate()
