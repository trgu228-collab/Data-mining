from src.data_utils import load_and_clean_data, select_features_and_target
from src.train import split_and_scale, train_all_models
from src.evaluate import evaluate_models, plot_scores

def main():
df = load_and_clean_data("data/dft-road-casualty-statistics-accident-2021.csv")
X, y = select_features_and_target(df)
X_train, X_test, y_train, y_test = split_and_scale(X, y)
models = train_all_models(X_train, y_train)
scores = evaluate_models(models, X_test, y_test)
plot_scores(scores)

if __name__ == '__main__':
main()
