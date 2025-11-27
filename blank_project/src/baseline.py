from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_basic_models():
return {
"Logistic Regression": LogisticRegression(),
"Decision Tree": DecisionTreeClassifier(),
"Random Forest": RandomForestClassifier(),
"KNN": KNeighborsClassifier(),
}

    joblib.dump(model, "baseline_model.pkl")

if __name__ == "__main__":
    train_baseline()
