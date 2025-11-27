from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def get_advanced_models():
return {
"Gradient Boosting": GradientBoostingClassifier(),
"XGBoost": XGBClassifier(),
}
