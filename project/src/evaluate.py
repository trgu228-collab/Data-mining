from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
scores = {}
for name, model in models.items():
y_pred = model.predict(X_test)
scores[name] = accuracy_score(y_test, y_pred)
return scores

def plot_scores(scores):
plt.figure(figsize=(10, 6))
names = list(scores.keys())
accs = list(scores.values())
plt.barh(names, accs, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.savefig("reports/accuracy_comparison.png")
plt.show()
