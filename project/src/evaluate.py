'''
负责打分和画图，保持代码整洁
'''

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    #预测并返回准确率
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} 准确率: {acc:.4f}")
    return acc

def plot_model_comparison(results):
    #绘制模型对比图
    scores = pd.Series(results)
    plt.figure(figsize=(15, 9))
    ax = scores.sort_values().plot(kind='barh', color=['black', 'gray', 'green', 'brown', 'pink', 'blue'])
    
    plt.title('Comparison of Models (Accuracy) Unseen Data', fontsize=15)
    plt.xlabel('Accuracy', fontsize=15)
    plt.ylabel('Models', fontsize=15)
    plt.xlim(0, 1.0)
    
    # 添加数值标签
    for i, v in enumerate(scores.sort_values()):
        ax.text(v + 0.01, i, '{:.2f}%'.format(100 * v), color='black', fontweight='bold')
    
    plt.show()
    print("对比图已生成！")