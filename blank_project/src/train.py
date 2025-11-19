import data_process
import baseline
import evaluate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import time

def main():
    # 数据准备
    filepath = r'blank_project\data\dft-road-casualty-statistics-accident-2021.csv'
    
    try:
        df = data_process.load_and_preprocess_data(filepath)
        X, y = data_process.get_features_and_target(df)
        X_train, X_test, y_train, y_test = data_process.split_and_scale(X, y)
    except FileNotFoundError:
        print(f"找不到文件 {filepath}")
        return

    # 运行基线模型
    print("\n--- 开始训练基线模型 ---")
    models = baseline.get_baseline_models()
    results = {}

    # 每个模型单独创建进度条
    for name, model in models.items():
        with tqdm(total=1, desc=f"正在训练 {name}", unit="step", ncols=100, leave=True) as pbar:
            start_time = time.time()
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 更新进度条到 100%
            pbar.update(1)
            
            # 预测与评估
            acc = evaluate.evaluate_model(model, X_test, y_test, name)
            results[name] = acc
            
            # 在进度条后面显示耗时和准确率
            elapsed = time.time() - start_time
            pbar.set_postfix(acc=f"{acc:.4f}", time=f"{elapsed:.2f}s")

    # 可视化对比
    print("\n--- 生成对比图表 ---")
    evaluate.plot_model_comparison(results)

    # 模型优化
    print("\n--- 开始逻辑回归超参数调优 ---")
    print("正在进行网格搜索 (Grid Search)...")
    
    hyperparameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'] 
    }

    grid_clf = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, verbose=1, n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_clf.best_params_}")
    print(f"最佳验证集分数: {grid_clf.best_score_:.4f}")

    # 在测试集上验证最佳模型
    best_model = grid_clf.best_estimator_
    # 打印结果
    final_acc = evaluate.evaluate_model(best_model, X_test, y_test, "Optimized Logistic Regression")

if __name__ == "__main__":
    main()