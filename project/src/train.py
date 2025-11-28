import data_process
import baseline
import evaluate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import time

def main():
    # 1. 数据准备流程
    filepath = r'blank_project\data\dft-road-casualty-statistics-accident-2021.csv'
    
    try:
        df = data_process.load_and_preprocess_data(filepath)
        X, y = data_process.get_features_and_target(df)
        X_train, X_test, y_train, y_test = data_process.split_and_scale(X, y)
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}，请检查路径！")
        return

    # 2. 运行基线模型
    print("\n" + "="*30)
    print("开始训练基线模型")
    print("="*30)
    
    models = baseline.get_baseline_models()
    results = {}

    for name, model in models.items():
        # [关键修改] 跳过 KNN 模型，因为它在大数据集上太慢了
        if name == 'KNN':
            print(f"⏩ 跳过模型: {name} (计算量过大)")
            continue

        print(f"\n正在训练模型: {name} ...")
        start_time = time.time()
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测与评估
        # evaluate_model 函数内部会打印准确率
        acc = evaluate.evaluate_model(model, X_test, y_test, name)
        results[name] = acc
        
        elapsed = time.time() - start_time
        print(f"✅ {name} 完成！耗时: {elapsed:.2f} 秒")

    # 3. 可视化对比
    print("\n" + "="*30)
    print("生成模型对比图表")
    print("="*30)
    evaluate.plot_model_comparison(results)

    # 4. 模型优化 (Hyperparameter Tuning)
    print("\n" + "="*30)
    print("开始逻辑回归超参数调优")
    print("="*30)
    print("正在进行网格搜索 (Grid Search)...")
    
    hyperparameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'] 
    }

    start_time = time.time()
    # n_jobs=-1 仍然保留，利用多核加速
    grid_clf = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, verbose=1, n_jobs=-1)
    grid_clf.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\n🎉 调优完成！耗时: {elapsed:.2f} 秒")
    print(f"最佳参数: {grid_clf.best_params_}")
    print(f"最佳验证集分数: {grid_clf.best_score_:.4f}")

    # 在测试集上验证最佳模型
    print("\n正在验证最佳模型...")
    best_model = grid_clf.best_estimator_
    final_acc = evaluate.evaluate_model(best_model, X_test, y_test, "Optimized Logistic Regression")

if __name__ == "__main__":
    main()