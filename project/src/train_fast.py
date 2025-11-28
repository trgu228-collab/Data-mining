import data_process
import baseline
import evaluate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import time

def main():
    # 1. 数据准备流程
    # 确保路径正确
    filepath = r'blank_project\data\dft-road-casualty-statistics-accident-2021.csv'
    
    try:
        print("正在加载部分数据")
        df = data_process.load_and_preprocess_data(filepath)

        # frac=0.05 表示只取 5% 的数据
        # random_state=42 保证每次采样的结果一样，结果可复现
        print(f"原始数据量: {df.shape[0]} 行")
        df = df.sample(frac=0.05, random_state=42) 
        print(f"⏩ 采样后数据量: {df.shape[0]} 行 ")

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
        # [核心修改] 明确跳过 KNN，其他全部运行
        if name == 'KNN':
            print(f"⏩ 跳过模型: {name} (避免卡顿)")
            continue

        print(f"\n正在训练模型: {name} ...")
        start_time = time.time()
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测与评估
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
    
    # 恢复较完整的参数搜索空间
    hyperparameters = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'] 
    }

    start_time = time.time()
    
    # [重要] 保持 n_jobs=1 以防止 Windows 报错
    grid_clf = GridSearchCV(LogisticRegression(), hyperparameters, cv=5, verbose=1, n_jobs=1)
    
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