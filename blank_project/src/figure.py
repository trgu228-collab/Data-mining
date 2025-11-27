import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 准备数据
# ==========================================
# 模型数据
model_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [0.8484, 0.8397, 0.8439, 0.8490, 0.8478],
    'Time': [0.08, 0.08, 2.17, 2.67, 0.25]
}
df_models = pd.DataFrame(model_data)

# 特征数据 (我帮你做了分类整理，让图表更有逻辑)
features_data = {
    'Feature': [
        'Speed Limit', 'No. of Vehicles', 'No. of Casualties',  # 数值型
        'Urban Area', 'Darkness (No Light)', 'Daylight',        # 环境
        'Single Carriageway', 'Roundabout (Road)',              # 道路
        'Police Attended',                                      # 事故处理
        'Auto Signal', 'Give Way', 'Data Missing (Junc)', 'Not at Junction', 'Roundabout (Junc)', # 路口控制
        'Metropolitan Police'                                   # 警力
    ],
    'Category': [
        'Core Factors', 'Core Factors', 'Core Factors',
        'Environment', 'Environment', 'Environment',
        'Road Layout', 'Road Layout',
        'Response',
        'Junction Info', 'Junction Info', 'Junction Info', 'Junction Info', 'Junction Info',
        'Response'
    ],
    # 给一个虚拟的“重要性”值，或者全部设为1表示“被选中”
    # 这里为了画图好看，我们假设它们同等重要，或者你可以填入 SelectKBest 的 scores
    'Selected': [1] * 15 
}
df_features = pd.DataFrame(features_data)
df_features = df_features.sort_values(by=['Category', 'Feature']) # 按类别排序

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 图 1: 纵轴拉长的效率对比散点图
# ==========================================
# figsize=(10, 8) -> 增加高度，让纵向看起来更舒展
plt.figure(figsize=(10, 8)) 

sns.scatterplot(x='Time', y='Accuracy', data=df_models, s=400, hue='Model', style='Model', palette='deep', legend=False)

plt.title('Efficiency Trade-off: Accuracy vs. Time', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Training Time (Seconds) - Lower is Faster', fontsize=14)
plt.ylabel('Accuracy Score - Higher is Better', fontsize=14)

# 设置坐标轴范围 (拉开差距的关键)
plt.ylim(0.838, 0.850) # 聚焦在 0.838 到 0.850 之间
plt.xlim(-0.2, 3.0)

# 标注点
for i in range(df_models.shape[0]):
    x_offset = 0.1
    y_offset = 0
    # 微调文字位置
    if df_models['Model'][i] == 'XGBoost': y_offset = -0.0008
    if df_models['Model'][i] == 'Logistic Regression': y_offset = 0.0005
    
    plt.text(df_models['Time'][i] + x_offset, 
             df_models['Accuracy'][i] + y_offset, 
             df_models['Model'][i], 
             fontsize=12, fontweight='bold', color='#333')

# 高效区背景色
plt.axvspan(-0.2, 0.5, color='green', alpha=0.05)
plt.text(0.05, 0.8385, "High Efficiency Zone\n(Fast & Accurate)", fontsize=12, color='green', fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plot_2_efficiency_tall.png', dpi=300)
print("✅ 图1 (拉长的散点图) 已保存")
plt.show()


# ==========================================
# 图 3: 高级特征棒棒糖图 (Categorized Lollipop Chart)
# ==========================================
plt.figure(figsize=(12, 10))

# 按照类别分组绘制
groups = df_features.groupby('Category')
y_pos = range(len(df_features))
colors = sns.color_palette("Set2", n_colors=len(groups))

# 绘制线条 (Stem)
plt.hlines(y=y_pos, xmin=0, xmax=1, color='grey', alpha=0.4, linewidth=1)

# 绘制圆点 (Head)，按类别上色
current_y = 0
for (name, group), color in zip(groups, colors):
    plt.scatter([1] * len(group), range(current_y, current_y + len(group)), 
                color=color, s=150, label=name, zorder=3)
    
    # 在圆点左侧添加标签文字
    for i, feature in enumerate(group['Feature']):
        plt.text(0.98, current_y + i, feature, 
                 ha='right', va='center', fontsize=12, fontweight='bold', color='#444')
    
    current_y += len(group)

# 美化
plt.title('Top 15 Key Features Selected by Model', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Relevance (Selected)', fontsize=14)
plt.yticks([]) # 隐藏Y轴刻度
plt.xticks([]) # 隐藏X轴刻度（因为只是展示选中状态）
plt.box(False) # 去掉边框

# 添加图例
plt.legend(title="Feature Category", title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加装饰性竖线
plt.axvline(x=1, color='black', alpha=0.1, linewidth=5)

plt.tight_layout()
plt.savefig('plot_3_features_lollipop.png', dpi=300)
print("✅ 图2 (高级特征棒棒糖图) 已保存")
plt.show()