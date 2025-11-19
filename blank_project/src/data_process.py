import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

pd.set_option('future.no_silent_downcasting', True)

def load_and_preprocess_data(filepath):
    """加载并清洗数据"""
    print("正在加载数据...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # 将所有列名转换为小写，解决大小写不匹配问题
    df.columns = df.columns.str.lower()
    
    # 时间特征提取
    # 表头中有 'date' 和 'time'
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.hour
    
    # 初始缺失值处理
    # 表头中有 'location_easting_osgr', 'longitude' 等
    cols_to_fill_mean = ['location_easting_osgr', 'location_northing_osgr']
    for col in cols_to_fill_mean:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    cols_to_fill_mode = ['longitude', 'latitude']
    for col in cols_to_fill_mode:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 目标变量二值化
    # 表头中有 'accident_severity'
    replace_dict = {'Serious': 0, 'Fatal': 0, 'Slight': 1}
    df['accident_severity'] = df['accident_severity'].replace(replace_dict)
    # 错误处理
    try:
        df['accident_severity'] = df['accident_severity'].astype(int)
    except ValueError:
        print("包含非数值，正在转换...")
        df = df[pd.to_numeric(df['accident_severity'], errors='coerce').notnull()]
        df['accident_severity'] = df['accident_severity'].astype(int)
    
    # 删除无关列
    # 删掉 ID 类、日期类和冗余的行政区划信息
    cols_to_drop = [
        'accident_index',               # ID，无预测意义
        'local_authority_(district)',   # 行政区ID
        'local_authority_(highway)',    # 行政区ID
        'lsoa_of_accident_location',    # 详细位置编码
        'date',                         # 已提取 day, month, week
        'time',                         # 已提取 hour
        'year'                          # 冗余
    ]
    
    # 只删除 DataFrame 中实际存在的列
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    print(f"正在删除无关列: {existing_cols_to_drop}")
    df.drop(existing_cols_to_drop, axis=1, inplace=True)
    
    # 处理分类变量 (One-Hot Encoding)
    print("正在进行独热编码 (One-Hot Encoding)...")
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def get_features_and_target(df):
    """特征选择与划分"""
    print("正在进行特征清洗与选择...")
    X = df.drop('accident_severity', axis=1)
    y = df['accident_severity']

    # 处理所有剩余的 NaN 值
    imputer = SimpleImputer(strategy='mean') 
    X_imputed = imputer.fit_transform(X)
    
    # 转回 DataFrame
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # SelectKBest
    print("正在筛选最佳的 15 个特征...")
    selector = SelectKBest(score_func=f_classif, k=15)
    X_new = selector.fit_transform(X, y)
    
    # 打印选中的列名
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    print(f"选中的特征: {list(selected_features)}")
    
    return X_new, y

def split_and_scale(X, y):
    """切分数据集并标准化"""
    print("正在切分与标准化数据...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test