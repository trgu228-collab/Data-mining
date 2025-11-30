# DATASET.md
This document outlines the source, features, preprocessing pipeline, and split strategy of the dataset used in the `Road traffic accident prediction.ipynb` project.

The csv is too large, we post the link to accessing the data here
```
https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles/data
```

## 1. Source

* **Filename**: `dft-road-casualty-statistics-accident-2021.csv`
* **Description**: This dataset contains detailed statistical information on road traffic accidents in 2021. The data covers the time, location, road conditions, weather conditions, and accident severity.
* **Original Data Size**: Approximately 2,047,256 records with 34 feature columns.
* **Potential Source**: The data is sourced from the official statistics of the **Department for Transport, UK**.

## 2. License

* This dataset typically belongs to the public domain and follows the **Open Government Licence (OGL)**.

## 3. Feature Descriptions

### Target Variable
* **Accident_Severity**: The severity of the accident.
    * Original Classes: `Fatal`, `Serious`, `Slight`.
    * **Post-processing**: Converted to a binary classification problem.
        * `0`: Fatal or Serious
        * `1`: Slight

### Key Features
The following are some key features from the original dataset:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `1st_Road_Class` | Object | Class of the first road (e.g., A, B, C, Motorway) |
| `Speed_limit` | Float | Speed limit of the accident road section |
| `Weather_Conditions` | Object | Weather conditions (e.g., Raining, Fine, High winds) |
| `Light_Conditions` | Object | Light conditions (e.g., Daylight, Darkness - lights lit) |
| `Road_Surface_Conditions` | Object | Road surface conditions (e.g., Dry, Wet, Frost) |
| `Number_of_Casualties` | Int | Number of casualties |
| `Number_of_Vehicles` | Int | Number of vehicles involved |
| `Day_of_Week` | Object | Day of the week |
| `Time` | Object | Time of accident occurrence |
| `Urban_or_Rural_Area` | Object | Urban or rural area |
| `Longitude` / `Latitude` | Float | Longitude / Latitude coordinates |

## 4. Cleaning & Preprocessing

Before modeling, the following cleaning and transformation steps were performed on the data:

### 4.1. Feature Engineering
* **Date Processing**: Converted the `Date` column to datetime objects and extracted the following new features:
    * `day`
    * `month`
    * `week`: Week number (ISO calendar week)

### 4.2. Missing Value Imputation
Imputation was performed for geographical location columns containing missing values:
* **Mean Imputation**: Used for `location_easting_osgr` and `location_northing_osgr`.
* **Mode Imputation**: Used for `longitude` and `latitude`.

### 4.3. Feature Dropping
Removed identifiers and redundant columns that do not aid prediction (intended removal, code includes the following columns):
* `accident_index`, `accident_reference`
* `local_authority_ons_district`, `local_authority_highway`
* `time`, `date`, `accident_year`
* `lsoa_of_accident_location`

### 4.4. Target Variable Encoding
To simplify the problem into binary classification, the target variable was mapped:
* Original values `1` (Fatal) and `2` (Serious) $\rightarrow$ Mapped to **0**.
* Original value `3` (Slight) $\rightarrow$ Mapped to **1**.

## 5. Feature Selection

To improve model performance and reduce dimensionality, statistical methods were used to select features:
* **Method**: `SelectKBest`
* **Score Function**: `f_classif` (ANOVA F-value)
* **Number Selected**: Retained the top **15 highest-scoring features**.

## 6. Dataset Splits

The data was split into training and testing sets to evaluate the model's generalization ability:

* **Training Set**: 80%
* **Test Set**: 20%
* **Random State**: `0` (To ensure reproducibility)

## 7. Data Scaling

* **Method**: `StandardScaler`
* **Application**: Performed standardization (Z-score Normalization) on the feature matrix to ensure features have 0 mean and unit variance. This was done after the data split and before model training.