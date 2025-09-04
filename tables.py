import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10
})

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DATA_PATH = "forestCover.csv" 
df = pd.read_csv(DATA_PATH, na_values=["?"])

target_col = df.columns[-1]
observation = df['Water_Level']
df['Water_Level'] = df['Observation_ID']
df['Observation_ID'] = observation
df["Soil_Type1"] = df["Soil_Type1"].replace({"positive": 1, "negative": 0}).astype(int)

drop_if_present = ["Water_Level", "Observation_ID", "Inclination", "Aspect"]
to_drop = [c for c in drop_if_present if c in df.columns]
df.drop(columns=to_drop, inplace=True)

# Remove outliers
if "Horizontal_Distance_To_Hydrology" in df.columns:
    outlier_count = (df["Horizontal_Distance_To_Hydrology"] > 10000).sum()
    outlier_percentage = (outlier_count / len(df)) * 100
    print(f"Outliers in Horizontal_Distance_To_Hydrology: {outlier_count} instances ({outlier_percentage:.2f}%)")
    df = df.loc[df["Horizontal_Distance_To_Hydrology"] <= 10000].copy()

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Define feature types for analysis
binary_cols = [col for col in X.columns if col.startswith(('Wilderness_Area', 'Soil_Type'))]
numeric_cols = [col for col in X.columns if col not in binary_cols]

print(f"Dataset shape: {X.shape}")
print(f"Target classes: {sorted(y.unique())}")

def create_class_distribution_table():
    class_counts = y.value_counts().sort_index()
    class_percentages = (class_counts / len(y) * 100).round(2)
    
    table_data = pd.DataFrame({
        'Forest Cover Type': class_counts.index,
        'Sample Count': class_counts.values,
        'Percentage (%)': class_percentages.values
    })
    
    print("TABLE 1: Class Distribution")
    print(table_data.to_string(index=False))
    print(f"Imbalance ratio (max/min): {class_counts.max() / class_counts.min():.2f}")
    return table_data

def create_missing_values_table():
    # Calculate missing value statistics
    missing_stats = df.isnull().sum()
    missing_features = missing_stats[missing_stats > 0]
    
    if len(missing_features) > 0:
        print(f"\nMissing values found in {len(missing_features)} features:")
        print(missing_features)
    
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f"Total missing values: {missing_percentage:.2f}% of all data points")
    else:
        print("No missing values found in the dataset")


def create_feature_statistics_table():
    numeric_stats = df[numeric_cols].describe()
    
    print("\nTABLE 2: Numeric Feature Statistics")
    print(numeric_stats.round(2))
    numeric_stats.round(2).to_csv('results/feature_statistics.csv')
    
    # Check scale differences
    ranges = numeric_stats.loc['max'] - numeric_stats.loc['min']
    print(f"\nScale analysis:")
    print(f"Largest range: {ranges.max():.0f} ({ranges.idxmax()})")
    print(f"Smallest range: {ranges.min():.0f} ({ranges.idxmin()})")
    print(f"Scale ratio (max/min): {ranges.max() / ranges.min():.0f}")
    
    return numeric_stats



create_class_distribution_table()
create_missing_values_table()
create_feature_statistics_table()