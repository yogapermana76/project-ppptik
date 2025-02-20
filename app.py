import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

url = "https://docs.google.com/spreadsheets/d/1WwL2jnNXBEVNTDOX-r4aMOtxEGLjljtlrLEq6aV0Q-s/export?format=csv"
df = pd.read_csv(url)

# Function to format price
def format_price(price):
    return f"Rp {price:,.0f}".replace(',', '.')

# Function to evaluate and display model results
def evaluate_model(X_train, X_test, y_train, y_test, condition_type):
    # Train model with adjusted parameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluasi Model untuk Kamera {condition_type}:")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSE: {mse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title(f'Perbandingan Harga Aktual vs Prediksi - Kamera {condition_type}')
    plt.show()

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title(f'Pentingnya Fitur dalam Prediksi - Kamera {condition_type}')
    plt.show()

    return model, mape, r2

df['Jumlah piksel'] = df['Jumlah piksel'].str.replace(' MP', '').astype(float)

# Calculate camera age
current_year = datetime.now().year
df['Umur Kamera'] = current_year - df['Tahun Rilis']

# Add price per megapixel feature
df['Harga per Megapixel'] = df['Harga'] / df['Jumlah piksel']

# Add ISO range feature
df['ISO Range'] = df['ISO max'] - df['ISO min']

# Split data by condition
df_new = df[df['Kondisi'] == 'Baru']
df_used = df[df['Kondisi'] == 'Bekas']

print(f"Jumlah data kamera baru: {len(df_new)}")
print(f"Jumlah data kamera bekas: {len(df_used)}")

# Create label encoders
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_category = LabelEncoder()
le_format = LabelEncoder()

# Prepare features with new additions
features = [
    'Brand_encoded', 'Model_encoded', 'Category_encoded',
    'Jumlah piksel', 'ISO min', 'ISO max', 'ISO Range',
    'fps', 'Format_encoded', 'Tahun Rilis', 'Umur Kamera',
    'Harga per Megapixel'
]

# Process new cameras
df_new_encoded = df_new.copy()
df_new_encoded['Brand_encoded'] = le_brand.fit_transform(df_new['Merek'])
df_new_encoded['Model_encoded'] = le_model.fit_transform(df_new['Model'])
df_new_encoded['Category_encoded'] = le_category.fit_transform(df_new['Kategori'])
df_new_encoded['Format_encoded'] = le_format.fit_transform(df_new['Format'])

X_new = df_new_encoded[features]
y_new = df_new_encoded['Harga']

# Process used cameras
df_used_encoded = df_used.copy()
df_used_encoded['Brand_encoded'] = le_brand.fit_transform(df_used['Merek'])
df_used_encoded['Model_encoded'] = le_model.fit_transform(df_used['Model'])
df_used_encoded['Category_encoded'] = le_category.fit_transform(df_used['Kategori'])
df_used_encoded['Format_encoded'] = le_format.fit_transform(df_used['Format'])

X_used = df_used_encoded[features]
y_used = df_used_encoded['Harga']

# Scale features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)
X_used_scaled = scaler.fit_transform(X_used)

# Split data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)
X_train_used, X_test_used, y_train_used, y_test_used = train_test_split(X_used_scaled, y_used, test_size=0.2, random_state=42)

# Evaluate models
model_new, mape_new, r2_new = evaluate_model(X_train_new, X_test_new, y_train_new, y_test_new, "Baru")
model_used, mape_used, r2_used = evaluate_model(X_train_used, X_test_used, y_train_used, y_test_used, "Bekas")

# Print price ranges
print("\nRange Harga Kamera Baru:")
print(f"Minimum: {format_price(df_new['Harga'].min())}")
print(f"Maximum: {format_price(df_new['Harga'].max())}")
print(f"Rata-rata: {format_price(df_new['Harga'].mean())}")

print("\nRange Harga Kamera Bekas:")
print(f"Minimum: {format_price(df_used['Harga'].min())}")
print(f"Maximum: {format_price(df_used['Harga'].max())}")
print(f"Rata-rata: {format_price(df_used['Harga'].mean())}")

# Sample predictions for both models
print("\nContoh Prediksi Kamera Baru:")
sample_new = X_test_new[:3]
actual_new = y_test_new.iloc[:3]
pred_new = model_new.predict(sample_new)

for i, (pred, actual) in enumerate(zip(pred_new, actual_new)):
    error_pct = abs(pred - actual) / actual * 100
    print(f"\nSample {i+1}:")
    print(f"Harga Aktual: {format_price(actual)}")
    print(f"Harga Prediksi: {format_price(pred)}")
    print(f"Error: {error_pct:.2f}%")

print("\nContoh Prediksi Kamera Bekas:")
sample_used = X_test_used[:3]
actual_used = y_test_used.iloc[:3]
pred_used = model_used.predict(sample_used)

for i, (pred, actual) in enumerate(zip(pred_used, actual_used)):
    error_pct = abs(pred - actual) / actual * 100
    print(f"\nSample {i+1}:")
    print(f"Harga Aktual: {format_price(actual)}")
    print(f"Harga Prediksi: {format_price(pred)}")
    print(f"Error: {error_pct:.2f}%")