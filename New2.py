import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

print("[DEBUG] Script started.")
# Load data
train = pd.read_csv("train/train.csv")
test = pd.read_csv("test_8gqdJqH.csv")
transactions = pd.read_csv("train/transactions.csv")
holidays = pd.read_csv("holiday_data_2023_2025.csv")

print("[DEBUG] Data loaded. Train shape:", train.shape, "Test shape:", test.shape, "Transactions shape:", transactions.shape, "Holidays shape:", holidays.shape)

# Generate route_key in train if not exists
train['doj'] = pd.to_datetime(train['doj'])
train['route_key'] = train['doj'].dt.strftime('%Y-%m-%d') + "" + train['srcid'].astype(str) + "" + train['destid'].astype(str)
test['doj'] = pd.to_datetime(test['doj'])  # test already has route_key
test_route_key = test['route_key'].copy()

transactions['doj'] = pd.to_datetime(transactions['doj'])
transactions['doi'] = pd.to_datetime(transactions['doi'])
holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])

# Filter for dbd = 15
trans_15 = transactions[transactions['dbd'] == 15].copy()
trans_15 = trans_15[['doj', 'srcid', 'destid', 'cumsum_seatcount', 'cumsum_searchcount',
                     'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']]

# Encode categorical columns
for col in ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']:
    trans_15[col] = LabelEncoder().fit_transform(trans_15[col].astype(str))

# Merge transactions
train = train.merge(trans_15, on=['doj', 'srcid', 'destid'], how='left')
test = test.merge(trans_15, on=['doj', 'srcid', 'destid'], how='left')

# Merge holidays
for df in [train, test]:
    df = df.merge(holidays, left_on='doj', right_on='holiday_date', how='left')
    df['is_holiday'] = df['holiday_name'].notnull().astype(int)
    df['is_long_weekend'] = df['is_long_weekend'].fillna(0).astype(int)
    df.drop(columns=['holiday_date', 'holiday_name', 'day_of_week'], inplace=True)

# Date features
for df in [train, test]:
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['doj'].dt.month
    df['day_of_year'] = df['doj'].dt.dayofyear
    df['week_of_year'] = df['doj'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['doj'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['doj'].dt.is_month_end.astype(int)

# Fallback mean prediction
route_means = train.groupby('route_key')['final_seatcount'].mean().to_dict()
test['fallback_pred'] = test['route_key'].map(route_means)
test['fallback_pred'].fillna(train['final_seatcount'].mean(), inplace=True)

# Feature engineering
for df in [train, test]:
    df['search_to_seat_ratio'] = df['cumsum_searchcount'] / (df['cumsum_seatcount'] + 1)
    df['tier_diff'] = abs(df['srcid_tier'] - df['destid_tier'])

# Fill missing
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Feature selection
features = [col for col in train.columns if col not in ['final_seatcount', 'doj', 'route_key']]
X = train[features]
y = train['final_seatcount']
X_test = test[features]

print("[DEBUG] Feature engineering and merging complete. Train shape:", train.shape, "Test shape:", test.shape)

# LightGBM KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("[DEBUG] Starting KFold training...")
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

print("[DEBUG] KFold training complete. Calculating RMSE...")

# Evaluate
print(f"[DEBUG] OOF predictions: {oof_preds[:5]}")
print(f"[DEBUG] y true: {y[:5].values}")
rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"OOF RMSE: {rmse:.4f}")

# Boost under predictions
test_preds = np.where(test_preds < 300, test_preds * 1.1, test_preds)

# Final fallback
final_preds = np.where(test_preds == 0, test['fallback_pred'], test_preds)

# Create submission
submission = pd.DataFrame({
    'route_key': test_route_key,
    'final_seatcount': final_preds.astype(int)
})
submission.to_csv("submission_file.csv", index=False)
print("Ã¢Å“â€¦ Final submission_file.csvÂ generated.")
Apollo.io