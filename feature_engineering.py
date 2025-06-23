import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc

def create_route_features(transactions_df):
    """Create features for each route and journey date"""
    print("Creating route-based features...")
    
    # Sort by route, date of journey, and date of inquiry
    transactions_df = transactions_df.sort_values(['srcid', 'destid', 'doj', 'doi'])
    
    # Create route identifier
    transactions_df['route'] = transactions_df['srcid'].astype(str) + '_' + transactions_df['destid'].astype(str)
    
    # Group by route and journey date
    route_features = []
    
    for (route, doj), group in transactions_df.groupby(['route', 'doj']):
        srcid, destid = route.split('_')
        srcid, destid = int(srcid), int(destid)
        
        # Sort by date of inquiry
        group = group.sort_values('doi')
        
        # Basic features
        features = {
            'srcid': srcid,
            'destid': destid,
            'doj': doj,
            'route': route,
            'total_inquiries': len(group),
            'total_seats_booked': group['cumsum_seatcount'].iloc[-1] if len(group) > 0 else 0,
            'total_searches': group['cumsum_searchcount'].iloc[-1] if len(group) > 0 else 0,
            'avg_days_before_journey': group['dbd'].mean(),
            'min_days_before_journey': group['dbd'].min(),
            'max_days_before_journey': group['dbd'].max(),
            'std_days_before_journey': group['dbd'].std()
        }
        
        # Booking momentum features (rate of change)
        if len(group) > 1:
            # Calculate booking momentum (seats booked per day)
            days_span = (group['doi'].max() - group['doi'].min()).days
            if days_span > 0:
                features['booking_momentum'] = group['cumsum_seatcount'].iloc[-1] / days_span
                features['search_momentum'] = group['cumsum_searchcount'].iloc[-1] / days_span
            else:
                features['booking_momentum'] = 0
                features['search_momentum'] = 0
            
            # Conversion rate (seats booked / searches)
            if group['cumsum_searchcount'].iloc[-1] > 0:
                features['conversion_rate'] = group['cumsum_seatcount'].iloc[-1] / group['cumsum_searchcount'].iloc[-1]
            else:
                features['conversion_rate'] = 0
        else:
            features['booking_momentum'] = 0
            features['search_momentum'] = 0
            features['conversion_rate'] = 0
        
        # Regional and tier features
        features['srcid_region'] = group['srcid_region'].iloc[0]
        features['destid_region'] = group['destid_region'].iloc[0]
        features['srcid_tier'] = group['srcid_tier'].iloc[0]
        features['destid_tier'] = group['destid_tier'].iloc[0]
        
        route_features.append(features)
    
    return pd.DataFrame(route_features)

def create_lag_features(df, lag_periods=[1, 3, 7, 14, 30]):
    """Create lag features for historical patterns"""
    print("Creating lag features...")
    
    # Sort by route and date
    df = df.sort_values(['route', 'doj'])
    
    lag_features = []
    
    for route in df['route'].unique():
        route_data = df[df['route'] == route].sort_values('doj')
        
        for i, row in route_data.iterrows():
            features = row.to_dict()
            
            # Create lag features for different periods
            for lag in lag_periods:
                lag_date = row['doj'] - timedelta(days=lag)
                lag_data = route_data[route_data['doj'] == lag_date]
                
                if len(lag_data) > 0:
                    features[f'lag_{lag}d_total_seats'] = lag_data['total_seats_booked'].iloc[0]
                    features[f'lag_{lag}d_total_searches'] = lag_data['total_searches'].iloc[0]
                    features[f'lag_{lag}d_inquiries'] = lag_data['total_inquiries'].iloc[0]
                    features[f'lag_{lag}d_conversion_rate'] = lag_data['conversion_rate'].iloc[0]
                else:
                    features[f'lag_{lag}d_total_seats'] = 0
                    features[f'lag_{lag}d_total_searches'] = 0
                    features[f'lag_{lag}d_inquiries'] = 0
                    features[f'lag_{lag}d_conversion_rate'] = 0
            
            lag_features.append(features)
    
    return pd.DataFrame(lag_features)

def create_rolling_features(df, windows=[3, 7, 14, 30]):
    """Create rolling average features"""
    print("Creating rolling average features...")
    
    # Sort by route and date
    df = df.sort_values(['route', 'doj'])
    
    rolling_features = []
    
    for route in df['route'].unique():
        route_data = df[df['route'] == route].sort_values('doj')
        
        for i, row in route_data.iterrows():
            features = row.to_dict()
            
            # Create rolling averages for different windows
            for window in windows:
                # Get data within the window
                window_start = row['doj'] - timedelta(days=window)
                window_data = route_data[(route_data['doj'] >= window_start) & (route_data['doj'] < row['doj'])]
                
                if len(window_data) > 0:
                    features[f'rolling_{window}d_avg_seats'] = window_data['total_seats_booked'].mean()
                    features[f'rolling_{window}d_avg_searches'] = window_data['total_searches'].mean()
                    features[f'rolling_{window}d_avg_inquiries'] = window_data['total_inquiries'].mean()
                    features[f'rolling_{window}d_avg_conversion'] = window_data['conversion_rate'].mean()
                    features[f'rolling_{window}d_std_seats'] = window_data['total_seats_booked'].std()
                    features[f'rolling_{window}d_std_searches'] = window_data['total_searches'].std()
                else:
                    features[f'rolling_{window}d_avg_seats'] = 0
                    features[f'rolling_{window}d_avg_searches'] = 0
                    features[f'rolling_{window}d_avg_inquiries'] = 0
                    features[f'rolling_{window}d_avg_conversion'] = 0
                    features[f'rolling_{window}d_std_seats'] = 0
                    features[f'rolling_{window}d_std_searches'] = 0
            
            rolling_features.append(features)
    
    return pd.DataFrame(rolling_features)

def create_booking_momentum_features(df):
    """Create advanced booking momentum features"""
    print("Creating booking momentum features...")
    
    # Sort by route and date
    df = df.sort_values(['route', 'doj'])
    
    momentum_features = []
    
    for route in df['route'].unique():
        route_data = df[df['route'] == route].sort_values('doj')
        
        for i, row in route_data.iterrows():
            features = row.to_dict()
            
            # Get recent data (last 7 days)
            recent_start = row['doj'] - timedelta(days=7)
            recent_data = route_data[(route_data['doj'] >= recent_start) & (route_data['doj'] < row['doj'])]
            
            if len(recent_data) > 0:
                # Recent momentum
                features['recent_booking_momentum'] = recent_data['total_seats_booked'].sum() / 7
                features['recent_search_momentum'] = recent_data['total_searches'].sum() / 7
                
                # Momentum acceleration (change in momentum)
                if len(recent_data) >= 2:
                    recent_momentum = recent_data['total_seats_booked'].iloc[-1] - recent_data['total_seats_booked'].iloc[0]
                    features['booking_acceleration'] = recent_momentum / 7
                else:
                    features['booking_acceleration'] = 0
            else:
                features['recent_booking_momentum'] = 0
                features['recent_search_momentum'] = 0
                features['booking_acceleration'] = 0
            
            # Peak booking period detection
            if len(route_data) > 0:
                avg_seats = route_data['total_seats_booked'].mean()
                features['is_peak_booking'] = 1 if row['total_seats_booked'] > avg_seats * 1.5 else 0
                features['booking_intensity'] = row['total_seats_booked'] / avg_seats if avg_seats > 0 else 0
            else:
                features['is_peak_booking'] = 0
                features['booking_intensity'] = 0
            
            momentum_features.append(features)
    
    return pd.DataFrame(momentum_features)

def create_temporal_features(df):
    """Create temporal features"""
    print("Creating temporal features...")
    
    df = df.copy()
    
    # Extract temporal components
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    df['quarter'] = df['doj'].dt.quarter
    df['year'] = df['doj'].dt.year
    df['day_of_year'] = df['doj'].dt.dayofyear
    df['week_of_year'] = df['doj'].dt.isocalendar().week
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Holiday season indicators (approximate)
    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1) | 
                               (df['month'] == 4) | (df['month'] == 5)).astype(int)
    
    # Peak travel months (summer and winter)
    df['is_peak_travel'] = ((df['month'] >= 6) & (df['month'] <= 8)) | ((df['month'] == 12) | (df['month'] == 1))
    df['is_peak_travel'] = df['is_peak_travel'].astype(int)
    
    return df

def get_indian_holidays():
    """Get Indian national holidays for 2023-2025"""
    print("Creating Indian holiday calendar...")
    
    # Major Indian National Holidays (fixed dates)
    national_holidays = {
        # Republic Day
        '2023-01-26': 'Republic Day',
        '2024-01-26': 'Republic Day', 
        '2025-01-26': 'Republic Day',
        
        # Independence Day
        '2023-08-15': 'Independence Day',
        '2024-08-15': 'Independence Day',
        '2025-08-15': 'Independence Day',
        
        # Gandhi Jayanti
        '2023-10-02': 'Gandhi Jayanti',
        '2024-10-02': 'Gandhi Jayanti',
        '2025-10-02': 'Gandhi Jayanti',
        
        # Christmas
        '2023-12-25': 'Christmas',
        '2024-12-25': 'Christmas',
        '2025-12-25': 'Christmas',
        
        # New Year
        '2023-01-01': 'New Year',
        '2024-01-01': 'New Year',
        '2025-01-01': 'New Year',
    }
    
    # Regional and Festival Holidays (approximate dates - these vary by year)
    festival_holidays = {
        # Diwali (approximate dates)
        '2023-11-12': 'Diwali',
        '2024-11-01': 'Diwali',
        '2025-10-21': 'Diwali',
        
        # Holi (approximate dates)
        '2023-03-08': 'Holi',
        '2024-03-25': 'Holi',
        '2025-03-14': 'Holi',
        
        # Raksha Bandhan (approximate dates)
        '2023-08-30': 'Raksha Bandhan',
        '2024-08-19': 'Raksha Bandhan',
        '2025-08-09': 'Raksha Bandhan',
        
        # Janmashtami (approximate dates)
        '2023-09-07': 'Janmashtami',
        '2024-08-26': 'Janmashtami',
        '2025-08-15': 'Janmashtami',
        
        # Ganesh Chaturthi (approximate dates)
        '2023-09-19': 'Ganesh Chaturthi',
        '2024-09-07': 'Ganesh Chaturthi',
        '2025-08-28': 'Ganesh Chaturthi',
        
        # Dussehra (approximate dates)
        '2023-10-24': 'Dussehra',
        '2024-10-12': 'Dussehra',
        '2025-10-02': 'Dussehra',
        
        # Eid al-Fitr (approximate dates)
        '2023-04-21': 'Eid al-Fitr',
        '2024-04-10': 'Eid al-Fitr',
        '2025-03-31': 'Eid al-Fitr',
        
        # Eid al-Adha (approximate dates)
        '2023-06-29': 'Eid al-Adha',
        '2024-06-17': 'Eid al-Adha',
        '2025-06-07': 'Eid al-Adha',
        
        # Guru Nanak Jayanti (approximate dates)
        '2023-11-27': 'Guru Nanak Jayanti',
        '2024-11-15': 'Guru Nanak Jayanti',
        '2025-11-05': 'Guru Nanak Jayanti',
        
        # Mahavir Jayanti (approximate dates)
        '2023-04-04': 'Mahavir Jayanti',
        '2024-03-21': 'Mahavir Jayanti',
        '2025-04-10': 'Mahavir Jayanti',
        
        # Buddha Purnima (approximate dates)
        '2023-05-05': 'Buddha Purnima',
        '2024-05-23': 'Buddha Purnima',
        '2025-05-13': 'Buddha Purnima',
    }
    
    # Combine all holidays
    all_holidays = {**national_holidays, **festival_holidays}
    
    # Create holiday dataframe
    holiday_df = pd.DataFrame([
        {'date': pd.to_datetime(date_str), 'holiday_name': name, 'holiday_type': 'national' if date_str in national_holidays else 'festival'}
        for date_str, name in all_holidays.items()
    ])
    
    return holiday_df

def add_holiday_features(df):
    """Add holiday features to the dataset"""
    print("Adding holiday features...")
    
    # Get holiday calendar
    holiday_df = get_indian_holidays()
    
    # Ensure doj is datetime
    df['doj'] = pd.to_datetime(df['doj'])
    
    # Convert doj to date for matching
    df['doj_date'] = df['doj'].dt.date
    
    # Create holiday features
    df['is_holiday'] = df['doj_date'].isin(holiday_df['date'].dt.date).astype(int)
    df['is_national_holiday'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_type'] == 'national']['date'].dt.date).astype(int)
    df['is_festival'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_type'] == 'festival']['date'].dt.date).astype(int)
    
    # Add specific holiday indicators
    df['is_diwali'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'Diwali']['date'].dt.date).astype(int)
    df['is_holi'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'Holi']['date'].dt.date).astype(int)
    df['is_republic_day'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'Republic Day']['date'].dt.date).astype(int)
    df['is_independence_day'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'Independence Day']['date'].dt.date).astype(int)
    df['is_christmas'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'Christmas']['date'].dt.date).astype(int)
    df['is_new_year'] = df['doj_date'].isin(holiday_df[holiday_df['holiday_name'] == 'New Year']['date'].dt.date).astype(int)
    
    # Days before/after major holidays
    diwali_dates = holiday_df[holiday_df['holiday_name'] == 'Diwali']['date']
    holi_dates = holiday_df[holiday_df['holiday_name'] == 'Holi']['date']
    
    def min_days_before_holiday(date_series, target_date):
        if len(date_series) == 0:
            return 365
        return min(abs((pd.to_datetime(target_date) - date_series).dt.days))
    
    df['days_before_diwali'] = df['doj'].apply(lambda x: min_days_before_holiday(diwali_dates, x))
    df['days_before_holi'] = df['doj'].apply(lambda x: min_days_before_holiday(holi_dates, x))
    
    # Remove temporary date column
    df = df.drop('doj_date', axis=1)
    
    return df

def main():
    print("Starting feature engineering process...")
    
    # Load cleaned transactions data
    print("Loading cleaned transactions data...")
    transactions_df = pd.read_csv('train/transactions_cleaned.csv')
    transactions_df['doj'] = pd.to_datetime(transactions_df['doj'])
    transactions_df['doi'] = pd.to_datetime(transactions_df['doi'])
    
    print(f"Loaded {len(transactions_df)} transaction records")
    
    # Step 1: Create route-based features
    route_features = create_route_features(transactions_df)
    print(f"Created route features for {len(route_features)} route-date combinations")
    
    # Step 2: Create lag features
    lag_features = create_lag_features(route_features)
    print(f"Created lag features for {len(lag_features)} records")
    
    # Step 3: Create rolling features
    rolling_features = create_rolling_features(lag_features)
    print(f"Created rolling features for {len(rolling_features)} records")
    
    # Step 4: Create booking momentum features
    momentum_features = create_booking_momentum_features(rolling_features)
    print(f"Created momentum features for {len(momentum_features)} records")
    
    # Step 5: Create temporal features
    temporal_features = create_temporal_features(momentum_features)
    print(f"Created temporal features for {len(temporal_features)} records")
    
    # Step 6: Create holiday features
    final_features = add_holiday_features(temporal_features)
    print(f"Created holiday features for {len(final_features)} records")
    
    # Save the engineered features
    print("Saving engineered features...")
    final_features.to_csv('route_features_engineered.csv', index=False)
    
    # Create a summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    print(f"Total route-date combinations: {len(final_features)}")
    print(f"Total features created: {len(final_features.columns)}")
    print(f"Memory usage: {final_features.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
    
    print(f"\nFeature categories:")
    print(f"  - Basic route features: 12")
    print(f"  - Lag features: {5 * 4} (5 periods × 4 metrics)")
    print(f"  - Rolling features: {4 * 6} (4 windows × 6 metrics)")
    print(f"  - Momentum features: 5")
    print(f"  - Temporal features: 8")
    print(f"  - Holiday features: 10")
    
    print(f"\nSample features:")
    sample_cols = final_features.columns[:10].tolist()
    for col in sample_cols:
        print(f"  - {col}")
    
    print(f"\nDate range: {final_features['doj'].min()} to {final_features['doj'].max()}")
    print(f"Unique routes: {final_features['route'].nunique()}")
    
    # Holiday statistics
    holiday_cols = [col for col in final_features.columns if col.startswith('is_') and 'holiday' in col or col in ['is_diwali', 'is_holi', 'is_christmas', 'is_new_year']]
    print(f"\nHoliday feature statistics:")
    for col in holiday_cols:
        if col in final_features.columns:
            count = final_features[col].sum()
            percentage = (count / len(final_features)) * 100
            print(f"  {col}: {count} records ({percentage:.2f}%)")
    
    # Clean up memory
    del transactions_df, route_features, lag_features, rolling_features, momentum_features, temporal_features
    gc.collect()
    
    print("\nFeature engineering completed successfully!")
    print("Output file: route_features_engineered.csv")

if __name__ == "__main__":
    main() 