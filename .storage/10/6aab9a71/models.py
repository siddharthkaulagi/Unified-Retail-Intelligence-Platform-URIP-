import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    """Collection of forecasting models for retail sales prediction"""
    
    def __init__(self):
        self.models = {}
    
    def create_features(self, df, date_col='ds'):
        """Create time-based features for ML models"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Time-based features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        
        # Lag features
        df['sales_lag_1'] = df['y'].shift(1)
        df['sales_lag_7'] = df['y'].shift(7)
        df['sales_lag_30'] = df['y'].shift(30)
        
        # Rolling features
        df['sales_roll_7'] = df['y'].rolling(window=7).mean()
        df['sales_roll_30'] = df['y'].rolling(window=30).mean()
        
        return df
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate forecasting metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def train_prophet(self, df, forecast_days=30, seasonality_mode='additive', include_holidays=True):
        """Train Prophet model"""
        try:
            from prophet import Prophet
            
            # Prepare data
            train_df = df.copy()
            if 'ds' not in train_df.columns or 'y' not in train_df.columns:
                train_df.columns = ['ds', 'y']
            
            # Initialize model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add holidays if requested
            if include_holidays:
                try:
                    from prophet.make_holidays import make_holidays_df
                    holidays = make_holidays_df(
                        year_list=range(train_df['ds'].dt.year.min(), 
                                      train_df['ds'].dt.year.max() + 2),
                        country='US'
                    )
                    model.add_country_holidays(country_name='US')
                except:
                    pass  # Skip holidays if not available
            
            # Train model
            model.fit(train_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Split historical and forecast
            historical = forecast[forecast['ds'] <= train_df['ds'].max()]
            future_forecast = forecast[forecast['ds'] > train_df['ds'].max()]
            
            # Calculate metrics on historical data
            y_true = train_df['y'].values
            y_pred = historical['yhat'].values[-len(y_true):]
            metrics = self.calculate_metrics(y_true, y_pred)
            
            return {
                'model': model,
                'forecast': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'historical': train_df,
                'metrics': metrics
            }
            
        except ImportError:
            # Fallback simple forecast if Prophet not available
            return self._simple_forecast(df, forecast_days, 'Prophet (Fallback)')
        except Exception as e:
            print(f"Prophet error: {e}")
            return self._simple_forecast(df, forecast_days, 'Prophet (Error)')
    
    def train_random_forest(self, df, forecast_days=30, train_split=0.8):
        """Train Random Forest model"""
        try:
            # Prepare data
            data_df = df.copy()
            if 'ds' not in data_df.columns or 'y' not in data_df.columns:
                data_df.columns = ['ds', 'y']
            
            # Create features
            feature_df = self.create_features(data_df)
            
            # Remove rows with NaN values
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 50:  # Not enough data
                return self._simple_forecast(df, forecast_days, 'Random Forest (Insufficient Data)')
            
            # Prepare features
            feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
                           'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
                           'sales_roll_7', 'sales_roll_30']
            
            X = feature_df[feature_cols]
            y = feature_df['y']
            
            # Train-test split
            split_idx = int(len(X) * train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Generate future predictions
            last_date = pd.to_datetime(data_df['ds'].max())
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            # Create future features (simplified)
            future_features = []
            last_sales = data_df['y'].iloc[-1]
            
            for date in future_dates:
                features = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.dayofweek,
                    'quarter': date.quarter,
                    'is_weekend': 1 if date.dayofweek >= 5 else 0,
                    'sales_lag_1': last_sales,
                    'sales_lag_7': last_sales,
                    'sales_lag_30': last_sales,
                    'sales_roll_7': last_sales,
                    'sales_roll_30': last_sales
                }
                future_features.append(features)
            
            future_X = pd.DataFrame(future_features)
            future_pred = model.predict(future_X)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': future_pred
            })
            
            return {
                'model': model,
                'forecast': forecast_df,
                'historical': data_df,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Random Forest error: {e}")
            return self._simple_forecast(df, forecast_days, 'Random Forest (Error)')
    
    def train_lightgbm(self, df, forecast_days=30, train_split=0.8):
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
            
            # Prepare data
            data_df = df.copy()
            if 'ds' not in data_df.columns or 'y' not in data_df.columns:
                data_df.columns = ['ds', 'y']
            
            # Create features
            feature_df = self.create_features(data_df)
            feature_df = feature_df.dropna()
            
            if len(feature_df) < 50:
                return self._simple_forecast(df, forecast_days, 'LightGBM (Insufficient Data)')
            
            # Prepare features
            feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
                           'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
                           'sales_roll_7', 'sales_roll_30']
            
            X = feature_df[feature_cols]
            y = feature_df['y']
            
            # Train-test split
            split_idx = int(len(X) * train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            
            model = lgb.train(params, train_data, num_boost_round=100)
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Generate future predictions
            last_date = pd.to_datetime(data_df['ds'].max())
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            # Create future features
            future_features = []
            last_sales = data_df['y'].iloc[-1]
            
            for date in future_dates:
                features = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'dayofweek': date.dayofweek,
                    'quarter': date.quarter,
                    'is_weekend': 1 if date.dayofweek >= 5 else 0,
                    'sales_lag_1': last_sales,
                    'sales_lag_7': last_sales,
                    'sales_lag_30': last_sales,
                    'sales_roll_7': last_sales,
                    'sales_roll_30': last_sales
                }
                future_features.append(features)
            
            future_X = pd.DataFrame(future_features)
            future_pred = model.predict(future_X)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': future_pred
            })
            
            return {
                'model': model,
                'forecast': forecast_df,
                'historical': data_df,
                'metrics': metrics
            }
            
        except ImportError:
            return self._simple_forecast(df, forecast_days, 'LightGBM (Not Available)')
        except Exception as e:
            print(f"LightGBM error: {e}")
            return self._simple_forecast(df, forecast_days, 'LightGBM (Error)')
    
    def _simple_forecast(self, df, forecast_days, model_name):
        """Simple fallback forecast using moving average"""
        data_df = df.copy()
        if 'ds' not in data_df.columns or 'y' not in data_df.columns:
            data_df.columns = ['ds', 'y']
        
        # Simple moving average forecast
        window = min(30, len(data_df))
        avg_sales = data_df['y'].tail(window).mean()
        
        # Add some trend
        recent_trend = (data_df['y'].tail(7).mean() - data_df['y'].head(7).mean()) / len(data_df)
        
        # Generate future dates and predictions
        last_date = pd.to_datetime(data_df['ds'].max())
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=forecast_days, freq='D')
        
        # Simple forecast with trend and seasonality
        future_pred = []
        for i, date in enumerate(future_dates):
            base_pred = avg_sales + (recent_trend * i)
            # Add weekly seasonality
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofweek / 7)
            pred = base_pred * seasonal_factor
            future_pred.append(max(0, pred))  # Ensure non-negative
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_pred
        })
        
        # Simple metrics (using last 30 days)
        recent_data = data_df.tail(30)
        y_true = recent_data['y'].values
        y_pred = np.full(len(y_true), avg_sales)
        metrics = self.calculate_metrics(y_true, y_pred)
        
        return {
            'model': f'{model_name} - Simple MA',
            'forecast': forecast_df,
            'historical': data_df,
            'metrics': metrics
        }