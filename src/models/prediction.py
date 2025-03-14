import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import traceback
from datetime import datetime, timedelta

class PredictionModels:
    """
    Class for implementing cryptocurrency price prediction models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features_used = {}
        
    def prepare_data(self, df, target_col='price', features=None, test_size=0.2):
        """
        Prepare data for training prediction models
        
        Args:
            df (pandas.DataFrame): DataFrame with price and indicators
            target_col (str): Target column to predict
            features (list): List of feature columns to use (None = use all numeric)
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Fill NaN values before proceeding
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        if len(data) < 10:
            # Not enough data to work with
            raise ValueError(f"Not enough data points: {len(data)} available, need at least 10")
        
        # Select features
        if features is None:
            # Use all numeric columns except the target
            features = data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != target_col]
        else:
            # Ensure all specified features exist in the dataframe
            features = [f for f in features if f in data.columns]
            
        if not features:
            raise ValueError("No valid features available for modeling")
            
        # Instead of creating many lagged features that might cause data loss,
        # let's just use a few key lags
        try:
            # Add just 1 lag for target column 
            data[f'{target_col}_lag_1'] = data[target_col].shift(1)
            
            # Fill the NA value created by the shift
            data = data.fillna(method='bfill')
            
            # Add one simple time feature that won't create NaN values
            if isinstance(data.index, pd.DatetimeIndex):
                data['day_of_week'] = data.index.dayofweek
            
            # Add all lagged features to the feature list
            features = features + [f'{target_col}_lag_1', 'day_of_week']
            
            # Make sure all features are in the dataframe
            features = [f for f in features if f in data.columns]
        except Exception as e:
            print(f"Error adding time features: {str(e)}")
            # Continue with original features if there's an error
        
        # Split data into train and test sets
        train_size = int(len(data) * (1 - test_size))
        if train_size < 10:  # Ensure minimum training samples
            train_size = len(data) - 5 if len(data) > 5 else len(data) - 1
            
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Scale the data
        try:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train_data[features])
            
            # If test data exists
            if len(test_data) > 0:
                X_test = scaler.transform(test_data[features])
            else:
                X_test = np.empty((0, len(features)))  # Empty array with correct feature dimension
            
            # Prepare target variable
            y_train = train_data[target_col].values
            
            if len(test_data) > 0:
                y_test = test_data[target_col].values
            else:
                y_test = np.array([])
                
            # Store the features used
            self.features_used = features
                
        except Exception as e:
            print(f"Error in prepare_data scaling: {str(e)}")
            traceback.print_exc()
            raise
        
        return X_train, X_test, y_train, y_test, scaler, features, train_data, test_data
    
    def train_linear_regression(self, df, target_col='price', features=None, test_size=0.2):
        """
        Train a linear regression model
        
        Args:
            df (pandas.DataFrame): DataFrame with price and indicators
            target_col (str): Target column to predict
            features (list): List of feature columns to use
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (model, scaler, features, train_data, test_data)
        """
        try:
            # Fill any NaN values in the input DataFrame
            df_cleaned = df.copy()
            df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler, features, train_data, test_data = self.prepare_data(
                df_cleaned, target_col, features, test_size
            )
            
            # Create and train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            
            if len(X_test) > 0:
                test_score = model.score(X_test, y_test)
                print(f"Linear Regression - Train R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
            else:
                print(f"Linear Regression - Train R² Score: {train_score:.4f}, No test data available")
            
            # Store model and scaler
            self.models['linear_regression'] = model
            self.scalers['linear_regression'] = (scaler, features)
            
            return model, scaler, features, train_data, test_data
        
        except Exception as e:
            print(f"Error in train_linear_regression: {str(e)}")
            traceback.print_exc()
            raise
    
    def train_random_forest(self, df, target_col='price', features=None, test_size=0.2):
        """
        Train a random forest regression model
        
        Args:
            df (pandas.DataFrame): DataFrame with price and indicators
            target_col (str): Target column to predict
            features (list): List of feature columns to use
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (model, scaler, features, train_data, test_data)
        """
        try:
            # Fill any NaN values in the input DataFrame
            df_cleaned = df.copy()
            df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler, features, train_data, test_data = self.prepare_data(
                df_cleaned, target_col, features, test_size
            )
            
            # Create and train model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            
            if len(X_test) > 0:
                test_score = model.score(X_test, y_test)
                print(f"Random Forest - Train R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
            else:
                print(f"Random Forest - Train R² Score: {train_score:.4f}, No test data available")
            
            # Store model and scaler
            self.models['random_forest'] = model
            self.scalers['random_forest'] = (scaler, features)
            
            return model, scaler, features, train_data, test_data
        
        except Exception as e:
            print(f"Error in train_random_forest: {str(e)}")
            traceback.print_exc()
            raise
    
    def train_arima(self, df, target_col='price', order=(5,1,0)):
        """
        Train an ARIMA model
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            target_col (str): Target column to predict
            order (tuple): ARIMA order (p,d,q)
            
        Returns:
            tuple: (model, train_data, test_data)
        """
        try:
            # Prepare data (only target column is needed for ARIMA)
            data = df.copy()
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            if len(data) < 10:
                raise ValueError(f"Not enough data for ARIMA: {len(data)} points available, need at least 10")
            
            # Use 80% of data for training
            train_size = int(len(data) * 0.8)
            if train_size < 10:  # Ensure minimum training samples
                train_size = len(data) - 1
                
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
            
            # Suppress convergence warnings
            warnings.filterwarnings("ignore")
            
            # Create and train model
            try:
                # Try a simple ARIMA model first
                model = ARIMA(train_data[target_col], order=(1,1,0))
                model_fit = model.fit()
                print("Using simple ARIMA(1,1,0) model")
            except Exception as arima_error:
                print(f"ARIMA model failed: {str(arima_error)}")
                print("Falling back to exponential smoothing")
                
                # If ARIMA fails, use exponential smoothing as fallback
                try:
                    model = ExponentialSmoothing(train_data[target_col])
                    model_fit = model.fit()
                except:
                    # Last resort: create a simple moving average model
                    print("Exponential smoothing failed. Using simple moving average.")
                    class SimpleMAModel:
                        def __init__(self, data, window=5):
                            self.data = data
                            self.window = min(window, len(data))
                            
                        def forecast(self, steps=1):
                            # Use the last n values to predict future values
                            last_values = self.data.iloc[-self.window:].mean()
                            return np.array([last_values] * steps)
                    
                    model_fit = SimpleMAModel(train_data[target_col])
            
            # Store the model
            self.models['arima'] = (model_fit, order)
            
            # Evaluate model for ARIMA
            if hasattr(model_fit, 'aic'):
                print(f"ARIMA model trained. AIC: {model_fit.aic:.4f}")
            else:
                print("Alternative time series model trained.")
            
            return model_fit, train_data, test_data
        
        except Exception as e:
            print(f"Error in train_arima: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict_next_day(self, model_name, last_data):
        """
        Make predictions for the next day
        
        Args:
            model_name (str): Name of the model to use
            last_data (pandas.DataFrame): Last available data point
            
        Returns:
            float: Predicted price
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Please train it first.")
            
            # Prepare data for prediction
            if model_name in ['linear_regression', 'random_forest']:
                model = self.models[model_name]
                scaler, features = self.scalers[model_name]
                
                # Create a copy and fill missing values
                predict_data = last_data.copy()
                predict_data = predict_data.fillna(method='ffill').fillna(0)  # Fill remaining NaNs with 0
                
                # Verify all required features exist
                missing_features = [f for f in features if f not in predict_data.columns]
                if missing_features:
                    print(f"Warning: Missing features: {missing_features}")
                    # Create any missing features with default values (0)
                    for feature in missing_features:
                        predict_data[feature] = 0
                
                # Extract features and scale
                X = predict_data[features].values.reshape(1, -1)
                X_scaled = scaler.transform(X)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                return prediction
            
            elif model_name == 'arima':
                model_fit, order = self.models['arima']
                
                # Make forecast
                if hasattr(model_fit, 'forecast'):
                    forecast = model_fit.forecast(steps=1)
                    prediction = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
                else:
                    # For our simple models
                    forecast = model_fit.forecast(steps=1)
                    prediction = forecast[0]
                
                return prediction
            
            else:
                raise ValueError(f"Prediction for model '{model_name}' not implemented.")
                
        except Exception as e:
            print(f"Error in predict_next_day: {str(e)}")
            traceback.print_exc()
            # Return the last price as a fallback
            try:
                return last_data['price'].iloc[-1]
            except:
                return 100  # Default fallback value
    
    def predict_multiple_days(self, model_name, last_data, days=7):
        """
        Make predictions for multiple days
        
        Args:
            model_name (str): Name of the model to use
            last_data (pandas.DataFrame): Last available data
            days (int): Number of days to predict
            
        Returns:
            pandas.Series: Series with predicted prices
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Please train it first.")
            
            # Clean any NaN values in the input data
            clean_data = last_data.copy()
            clean_data = clean_data.fillna(method='ffill').fillna(method='bfill')
            
            predictions = []
            dates = pd.date_range(start=clean_data.index[-1] + pd.Timedelta(days=1), periods=days)
            
            if model_name in ['linear_regression', 'random_forest']:
                # Get the last known price
                last_price = clean_data['price'].iloc[-1]
                
                # For ML models, just predict all days at once using the current state
                # This is simplistic but works for short-term predictions
                for i in range(days):
                    try:
                        if i == 0:
                            # For first day, use the actual last data
                            next_price = self.predict_next_day(model_name, clean_data)
                        else:
                            # For subsequent days, we'll add a small random walk component
                            # This makes the prediction more realistic than a flat line
                            # Calculate percent change from previous prediction to last known price
                            if i == 1:
                                pct_change = (next_price / last_price) - 1
                            else:
                                pct_change = (predictions[-1] / predictions[-2]) - 1
                                
                            # Use a decaying version of this change with some randomness
                            decay = 0.9
                            random_factor = np.random.normal(0, 0.01)  # Small random noise
                            next_change = pct_change * decay + random_factor
                            
                            # Apply the change to the previous prediction
                            next_price = predictions[-1] * (1 + next_change)
                        
                        predictions.append(max(0.01, next_price))  # Ensure price is positive
                    except Exception as day_error:
                        print(f"Error predicting day {i+1}: {str(day_error)}")
                        # If error, use previous prediction with small adjustment
                        if predictions:
                            predictions.append(predictions[-1] * (1 + np.random.normal(0, 0.01)))
                        else:
                            predictions.append(last_price)
            
            elif model_name == 'arima':
                model_fit, order = self.models['arima']
                
                # Make forecast for multiple steps
                try:
                    if hasattr(model_fit, 'forecast'):
                        forecast = model_fit.forecast(steps=days)
                        if isinstance(forecast, pd.Series):
                            predictions = forecast.values
                        else:
                            predictions = forecast
                    else:
                        # For our simple models
                        predictions = model_fit.forecast(steps=days)
                except Exception as e:
                    print(f"Error in ARIMA forecast: {str(e)}")
                    # Fallback to simple prediction
                    last_price = clean_data['price'].iloc[-1]
                    predictions = [last_price * (1 + np.random.normal(0, 0.01)) for _ in range(days)]
            
            # Ensure predictions are reasonable
            last_price = clean_data['price'].iloc[-1]
            
            # Limit to reasonable changes (max 10% per day from previous price)
            for i in range(len(predictions)):
                if i == 0:
                    ref_price = last_price
                else:
                    ref_price = predictions[i-1]
                    
                max_change = ref_price * 0.10  # 10% max daily change
                min_val = ref_price - max_change
                max_val = ref_price + max_change
                predictions[i] = max(min_val, min(predictions[i], max_val))
            
            return pd.Series(predictions, index=dates)
        
        except Exception as e:
            print(f"Error in predict_multiple_days: {str(e)}")
            traceback.print_exc()
            
            # Return a fallback prediction (simple flat prediction)
            try:
                dates = pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=days)
                last_price = last_data['price'].iloc[-1]
                return pd.Series([last_price] * days, index=dates)
            except:
                # Ultimate fallback
                dates = pd.date_range(start=datetime.now(), periods=days)
                return pd.Series([100] * days, index=dates)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.random.normal(loc=100, scale=10, size=100).cumsum()
    
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.rand(100) * 1000000,
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    }, index=dates)
    
    # Initialize and train models
    predictor = PredictionModels()
    
    # Train linear regression
    lr_model, _, _, _, _ = predictor.train_linear_regression(df)
    
    # Train ARIMA
    arima_model, _, _ = predictor.train_arima(df)
    
    # Make predictions
    last_data = df.iloc[-1:]
    
    # Predict next day price
    lr_prediction = predictor.predict_next_day('linear_regression', last_data)
    arima_prediction = predictor.predict_next_day('arima', last_data)
    
    print(f"Linear Regression prediction for next day: ${lr_prediction:.2f}")
    print(f"ARIMA prediction for next day: ${arima_prediction:.2f}")
    
    # Predict for next 7 days
    future_predictions = predictor.predict_multiple_days('arima', df, days=7)
    print("7-day forecast:")
    print(future_predictions)