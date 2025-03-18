import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
import warnings
import traceback
from datetime import datetime, timedelta

# Attempt to import Prophet (optional dependency)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Prophet model will not be available.")

class PredictionModels:
    """
    Class for implementing cryptocurrency price prediction models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features_used = {}
        self.prophet_uncertainty = None
        
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
        
        # Fill NaN values before proceeding - use modern syntax
        data = data.ffill().bfill()
        
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
            
        # Simplify feature engineering to avoid missing feature issues during prediction
        core_features = features.copy()
        
        # Split data into train and test sets
        train_size = int(len(data) * (1 - test_size))
        if train_size < 10:  # Ensure minimum training samples
            train_size = len(data) - 5 if len(data) > 5 else len(data) - 1
            
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Scale the data
        try:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train_data[core_features])
            
            # If test data exists
            if len(test_data) > 0:
                X_test = scaler.transform(test_data[core_features])
            else:
                X_test = np.empty((0, len(core_features)))  # Empty array with correct feature dimension
            
            # Prepare target variable
            y_train = train_data[target_col].values
            
            if len(test_data) > 0:
                y_test = test_data[target_col].values
            else:
                y_test = np.array([])
                
        except Exception as e:
            print(f"Error in prepare_data scaling: {str(e)}")
            traceback.print_exc()
            raise
        
        return X_train, X_test, y_train, y_test, scaler, core_features, train_data, test_data
    
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
            df_cleaned = df_cleaned.ffill().bfill()
            
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
            df_cleaned = df_cleaned.ffill().bfill()
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler, features, train_data, test_data = self.prepare_data(
                df_cleaned, target_col, features, test_size
            )
            
            # Create and train model with more moderate parameters to avoid overfitting
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

    def train_gradient_boosting(self, df, target_col='price', features=None, test_size=0.2):
        """
        Train a gradient boosting regression model
        
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
            df_cleaned = df_cleaned.ffill().bfill()
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler, features, train_data, test_data = self.prepare_data(
                df_cleaned, target_col, features, test_size
            )
            
            # Create and train model with carefully tuned parameters for time series
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            
            if len(X_test) > 0:
                test_score = model.score(X_test, y_test)
                print(f"Gradient Boosting - Train R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
            else:
                print(f"Gradient Boosting - Train R² Score: {train_score:.4f}, No test data available")
            
            # Store model and scaler
            self.models['gradient_boosting'] = model
            self.scalers['gradient_boosting'] = (scaler, features)
            
            return model, scaler, features, train_data, test_data
        
        except Exception as e:
            print(f"Error in train_gradient_boosting: {str(e)}")
            traceback.print_exc()
            raise
            
    def train_svr(self, df, target_col='price', features=None, test_size=0.2):
        """
        Train a Support Vector Regression model
        
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
            df_cleaned = df_cleaned.ffill().bfill()
            
            # Prepare data
            X_train, X_test, y_train, y_test, scaler, features, train_data, test_data = self.prepare_data(
                df_cleaned, target_col, features, test_size
            )
            
            # Create and train model - using RBF kernel for non-linear patterns
            model = SVR(
                kernel='rbf',      # Radial basis function is good for financial data
                C=100,             # Regularization parameter
                gamma='scale',     # Kernel coefficient
                epsilon=0.1,       # Epsilon in the epsilon-SVR model
                cache_size=1000    # Speed up training with more cache
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            train_score = self._calculate_r2(y_train, train_pred)
            
            if len(X_test) > 0:
                test_pred = model.predict(X_test)
                test_score = self._calculate_r2(y_test, test_pred)
                print(f"SVR - Train R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
            else:
                print(f"SVR - Train R² Score: {train_score:.4f}, No test data available")
            
            # Store model and scaler
            self.models['svr'] = model
            self.scalers['svr'] = (scaler, features)
            
            return model, scaler, features, train_data, test_data
        
        except Exception as e:
            print(f"Error in train_svr: {str(e)}")
            traceback.print_exc()
            raise
    
    def train_prophet(self, df, changepoint_prior_scale=0.05, seasonality_mode='multiplicative'):
        """
        Train a Facebook Prophet model for time series forecasting
        
        Args:
            df (pandas.DataFrame): DataFrame with 'ds' (dates) and 'y' (price) columns
            changepoint_prior_scale (float): Controls flexibility of the trend
            seasonality_mode (str): 'additive' or 'multiplicative' seasonality
            
        Returns:
            tuple: (model, train_data)
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Please install with: pip install prophet")
            
        try:
            # Prepare data (Prophet requires specific column names)
            data = df.copy()
            
            # Make sure the DataFrame has the required columns
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("DataFrame must have 'ds' and 'y' columns for Prophet model")
            
            # Create and train model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False  # Most crypto price series aren't long enough for yearly patterns
            )
            
            # Add additional seasonality if we have enough data
            if len(data) >= 30:
                model.add_seasonality(name='monthly', period=30, fourier_order=5)
            
            # Fit the model
            model.fit(data)
            
            # Make a forecast to evaluate uncertainty
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            
            # Calculate uncertainty as the average percentage difference between upper and lower bounds
            uncertainty = ((forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat']).mean() * 100 / 2
            self.prophet_uncertainty = uncertainty
            
            # Store the model
            self.models['prophet'] = model
            
            print(f"Prophet model trained. Uncertainty: ±{uncertainty:.2f}%")
            
            return model, data
        
        except Exception as e:
            print(f"Error in train_prophet: {str(e)}")
            traceback.print_exc()
            raise
            
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R^2 score manually since SVR doesn't have score method"""
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        if ss_total == 0:  # Avoid division by zero
            return 0
            
        return 1 - (ss_residual / ss_total)
    
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
            if model_name in ['linear_regression', 'random_forest', 'gradient_boosting', 'svr']:
                model = self.models[model_name]
                scaler, features = self.scalers[model_name]
                
                # Create a copy and fill missing values
                predict_data = last_data.copy()
                predict_data = predict_data.ffill().bfill()
                
                # Create a new DataFrame with ONLY the features used during training
                prediction_features = pd.DataFrame(index=predict_data.index)
                
                # Verify all required features exist
                missing_features = [f for f in features if f not in predict_data.columns]
                if missing_features:
                    print(f"Warning: Missing features: {missing_features}")
                
                # Add each feature, using 0 as default for missing ones
                for feature in features:
                    if feature in predict_data.columns:
                        prediction_features[feature] = predict_data[feature]
                    else:
                        prediction_features[feature] = 0
                
                # Use only last row if multiple rows provided
                if len(prediction_features) > 1:
                    prediction_features = prediction_features.iloc[-1:]
                
                # Debug info
                print(f"Features expected by model: {len(features)}")
                print(f"Features provided for prediction: {prediction_features.shape[1]}")
                
                # Extract values and scale
                X = prediction_features.values
                X_scaled = scaler.transform(X)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                return prediction
            
            elif model_name == 'prophet':
                if not PROPHET_AVAILABLE:
                    raise ImportError("Prophet not installed")
                    
                model = self.models['prophet']
                
                # Prophet needs future dates
                future = model.make_future_dataframe(periods=1)
                forecast = model.predict(future)
                
                # Get the last prediction (next day)
                prediction = forecast['yhat'].iloc[-1]
                
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
            clean_data = clean_data.ffill().bfill()
            
            predictions = []
            dates = pd.date_range(start=clean_data.index[-1] + pd.Timedelta(days=1), periods=days)
            
            if model_name in ['linear_regression', 'random_forest', 'gradient_boosting', 'svr']:
                # Get the last known price
                last_price = clean_data['price'].iloc[-1]
                
                # Apply a more conservative approach to multi-day predictions
                for i in range(days):
                    try:
                        if i == 0:
                            # For first day, use the actual last data
                            next_price = self.predict_next_day(model_name, clean_data)
                        else:
                            # For subsequent days, add a more conservative trend
                            # This makes multi-day predictions more reasonable
                            if i == 1:
                                # First day change rate
                                pct_change = (next_price / last_price) - 1
                                
                                # Apply a dampening factor to reduce extreme predictions
                                if abs(pct_change) > 0.05:  # If change is more than 5%
                                    pct_change = pct_change * 0.5  # Dampen by 50%
                            else:
                                # Calculate a dampened change from previous predictions
                                pct_change = (predictions[-1] / predictions[-2]) - 1
                                pct_change = pct_change * 0.8  # Reduce effect over time
                            
                            # Add a small amount of randomness
                            random_factor = np.random.normal(0, 0.005)  # Reduced volatility
                            next_change = pct_change + random_factor
                            
                            # Apply the change to the previous prediction with limits
                            max_daily_change = 0.05  # Limit daily changes to 5%
                            next_change = max(-max_daily_change, min(next_change, max_daily_change))
                            
                            # Calculate next price
                            next_price = predictions[-1] * (1 + next_change)
                        
                        predictions.append(max(0.01, next_price))  # Ensure price is positive
                    except Exception as day_error:
                        print(f"Error predicting day {i+1}: {str(day_error)}")
                        # If error, use previous prediction with small adjustment
                        if predictions:
                            predictions.append(predictions[-1] * (1 + np.random.normal(0, 0.01)))
                        else:
                            predictions.append(last_price)
            
            elif model_name == 'prophet':
                if not PROPHET_AVAILABLE:
                    raise ImportError("Prophet not installed")
                    
                model = self.models['prophet']
                
                # Create future dataframe for prediction
                future = model.make_future_dataframe(periods=days)
                
                # Make forecast
                forecast = model.predict(future)
                
                # Get the predictions for the future days
                # Extract only the days we want to predict (last 'days' rows)
                future_predictions = forecast.iloc[-days:]['yhat'].values
                
                # Convert to series with dates
                return pd.Series(future_predictions, index=dates)
            
            # Make sure predictions is not empty before creating a pandas Series
            if not predictions:
                print(f"Warning: No predictions generated for {model_name}, using fallback")
                last_price = clean_data['price'].iloc[-1]
                predictions = [last_price * (1 + np.random.normal(0.001, 0.01)) for _ in range(days)]
            
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
    
    # Train Random Forest
    rf_model, _, _, _, _ = predictor.train_random_forest(df)
    
    # Make predictions
    last_data = df.iloc[-1:]
    
    # Predict next day price
    lr_prediction = predictor.predict_next_day('linear_regression', last_data)
    rf_prediction = predictor.predict_next_day('random_forest', last_data)
    
    print(f"Linear Regression prediction for next day: ${lr_prediction:.2f}")
    print(f"Random Forest prediction for next day: ${rf_prediction:.2f}")
    
    # Predict for next 7 days
    future_predictions = predictor.predict_multiple_days('random_forest', df, days=7)
    print("7-day forecast:")
    print(future_predictions)