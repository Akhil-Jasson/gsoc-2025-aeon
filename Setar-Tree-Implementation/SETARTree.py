import numpy as np
from statsmodels.tsa.ar_model import AutoReg


class SETARTree:
    """
    Self-Exciting Threshold AutoRegressive (SETAR) Tree model.
   
    This model combines decision trees with autoregressive models by recursively partitioning
    the time series data based on threshold values, then fitting separate AR models in each segment.
    This allows for modeling of nonlinear time series with regime-switching behavior.
    """
    def __init__(self, lag=1, depth=1000, significance=0.05,
                 significance_divider=2, error_threshold=0.03,
                 stopping_criteria="both", mean_norm=False,
                 window_norm=False, categorical_covariates=None):
        """
        Initialize the SETAR Tree model with hyperparameters.
       
        Parameters:
        -----------
        lag : int
            Number of lagged values to use as predictors in the autoregressive models
        depth : int
            Maximum depth of the decision tree
        significance : float
            Initial significance level for splitting decisions
        significance_divider : float
            Factor by which significance is reduced at each tree level
        error_threshold : float
            Minimum error reduction required to justify a split
        stopping_criteria : str
            Criteria for stopping tree growth (currently not fully implemented)
        mean_norm : bool
            Flag for mean normalization of data (not currently implemented)
        window_norm : bool
            Flag for window-based normalization (not currently implemented)
        categorical_covariates : list
            List of categorical covariates (not currently implemented)
        """
        self.lag = lag
        self.depth = depth
        self.significance = significance
        self.significance_divider = significance_divider
        self.error_threshold = error_threshold
        self.stopping_criteria = stopping_criteria
        self.mean_norm = mean_norm
        self.window_norm = window_norm
        self.categorical_covariates = categorical_covariates
        self.tree = None
        self.residuals = []
        self.series_name = None


    def fit(self, data, label, series_name="default_series"):
        """
        Fit the SETAR Tree model to the given time series data.
       
        Parameters:
        -----------
        data : array-like
            Input time series data
        label : array-like
            Target values (usually the same as data, but potentially transformed)
        series_name : str
            Name identifier for the time series
           
        Returns:
        --------
        self : SETARTree
            The fitted model instance
        """
        self.series_name = series_name
        X = self._create_features(data)
        y = label[self.lag:]  # Shift target values to align with lagged features
        self.tree = self._build_tree(X, y)
        return self


    def _create_features(self, data):
        """
        Create lagged features from the time series data.
       
        Parameters:
        -----------
        data : array-like
            Input time series data
           
        Returns:
        --------
        X : ndarray
            Matrix of lagged features where each row represents a time point
            and columns represent lags (t-1, t-2, ..., t-lag)
        """
        if len(data) < self.lag:
            raise ValueError(f"Data length ({len(data)}) must be >= lag ({self.lag})")
        return np.array([data[i:-self.lag+i] for i in range(self.lag)]).T


    def _calculate_error(self, y, model=None):
        """
        Calculate the mean squared error of an autoregressive model.

        Parameters:
        -----------
        y : array-like
            Target values.
        model : AutoReg fitted model, optional
            Pre-fitted AutoReg model. If None, a new model is fitted.

        Returns:
        --------
        error : float
            Mean squared error (sum of squared residuals divided by sample size).
        """
        model = AutoReg(y, lags=self.lag).fit()
        # if model is None:
        #     model = AutoReg(y, lags=self.lag).fit()
        residuals = model.resid  # Get residuals
        ssr = np.sum(residuals**2)  # Calculate sum of squared residuals
        abs_residuals = np.abs(residuals)  # Absolute residuals
        n = len(y)  # Number of observations
        p = self.lag + 1  # Number of parameters (lags + intercept)
        # Mean Squared Error (MSE)
        mse = ssr / (n - p) if n > p else float('inf')

        # Mean Absolute Error (MAE)
        mae = np.mean(abs_residuals)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Mean Absolute Scaled Error (MASE)
        naive_forecast_errors = np.abs(np.diff(y))  # Naive forecast errors (lag-1)
        naive_mae = np.mean(naive_forecast_errors) if len(naive_forecast_errors) > 0 else float('inf')
        mase = mae / naive_mae if naive_mae > 0 else float('inf')

        # Return all error metrics
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MASE': mase
        }





    def _find_best_threshold(self, X, y):
        """
        Find the optimal threshold value for splitting the data.

        Samples 10 potential threshold values evenly spaced across the range of the
        first lag feature, and chooses the one that minimizes prediction error
        while ensuring a minimum error reduction.

        Parameters:
        -----------
        X : ndarray
            Matrix of lagged features.
        y : array-like
            Target values.

        Returns:
        --------
        best_threshold : float or None
            Optimal threshold value, or None if no valid split is found.
        """
        best_threshold = None
        best_error = float('inf')
        parent_error = self._calculate_error(y)['MSE'] # Use MSE as the primary metric

        for threshold in np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10):
            left_mask = X[:, 0] < threshold
            right_mask = ~left_mask

            # Ensure minimum sample size in each split
            if np.sum(left_mask) < self.lag * 2 or np.sum(right_mask) < self.lag * 2:
                continue

            # Calculate errors for each split
            left_error = self._calculate_error(y[left_mask])['MSE']  # Use MSE for left split
            right_error = self._calculate_error(y[right_mask])['MSE']  # Use MSE for right split

            # Calculate weighted average error
            combined_error = (left_error * len(y[left_mask]) +
                              right_error * len(y[right_mask])) / len(y)

            # Check if split provides sufficient error reduction
            error_reduction = parent_error - combined_error
            print(error_reduction, parent_error, combined_error)
            
            if error_reduction > self.error_threshold and combined_error < best_error:
                best_error = combined_error
                best_threshold = threshold
        print(f"Threshold: {threshold}, Combined Error: {combined_error}, Error Reduction: {error_reduction}")
        return best_threshold


    def _build_tree(self, X, y, depth=0, current_significance=None):
        """
        Recursively build the SETAR tree.
       
        Parameters:
        -----------
        X : ndarray
            Matrix of lagged features
        y : array-like
            Target values
        depth : int
            Current depth in the tree
        current_significance : float
            Current significance level (decreases with depth)
           
        Returns:
        --------
        node : dict
            Tree node representation, either an internal node with left/right children
            or a leaf node with an AutoReg model
        """
        if current_significance is None:
            current_significance = self.significance


        # Stopping criteria - create leaf node with AR model
        if (depth >= self.depth or
            len(y) < self.lag*2 or
            current_significance < 0.001):
            model = AutoReg(y, lags=self.lag).fit()
            self.residuals.extend(model.resid)
            return {'model': model}


        # Try to find optimal split
        threshold = self._find_best_threshold(X, y)
        if threshold is None:
            # No valid split found - create leaf node
            model = AutoReg(y, lags=self.lag).fit()
            self.residuals.extend(model.resid)
            return {'model': model}


        # Create split node
        left_mask = X[:, 0] < threshold
        right_mask = ~left_mask


        # Update significance for next level (increasing strictness with depth)
        new_significance = current_significance / self.significance_divider


        # Create internal node and recursively build subtrees
        return {
            'threshold': threshold,
            'significance': current_significance,
            'left': self._build_tree(X[left_mask], y[left_mask], depth+1, new_significance),
            'right': self._build_tree(X[right_mask], y[right_mask], depth+1, new_significance)
        }


    def predict(self, X, confidence_level=90, series_name=None):
        """
        Predict using either raw time series or preprocessed features.
       
        Parameters:
        -----------
        X : array-like
            1D time series or 2D matrix of lagged features
        confidence_level : int (0-100)
            Confidence level for prediction intervals
        series_name : str
            Optional override for output identification
           
        Returns:
        --------
        result : dict
            Dictionary containing predictions and confidence intervals:
            - method: Model identification
            - series: Series name
            - x: Input data
            - mean: Point predictions
            - upper: Upper confidence bounds
            - lower: Lower confidence bounds
            - level: Confidence level
        """
        # Convert input to numpy array
        X_original = np.asarray(X)
       
        # Handle 1D input (time series)
        if X_original.ndim == 1:
            if len(X_original) < self.lag:
                raise ValueError(f"Time series length ({len(X_original)}) must be >= lag ({self.lag})")
            X_processed = self._create_features(X_original)
        else:
            X_processed = X_original


        # Generate predictions by traversing the tree for each input vector
        predictions = []
        for x in X_processed:
            node = self.tree
            # Navigate through tree based on threshold comparisons
            while 'threshold' in node:
                if x[0] < node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            # Get prediction from leaf node's AR model
            model = node['model']
            pred = model.predict(start=len(model.params), end=len(model.params))[0]
            predictions.append(pred)


        # Calculate confidence intervals using normal approximation
        normal_samples = np.random.normal(0, 1, 100000)
        z_score = np.abs(np.percentile(normal_samples,
                                     (1 + confidence_level/100) / 2 * 100))
        std_error = np.std(self.residuals)
       
        mean_preds = np.array(predictions)
        lower = mean_preds - z_score * std_error
        upper = mean_preds + z_score * std_error


        # Return formatted results
        return {
            'method': 'SETAR_TREE',
            'series': series_name if series_name else self.series_name,
            'x': X_original.tolist(),
            'mean': mean_preds.tolist(),
            'upper': upper.tolist(),
            'lower': lower.tolist(),
            'level': confidence_level
        }

