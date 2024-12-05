import logging
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
logger = logging.getLogger('backend.model')

class ModelService:
    """
    ModelService contains methods for training, predicting and evaluating machine learning models.
    """

    def __init__(self):
        """
        Initialize ModelService.
        """
        pass

    def train(self, df, model, ticker, random_state=42, params=None, extended_training=False):
        """
        Train a machine learning model.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target variable.
        model : object
            Machine learning model.
        random_state : int
            Random state for training.
        params : dict
            Parameters for the model.
        extended_training : boolean
            Whether to use extended training (RandomizedSearchCV).

        Returns
        -------
        model : object
            Trained machine learning model.
        """
        params = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 10],
            "max_features": [0.5, 0.75, 1.0, 'sqrt', 'log2'],
            "bootstrap": [True, False]
        }
        rscv_params = {
            'n_iter': 100,
            'cv': 5
        }


        logger.info(f'Starting model...')
        y = df['Next Close']
        X = df.drop('Next Close', axis=1)
        if X.empty or y.empty:
            logger.critical('X, y, and model are required parameters.')
            raise ValueError('X, y, and model are required parameters.')
        
        if extended_training:
            logger.info(f'RandomizedSearchCV training selected.')
            if not rscv_params or not params:
                logger.critical('rscv_params and params are required parameters.')
                raise ValueError('rscv_params and params are required parameters.')
            trained_model = self.train_with_rscv(X, y, model, rscv_params, random_state, params)
        else:
            logger.info(f'Normal training selected.')
            trained_model = self.normal_train(X, y, model, random_state)

        logger.info(f'Model trained.') 


        return trained_model
    
    def normal_train(self, X, y, model, random_state=42):
        """
        Train a machine learning model normally.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target variable.
        model : object
            Machine learning model.
        random_state : int
            Random state for training.

        Returns
        -------
        model : object
            Trained machine learning model.
        """
        logger.info(f'Starting model training...') 

        start_time = time.time()
        model.fit(X, y)

        logger.info(f'Model trained in {time.time() - start_time:.2f} seconds.')

        return model
    
    def predict(self, model, X):
        """
        Predict using a machine learning model.

        Parameters
        ----------
        model : object
            Machine learning model.
        X : array
            Data to predict.

        Returns
        -------
        y_pred : array
            Predicted values.
        """
        return model.predict(X)
    
    def train_with_rscv(self, X, y, model, rscv_params, random_state=42, params=None):
        """
        Train a machine learning model using RandomizedSearchCV.

        Parameters
        ----------
        X : array
            Training data.
        y : array
            Target variable.
        model : object
            Machine learning model.
        rscv_params : dict
            Parameters for RandomizedSearchCV.
        random_state : int
            Random state for training.
        params : dict
            Parameters for the model.

        Returns
        -------
        model : object
            Trained machine learning model.
        """

        rs_model = RandomizedSearchCV(model, params, n_iter=rscv_params['n_iter'], cv=rscv_params['cv'], n_jobs=-1, random_state=random_state)

        logger.info(f'Starting RandomizedSearchCV model training...') 

        start_time = time.time()

        rs_model.fit(X, y)
        
        logger.info(f'Model trained in {time.time() - start_time:.2f} seconds.')

        model = rs_model
        return model
    
    def evaluate(self, model, X, y):
        """
        Evaluate a machine learning model.

        Parameters
        ----------
        model : object
            Machine learning model.
        X : array
            Evaluation data.
        y : array
            Target variable.

        Returns
        -------
        mse : float
            Mean squared error.
        accuracy : float
            Accuracy of the model.
        """
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        accuracy = model.score(X, y)
        return mse, accuracy


