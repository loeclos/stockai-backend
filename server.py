import requests
from sklearn.ensemble import RandomForestRegressor
from services import Services
import signal
from flask import Flask, jsonify, request
import joblib
import pandas as pd
from datetime import date, timedelta, datetime
import os
from flask_cors import CORS
import logging
from model import ModelService
from dotenv import load_dotenv
from pymongo import MongoClient
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
MONGODB_PASSWORD = os.getenv('MONGODBPASSWORD')
BASE_LOG_DIRECTRY = './logs/'
EQUITY_DATA_FILE = './stock_data.json'

services = Services()
model_services = ModelService()

logging.basicConfig(filename=f'{BASE_LOG_DIRECTRY}{date.today()}.log', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.DEBUG)

logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("pymongo.topology").setLevel(logging.WARNING)


logger = logging.getLogger('server')

# MongoDB connection
client = MongoClient(f"mongodb+srv://admin:{MONGODB_PASSWORD}@cluster0.hqlch6b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["Stock-Prediction"]
previous_predictions_collection = db["previous-predictions"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        ticker = data.get('ticker')
        random_seed = data.get('random_seed')
        extended_training = data.get('extended_training')

        user_input_data = {
            'Date': date.today(),
            'Open': data.get('open'),
            'High': data.get('high'),
            'Low': data.get('low'),
            'Close': data.get('close'),
            'Volume': data.get('volume'),
        }
    except Exception as e:
        logger.error(f'Error getting client data: {e}')
        return jsonify({'error': 'Error processing request. Getting client data failed.'}), 400
    currrent_equity_data = services.find_equity_data(ticker.upper(), './stock_data.json')
    if currrent_equity_data:
        logger.info(f'Model for {ticker.upper()} already exists. Loading model...')
        model = joblib.load(f'./models/{ticker.upper()}.pkl')

        logger.info(f'Model loaded. Preprocessing user input...')
        user_input_df = pd.DataFrame(user_input_data, index=[0])
        user_input_df_processed = services.preprocess_dataframe(user_input_df, columns_to_add=currrent_equity_data['missing_columns_added'], training=False)['dataframe']

        logger.info(f'User input preprocessed. Making prediction...')
        try:
            equity_price_prediction = model.predict(user_input_df_processed)
            logger.info(f'Prediction for {ticker.upper()} succesful. Predicted price: {equity_price_prediction}')
            logger.info(f'Saving prediction for {ticker.upper()} into mongoDB...')
            previous_predictions_collection.insert_one({'id': str(uuid.uuid4()), 'ticker': ticker, 'prediction': equity_price_prediction[0], 'date': datetime.now(), 'random_seed': random_seed})
            return jsonify({'prediction': equity_price_prediction[0]}), 200
        except Exception as e:
            logger.error(f'Problems with making prediction for {ticker.upper()}. Error: {e}')
            return jsonify({'error': 'Error processing request. Making prediction failed.'}), 500
    else: 
        logger.info(f'Model for {ticker.upper()} does not exists. Preparing to train new model...')
        logger.info(f'Getting data for {ticker.upper()}...')
        if os.path.exists(f'./data/{ticker.upper()}.csv'):
            df = pd.read_csv(f'./data/{ticker.upper()}.csv', parse_dates=["Date"])
            equity_data_dictionary = services.preprocess_dataframe(df, columns_to_add=None)
            preprocessed_df = equity_data_dictionary['dataframe']
            logger.info(f'Data succesfully processed. Checking whether it is outadated...')
            current_year = datetime.now().year 
            if preprocessed_df.iloc[-1]["saleYear"] < (current_year - 1):
                logger.info(f'Data for {ticker.upper()} was last updated more than one year ago. Downloading new CSV data file for {ticker.upper()}...')
                try: 
                    downloaded_dataframe = services.download_equity_csv_file(ticker=ticker.upper())
                    downloaded_dataframe.to_csv(f'./data/{ticker.upper()}.csv', index=None)
                    logger.info(f'Data downloaded succesfully. Preprocessing for training...')
                    new_equity_data_dictionary = services.preprocess_dataframe(downloaded_dataframe, columns_to_add=None)
                    preprocessed_df = new_equity_data_dictionary['dataframe']
                    preprocessed_df_added_coloumns = new_equity_data_dictionary['missing_columns_added']
                    logger.info(f'Data for {ticker.upper()} processed for training. Adding entry into {EQUITY_DATA_FILE}')
                    services.add_equity_model_data({
                        'tkr': ticker,
                        'missing_columns_added': preprocessed_df_added_coloumns,
                        'model':f'{ticker}.pkl',
                        'year': current_year
                    }, EQUITY_DATA_FILE)
                except Exception as e:
                    logger.error(f'Problems with downloading data for {ticker.upper()}. Error: {e}')
                    return jsonify({'error': 'Downloading data from Alpha Vantage failed.'}), 500
            else:
                preprocessed_df_added_coloumns = equity_data_dictionary['missing_columns_added']
                logger.info(f'Data for {ticker.upper()} is up to date. Adding entry into {EQUITY_DATA_FILE}')
                services.add_equity_model_data({
                    'tkr': ticker,
                    'missing_columns_added': preprocessed_df_added_coloumns,
                    'model':f'{ticker}.pkl',
                    'year': current_year
                }, EQUITY_DATA_FILE)   

            logger.info(f'Data for {ticker.upper()} succesfully preprocessed and recorded into {EQUITY_DATA_FILE}. Beginning training...')
            trained_model = model_services.train(df=preprocessed_df, ticker=ticker, model=RandomForestRegressor(), random_state=random_seed, extended_training=extended_training)
            
            if trained_model:
                logger.info(f'Model for {ticker.upper()} trained successfully. Saving model...')
                joblib.dump(trained_model, f'./models/{ticker.upper()}.pkl')
                logger.info(f'Model for {ticker.upper()} saved successfully.')
            else:
                logger.error(f'Model for {ticker.upper()} failed to train.')
                return jsonify({'error': 'Error processing request. Model training failed.'}), 500
            
            logger.info(f'Using trained model to make prediction...')
            user_input_df = pd.DataFrame(user_input_data, index=[0])

            user_input_df_processed = services.preprocess_dataframe(user_input_df, columns_to_add=preprocessed_df_added_coloumns, training=False)['dataframe']
            try:
                equity_price_prediction = trained_model.predict(user_input_df_processed)
                logger.info(f'Prediction for {ticker.upper()} succesful. Predicted price: {equity_price_prediction}')
                logger.info(f'Saving prediction for {ticker.upper()} into mongoDB...')
                previous_predictions_collection.insert_one({'id': str(uuid.uuid4()), 'ticker': ticker, 'prediction': equity_price_prediction[0], 'date': datetime.now(), 'random_seed': random_seed})
                return jsonify({'prediction': equity_price_prediction[0]}), 200 
            except Exception as e:
                logger.error(f'Problems with making prediction for {ticker.upper()}. Error: {e}')
                return jsonify({'error': 'Error processing request. Making prediction failed.'}), 500
        else:
            logger.info(f'Data for {ticker.upper()} is missing. Downloading new CSV data file for {ticker.upper()}...')
            current_year = datetime.now().year 
            try: 
                downloaded_dataframe = services.download_equity_csv_file(ticker=ticker.upper())
                downloaded_dataframe.to_csv(f'./data/{ticker.upper()}.csv', index=None)
                logger.info(f'Data downloaded succesfully. Preprocessing for training...')
                new_equity_data_dictionary = services.preprocess_dataframe(downloaded_dataframe, columns_to_add=None)
                preprocessed_df = new_equity_data_dictionary['dataframe']
                preprocessed_df_added_coloumns = new_equity_data_dictionary['missing_columns_added']
                logger.info(f'Data for {ticker.upper()} processed for training. Adding entry into {EQUITY_DATA_FILE}')
                services.add_equity_model_data({
                    'tkr': ticker,
                    'missing_columns_added': preprocessed_df_added_coloumns,
                    'model':f'{ticker}.pkl',
                    'year': current_year
                }, EQUITY_DATA_FILE)
            except Exception as e:
                logger.error(f'Problems with downloading data for {ticker.upper()}. Error: {e}')
                return jsonify({'error': 'Downloading data from Alpha Vantage failed.'}), 500
            
            logger.info(f'Data for {ticker.upper()} succesfully preprocessed and recorded into {EQUITY_DATA_FILE}. Beginning training...')
            trained_model = model_services.train(df=preprocessed_df, ticker=ticker, model=RandomForestRegressor(), random_state=random_seed, extended_training=extended_training)
            
            if trained_model:
                logger.info(f'Model for {ticker.upper()} trained successfully. Saving model...')
                joblib.dump(trained_model, f'./models/{ticker.upper()}.pkl')
                logger.info(f'Model for {ticker.upper()} saved successfully.')
            else:
                logger.error(f'Model for {ticker.upper()} failed to train.')
                return jsonify({'error': 'Error processing request. Model training failed.'}), 500
            
            logger.info(f'Using trained model to make prediction...')
            user_input_df = pd.DataFrame(user_input_data, index=[0])

            user_input_df_processed = services.preprocess_dataframe(user_input_df, columns_to_add=preprocessed_df_added_coloumns, training=False)['dataframe']
            try:
                equity_price_prediction = trained_model.predict(user_input_df_processed)
                logger.info(f'Prediction for {ticker.upper()} succesful. Predicted price: {equity_price_prediction}')
                logger.info(f'Saving prediction for {ticker.upper()} into mongoDB...')
                previous_predictions_collection.insert_one({'id': str(uuid.uuid4()), 'ticker': ticker, 'prediction': equity_price_prediction[0], 'date': datetime.now(), 'random_seed': random_seed, 'specific_data': {'year': datetime.now().year, 'month': datetime.now().month, 'day': datetime.now().day}})
                return jsonify({'prediction': equity_price_prediction[0]}), 200
            except Exception as e:
                logger.error(f'Problems with making prediction for {ticker.upper()}. Error: {e}')
                return jsonify({'error': 'Error processing request.  Making prediction failed.'}), 500
        
   

@app.route("/ticker", methods=["POST"])
def check_ticker():
    data = request.get_json()
    ticker = data.get('ticker')

    if services.check_if_equity_model_exists(ticker, 'company_data.json'):
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})
    
# from bson.json_util import dumps

# @app.route('/autofill', methods=['POST'])
# def autofill():
#     data = request.get_json()
#     ticker = data.get('ticker')
#     response = requests.get(f'https://api.twelvedata.com/time_series?apikey=fec5830d86bf4f65a84a7283519ef000&interval=1min&outputsize=1&symbol={ticker}')
#     data = response.json()
#     return dumps(data['values'][0])

@app.route('/autofill', methods=['POST'])
def autofill():
    data = request.get_json()
    ticker = data.get('ticker')
    response = requests.get(f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')}/{datetime.today().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey=kN6j7cmDd_x_F8bLMMwq60yD8vBo8Px8')
    response_data = response.json()
    
    # Ensure the API response contains data
    if 'results' not in response_data or not response_data['results']:
        return {"error": "No data found for the ticker"}, 400
    
    latest_data = response_data['results'][0]
    print({
        "open": latest_data.get("o"),
        "low": latest_data.get("l"),
        "high": latest_data.get("h"),
        "volume": latest_data.get("v"),
        "close": latest_data.get("c"),
    })
    # Format the response to match frontend field names
    return jsonify({
        "open": latest_data.get("o"),
        "low": latest_data.get("l"),
        "high": latest_data.get("h"),
        "volume": latest_data.get("v"),
        "close": latest_data.get("c"),
    }), 200


@app.route('/get_all_predictions', methods=['GET'])
def get_all_predictions():
    """
    Get all predictions from the collection.
    
    Returns:
        list: A list of all predictions in the collection.
    """
    # Fetch all predictions from the collection
    predictions = list(previous_predictions_collection.find())
    
    # Convert the predictions list to a JSON-serializable format
    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])  # Convert ObjectId to string

    return jsonify(predictions[::-1]), 200 


@app.route('/delete_prediction', methods=['POST'])
def delete_prediction(): 
    """
    Delete a prediction from the collection by its ID.
    
    Parameters:
        id (str): The ID of the prediction to delete.
    
    Returns:
        dict: The deleted prediction.
    """
    data = request.json
    filter_criteria = {'id': data['id']}
    previous_predictions_collection.delete_one(filter_criteria)
    return jsonify(filter_criteria), 200
    
@app.route("/add_tracker", methods=["POST"])
def add_tracker():
    """
    Add a ticker symbol to the collection of trackers.
    
    Parameters:
        ticker (str): The ticker symbol to add.
    
    Returns:
        dict: Whether the ticker symbol already exists in the collection.
    """
    data = request.json

    ticker = data.get('ticker')
    if services.check_if_equity_model_exists(ticker, 'trackers.json'):
        return jsonify({'exists': True}), 409
    else:
        services.add_equity_model_data({'tkr': ticker}, 'trackers.json')
        return jsonify({'exists': False}), 200
    
@app.route('/stopServer', methods=['GET'])
def stopServer():
    """
    Stop the server.
    
    Returns:
        dict: A message indicating that the server is shutting down.
    """
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })



if __name__ == '__main__':
    app.run(debug=True)

