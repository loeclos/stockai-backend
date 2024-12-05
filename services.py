import requests
import pandas as pd
from io import StringIO
import os
import json
import logging
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
load_dotenv()
logger = logging.getLogger('services')

class Services:
    def __init__(self):
        super().__init__()
        self.base_alpha_vatage_url = 'https://www.alphavantage.co/'
        self.alpa_vantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY')

    def download_equity_csv_file(self, ticker):
        """
        Download the CSV file for the given ticker symbol.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.

        Returns
        -------
        pandas.DataFrame
            The downloaded CSV data.
        """
        # Construct the URL for the Alpha Vantage API
        url = f'{self.base_alpha_vatage_url}/query'

        # Construct the parameters for the request
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': self.alpa_vantage_api_key,
            'datatype': 'csv',
        }

        # Send the request and get the response
        try: 
            response = requests.get(url, params=params)
        except Exception as e:
            logger.critical(f'Error downloading data for {ticker}: {e}')
            raise e


        # Read the response into a Pandas DataFrame
        df = pd.read_csv(StringIO(response.text))
        # Convert the 'Date' column to datetime

        # Process the DataFrame
        # ---------------------

        # Add a 'Date' column
        df['Date'] = df['timestamp']

        # Drop the 'timestamp' column
        df.drop('timestamp', axis=1, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'])
 
        # Sort the DataFrame by 'Date'
        df.sort_values(by=['Date'], inplace=True, ascending=True) 

        # Capitalize the column names
        for name, values in df.items():
            if name.capitalize() != name:
                df[name.capitalize()] = df[name]
                if name != 'Date':
                    df.drop(name, axis=1, inplace=True)

        # Return the DataFrame
        return df

    def preprocess_dataframe(self, df, columns_to_add=None, training=True):
        """
        Preprocess a DataFrame by doing the following steps:

        1. Drop the 'Adj Close' column if it exists
        2. Capitalize all column names
        3. Convert the 'Date' column to datetime
        4. Add new columns 'saleYear', 'saleMonth', 'saleDay', 'saleDayOfWeek', 'saleDayOfYear'
        5. Drop the 'Date' column
        6. Add a 'Next Close' column
        7. Convert all object columns to categorical columns
        8. For each numeric column, if the column contains missing values, add a binary column to indicate if sample had missing value and fill the missing values with the median of the column
        9. If columns_to_add is specified, add the columns to the DataFrame

        Parameters:
            df (pandas.DataFrame): The DataFrame to be preprocessed
            columns_to_add (list): The list of columns to add to the DataFrame

        Returns:
            A dictionary with the preprocessed DataFrame and a list of the missing columns that were added
        """
        missing_columns_added = []
        
        if 'Adj Close' in df.columns:
            df.drop('Adj Close', axis=1, inplace=True)
        
        for name, values in df.items():
            if name.capitalize() != name:
                df[name.capitalize()] = df[name]
                if name != 'Date':
                    df.drop(name, axis=1, inplace=True)
        if 'Date' in df.columns:
            df.sort_values(by=['Date'], inplace=True, ascending=True)
            df['Date'] = pd.to_datetime(df['Date'])

            df['saleYear'] = df.Date.dt.year
            df['saleMonth'] = df.Date.dt.month
            df['saleDay'] = df.Date.dt.day
            df['saleDayOfWeek'] = df.Date.dt.dayofweek
            df['saleDayOfYear'] = df.Date.dt.dayofyear

            df.drop('Date', axis=1, inplace=True)
        if training:
            df['Next Close'] = df['Close'].shift(-1)
            df = df.head(-1)
        for label, content in df.select_dtypes(include='object').items():
            if pd.api.types.is_string_dtype(content):
                df[label] = content.astype('category').cat.as_ordered()
            elif pd.api.types.is_object_dtype(content):
                df[label] = content.astype('category').cat.as_ordered()

        for label, content in df.items():
            if pd.api.types.is_numeric_dtype(content):
                if pd.isnull(content).sum():
                    df[f'{label}_is_missing'] = pd.isnull(content)
                    missing_columns_added.append(f'{label}_is_missing')
                    df[label] = content.fillna(content.median())

                if not pd.api.types.is_numeric_dtype(content):
                    # Add a binary column to indicate if sample had missing value
                    df[f'{label}_is_missing'] = pd.isnull(content)
                    missing_columns_added.append(f'{label}_is_missing')
                    df[label] = pd.Categorical(content).codes + 1

        if columns_to_add:
            for column in columns_to_add:
                if column in df.columns:
                    # If the column exists, do nothing
                    pass
                else:
                    # If the column doesn't exist, add it with a value of False
                    df[column] = False

        return {
            'dataframe': df, 
            'missing_columns_added': missing_columns_added
        }

    def check_if_equity_model_exists(self, tkr, file_path):
        """
        Check if a company model exists in the JSON file.

        Parameters:
            tkr (str): The ticker symbol of the company.
            file_path (str): The path to the JSON file containing company data.

        Returns:
            bool: True if the company model exists, False otherwise.
        """
        # Check if the file exists, if not, create an empty JSON file
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump([], file)

        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if the ticker exists in the data
        ticker_exists = any(item['tkr'] == tkr for item in data)

        return ticker_exists

    def find_equity_data(self, ticker, file_path='./stock_data.json'):
        """
        Search for a company's stock data using the Alpha Vantage API.

        Parameters:
            ticker (str): The ticker symbol of the company.

        Returns:
            str: The ticker symbol of the company if found, None otherwise.
        """


        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump([], file)


        with open(file_path, 'r') as file:
            stock_data = json.load(file)


        stock = next((stock for stock in stock_data if stock['tkr'] == ticker), None)

        if stock:
            return stock 
        else:
            return None
        
    def add_equity_model_data(self, data, file_path):
        """
        Add stock model data to the JSON file.

        Parameters:
            data (dict): The data to add to the JSON file.
            file_path (str): The path to the JSON file containing stock data.

        Returns:
            None
        """
        # Check if the file exists, if not, create an empty JSON file
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump([], file)

        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            existing_data = json.load(file)

        # Add the data to the existing data
        existing_data.append(data)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)


    def check_if_stock_exists(self, ticker):
        """
        Check if a stock exists in the Alpha Vantage database.

        Parameters:
            ticker (str): The ticker symbol of the stock.

        Returns:
            list: A list of dictionaries containing the information about the stock.
        """
                # Make a request to the Alpha Vantage API with the specified parameters
        response = requests.get(
            self.base_alpha_vatage_url + 'query', 
            params={
                'function': 'SYMBOL_SEARCH', 
                'keywords': ticker, 
                'apikey': self.alpa_vantage_api_key
            }
        )
        
        # Check if the response contains any matches and return the first match's symbol
        return response.json()['bestMatches'][0]['1. symbol'] if response.json()['bestMatches'] else None
        # return requests.get(f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={ticker}&apikey=V3GYHPPKWIHBSGRS').json()['bestMatches']



    # def get_current_equity_data(self, ticker):
    #     # Make the GET request
    #     response = requests.get(f'https://finance.yahoo.com/quote/VST/history/')

    #     # Pass the text content of the response to BeautifulSoup
    #     soup = BeautifulSoup(response.text, 'html.parser')

    #     # Locate the table
    #     table = soup.find('table', class_='table yf-j5d1ld')

    #     # Get the first <tr> element
    #     first_tr = table.tbody.tr if table and table.tbody else None

    #     if first_tr:
    #         print(first_tr)
    #     else:
    #         print(response.text)


