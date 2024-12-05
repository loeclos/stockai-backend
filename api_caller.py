import json
import os
import statistics
import requests
import datetime
import time

tracking_tkrs = ['AMZN']

def add_company_data(data, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump([], file)

    with open(file_path, 'r') as file:
        existing_data = json.load(file)  

    existing_data.append(data)

    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


# start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time = '2024-09-26 00:00:00'

# while True:
# time.sleep(3600)

# if int(datetime.datetime.now().strftime("%H")) >= 15:
#     break

for ticker in tracking_tkrs:
    response = requests.get(f'https://api.twelvedata.com/time_series?apikey=fec5830d86bf4f65a84a7283519ef000&interval=1h&symbol={ticker}&start_date={start_time}&end_date={datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}&format=JSON')
    data = response.json()

    predictions = []
    for data_entry in data["values"]:
        # Construct the POST request parameters
        params = {
            "ticker": f"{ticker}",
            "data_stop": "None",
            "random_seed": 42,
            "open": float(data_entry["open"]),
            "high": float(data_entry["high"]),
            "low": float(data_entry["low"]),
            "volume": float(data_entry["volume"]),
        }

        # Send POST request with JSON data
        headers = {"Content-Type": "application/json; charset=utf8"}
        prediction = requests.post(f"http://localhost:5000/predict", json=params, headers=headers)

        # Handle response
        predictions.append(prediction.json()["prediction"])

    # Calculate the average prediction
    average_prediction = sum(predictions) / len(predictions)

    # Add to data dict
    data_dict = {
        "ticker": f"{ticker}",
        "data": {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(data["values"][-1]["open"]),
            "high": float(data["values"][-1]["high"]),
            "low": float(data["values"][-1]["low"]),
            "volume": float(data["values"][-1]["volume"]),
        },
        "predictions": predictions,
        "Average predicted close": average_prediction,
        "Median predicted close": statistics.median(predictions),
        "Last predicted close": predictions[-1],
        "Actual close": float(data["values"][-1]["close"]),
    }

    add_company_data(data_dict, 'predictions.json')
    # time.sleep(60)


import smtplib, ssl

def get_tracker_data(ticker):
    with open('predictions.json', 'r') as file:
        data = json.load(file)
        for entry in data:
            if entry['ticker'] == ticker:
                return entry

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "robinsonkukuruzo@gmail.com"  # Enter your address
receiver_email = "robinsonkukuruzo@gmail.com"  # Enter receiver address
password = 'fxix zinp fkam seee' # input("Type your password and press enter: ")

message = f"""\
Subject: Stock Tracker Prediction for {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Here are the predictions for {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:
"""

for ticker in tracking_tkrs:
    tracker_data = get_tracker_data(ticker)

    message += f"Ticker: {ticker}\n"
    message += f"Time: {tracker_data['data']['time']}\n"
    message += f"Average predicted close: {tracker_data['Average predicted close']}\n"
    message += f"Median predicted close: {tracker_data['Median predicted close']}\n"
    message += f"Last predicted close: {tracker_data['Last predicted close']}\n"
    message += f"Actual close: {tracker_data['Actual close']}\n"
    message += "\n"




context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)



requests.get('http://localhost:5000/stopServer')