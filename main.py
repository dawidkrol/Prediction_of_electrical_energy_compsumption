import urllib
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, request
from keras.models import load_model
import json
import os
import tempfile
from prediction_methods import predict_annualy_plot, predict_annualy, build_df, process_input_data

app = Flask(__name__, template_folder='views')

one_year = 24 * 365
one_month = 24 * 30
one_week = 24 * 7
one_day = 24


def get_data():
    data_2018 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vQyLPKm_djdrL4Cz-w-OsQ2QJUjiRqA-gSW-D1BLX09bF2OAdf7fjqDZpMW1IouxiV8GiY-5Reg11uL/pub?output=csv',
        decimal=",", thousands=",")
    data_2019 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vTl3NavnK7Emg1QEHIXdLLK5NfpeDzZts8qgu30g3Wlg_3yjIdB5Vf3XmUHMKNIFbMl2zq9XfCb3772/pub?output=csv',
        decimal=",", thousands=",")
    data_2020 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vSK8d9mhk0H-SgL38vWJgFpT7_Vf9VxaemoQXN3PJ02Q3EU9l0CeD6VLfzhCIajD0upPHDK1oGe2Jnh/pub?output=csv',
        decimal=",", thousands=",")
    data_2021 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vSAuH8mfWbcpOzhf-7PMBAoCwXg9bp6IxD7gJrkDJb7UNtDA4q4h0Izlyd-R9PNYK4hSzpN2GNBLrbf/pub?output=csv',
        decimal=",", thousands=",")
    data_2022 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vTGGEkWCU94jHUis6JTdUogvNG4NYLqWURqQLf3cPdyRCYbAmBcUDYjEyDDKmiMms4OE1Hz-SK1ilpp/pub?output=csv',
        decimal=",", thousands=",")
    data_2023 = pd.read_csv(
        'https://docs.google.com/spreadsheets/d/e/2PACX-1vRsJOwo51grLVtH3HbS0tkckdX7uWkULxgRm8SAtBgPEkYgyjUDFqVGRU77Rrz7o5PqS0RU7q1SbCBo/pub?output=csv',
        decimal=",", thousands=",")

    data = pd.concat([process_input_data(data_2018), process_input_data(data_2019), process_input_data(data_2020),
                      process_input_data(data_2021), process_input_data(data_2022),
                      process_input_data(data_2023)]).sort_values(by=['time'])

    weather_2021_2022_api = urllib.request.urlopen(
        'https://archive-api.open-meteo.com/v1/archive?latitude=50.064651&longitude=19.944981&start_date=2018-01-01&end_date=2024-01-09&hourly=temperature_2m,rain,snowfall,snow_depth,weathercode&timezone=Europe%2FBerlin').read()
    op = json.loads(weather_2021_2022_api)
    weather = pd.DataFrame(
        [op['hourly']['time'], op['hourly']['temperature_2m'], op['hourly']['rain'], op['hourly']['snowfall'],
         op['hourly']['snow_depth'], op['hourly']['weathercode']]).T
    weather.columns = ['time', 'temperature', 'rain', 'snowfall', 'snow_depth', 'weathercode']
    weather['time'] = pd.to_datetime(weather['time'])
    weather['temperature'] = weather['temperature'].astype(np.float32)
    weather['rain'] = weather['rain'].astype(np.float32)
    weather['snowfall'] = weather['snowfall'].astype(np.float32)
    weather['snow_depth'] = weather['snow_depth'].astype(np.float32)
    weather['weathercode'] = weather['weathercode'].astype(np.float32)
    weather = weather.set_index('time')

    data = data.join(weather)
    data = build_df(data)
    return data


# data = get_data()
# data.to_pickle("data.pkl")
data = pd.read_pickle('data/data.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model(f'models/annual.h5')

    date_to_predict = request.form['date']
    step = 24
    graphJSON = predict_annualy_plot(date_to_predict, model, data, step)

    return render_template('index.html', graphJSON=graphJSON)


@app.route('/download', methods=['POST'])
def download():
    model = load_model('models/annual.h5')

    date_to_predict = request.form['date']
    step = 24

    pred_day_start = datetime.strptime(str(date_to_predict), "%Y-%m-%dT%H:%M")
    pred_day_start = pred_day_start.replace(minute=0, second=0)
    predicted_data = predict_annualy(date_to_predict, data, model, step)

    predicted_data = [float(value) for value in predicted_data]

    result_list = []
    for i, value in enumerate(predicted_data):
        current_date = (pred_day_start + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:%M")
        result_list.append({'date': current_date, 'value': value})

    path = os.path.join(tempfile.gettempdir(), "prediction.json")
    with open(path, "w") as f:
        json.dump({'data': result_list}, f)

    return send_file(path, as_attachment=True)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
