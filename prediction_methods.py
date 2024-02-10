import json
import urllib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly

one_year = 24 * 365
one_month = 24 * 30
one_week = 24 * 7
one_day = 24


def process_input_data(dt):
  data_input = dt.copy()
  data_input['Godz.'] -= 1
  time = [datetime.strptime(str(x), '(%Y%m%d, %H)') for x in zip(data_input['Data'], data_input['Godz.'])]
  day_of_the_year = [datetime.strptime(str(x), '(%Y%m%d, %H)').timetuple().tm_yday for x in zip(data_input['Data'], data_input['Godz.'])]
  day_of_the_week = [x.weekday() for x in time]
  data_input['time'] = time
  data_input['day_of_the_year'] = day_of_the_year
  data_input['day_of_the_week'] = day_of_the_week

  data = pd.DataFrame()

  data['time'] = data_input['time']
  data['day_of_the_week'] = data_input['day_of_the_week'].astype(np.float32)
  data['current_energy_demand'] = data_input['Rzeczywiste zapotrzebowanie KSE'].astype(np.float32)/100000
  data['hour'] = data_input['Godz.'].astype(np.float32)
  data['day_of_the_year'] = data_input['day_of_the_year'].astype(np.float32)
  data['RCE'] = data_input['RCE'].astype(np.float32)/100000

  data = data.set_index('time')
  return data


def build_df(data):
  one_year = 24 * 365
  one_month = 24 * 30
  one_week = 24 * 7
  one_day = 24

  mean_energy_demant_last_year = np.zeros(len(data))
  mean_energy_demant_last_month = np.zeros(len(data))
  mean_energy_demant_last_week = np.zeros(len(data))
  mean_energy_demant_last_day = np.zeros(len(data))

  rce_last_year = np.zeros(len(data))
  rce_last_month = np.zeros(len(data))
  rce_last_week = np.zeros(len(data))
  rce_last_day = np.zeros(len(data))

  temperature_last_year = np.zeros(len(data))
  temperature_last_month = np.zeros(len(data))
  temperature_last_week = np.zeros(len(data))
  temperature_last_day = np.zeros(len(data))

  for i in range(one_year, len(data)):
    mean_energy_demant_last_year[i] = np.mean(data['current_energy_demand'][i:i+one_year])
    rce_last_year[i] = np.mean(data['RCE'][i:i+one_year])
    temperature_last_year[i] = np.mean(data['temperature'][i:i+one_year])

  for i in range(one_month, len(data)):
    mean_energy_demant_last_month[i] = np.mean(data['current_energy_demand'][i:i+one_month])
    rce_last_month[i] = np.mean(data['RCE'][i:i+one_month])
    temperature_last_month[i] = np.mean(data['temperature'][i:i+one_month])

  for i in range(one_week, len(data)):
    mean_energy_demant_last_week[i] = np.mean(data['current_energy_demand'][i:i+one_week])
    rce_last_week[i] = np.mean(data['RCE'][i:i+one_week])
    temperature_last_week[i] = np.mean(data['temperature'][i:i+one_week])

  for i in range(one_day, len(data)):
    mean_energy_demant_last_day[i] = np.mean(data['current_energy_demand'][i:i+one_day])
    rce_last_day[i] = np.mean(data['RCE'][i:i+one_day])
    temperature_last_day[i] = np.mean(data['temperature'][i:i+one_day])

  new_df = pd.DataFrame(index=data.index)
  new_df['mean_energy_demant_last_year'] = mean_energy_demant_last_year
  new_df['mean_energy_demant_last_month'] = mean_energy_demant_last_month
  new_df['mean_energy_demant_last_week'] = mean_energy_demant_last_week
  new_df['mean_energy_demant_last_day'] = mean_energy_demant_last_day
  new_df['current_energy_demand'] = data['current_energy_demand']

  new_df['mean_rce_last_year'] = rce_last_year
  new_df['mean_energy_demant_last_month'] = rce_last_month
  new_df['mean_energy_demant_last_week'] = rce_last_week
  new_df['mean_energy_demant_last_day'] = rce_last_day
  new_df['RCE'] = data['RCE']

  new_df['temperature_last_year'] = temperature_last_year
  new_df['temperature_last_month'] = temperature_last_month
  new_df['temperature_last_week'] = temperature_last_week
  new_df['temperature_last_day'] = temperature_last_day
  new_df['temperature'] = data['temperature']

  new_df['hour'] = data['hour']
  new_df['day_of_the_week'] = data['day_of_the_week']
  new_df['day_of_the_year'] = data['day_of_the_year']
  new_df['rain'] = data['rain']
  new_df['snowfall'] = data['snowfall']
  new_df['snow_depth'] = data['snow_depth']
  new_df['weathercode'] = data['weathercode']

  return new_df


def convertToMatrix(data, step):
    X =[]
    for i in range(len(data)-step):
        d=i+step
        X.append(data[i:d,])
    return np.array(X)


def predict_annualy(date_to_predict, data, model, step=24):
  pred_day_start = datetime.strptime(str(date_to_predict), "%Y-%m-%dT%H:%M")
  pred_day_start = pred_day_start.replace(minute=0, second=0)
  prd = pred_day_start - timedelta(hours=24)
  trainX = data[prd:pred_day_start].to_numpy()
  trainX = convertToMatrix(trainX, step)
  trainPredict = model.predict(trainX).reshape((one_year,))
  return trainPredict


def normalize(df):
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result


def predict_annualy_plot(date_to_predict, model, data, step=24):
  pred_day_start = datetime.strptime(str(date_to_predict), "%Y-%m-%dT%H:%M")
  pred_day_start = pred_day_start.replace(minute=0, second=0)
  start_index = data.index.get_loc(pred_day_start) + 1
  # trainY = data[['current_energy_demand']].iloc[start_index:start_index + one_year].to_numpy().reshape((one_year,))
  trainPredict = predict_annualy(date_to_predict, data, model, step)

  # x = data.iloc[start_index:start_index + one_year].index.values
  x = [pred_day_start + timedelta(hours=i) + timedelta(hours=1) for i in range(one_year)]

  trace_pred = go.Scatter(x=x, y=trainPredict * 100000, mode='lines', name='Prediction')
  # trace_real = go.Scatter(x=x, y=trainY * 100000, mode='lines', name='Real', line=dict(dash='dash'))

  # data = [trace_pred, trace_real]
  data = [trace_pred, ]

  layout = go.Layout(
      title=f'Data from {pred_day_start.strftime("%Y-%m-%d %H:%M")}',
      xaxis=dict(title='Date'),
      yaxis=dict(title='Value'),
  )

  fig = go.Figure(data=data, layout=layout)
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

  return graphJSON