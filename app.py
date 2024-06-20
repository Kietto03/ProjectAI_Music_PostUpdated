import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive, TBATS
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from datasetsforecast.losses import mae, mape, rmse, smape
from io import BytesIO
import base64

# Set Matplotlib to use the Agg backend
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

path = "daily-website-visitors.csv"
df = pd.read_csv(path)
df.columns = [col.replace(".", "_") for col in df.columns]
df["Date"] = pd.to_datetime(df["Date"])
df["Page_Loads"] = df["Page_Loads"].replace(",", "", regex=True).astype("int16")
df["Unique_Visits"] = df["Unique_Visits"].replace(",", "", regex=True).astype("int16")
df["First_Time_Visits"] = df["First_Time_Visits"].str.replace(",", "", regex=True).astype("int16")
df["Returning_Visits"] = df["Returning_Visits"].str.replace(",", "", regex=True).astype("int16")
df.drop(columns=["Row", "Day", "Day_Of_Week"], inplace=True)
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

columns_to_forecast = ["Page_Loads", "Unique_Visits", "First_Time_Visits", "Returning_Visits"]

def transform_dataframe(df, target_column):
    df_transformed = df.reset_index()
    df_transformed.rename(columns={"Date": "ds", target_column: "y"}, inplace=True)
    df_transformed["unique_id"] = "1"
    return df_transformed

# Define HMM forecasting model with scaling
class HMMForecast:
    def __init__(self, n_components=4):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
        self.scaler = StandardScaler()
    
    def fit(self, df):
        self.scaled_data = self.scaler.fit_transform(df[['y']])
        self.model.fit(self.scaled_data)
    
    def predict(self, h, df):
        last_value = self.scaled_data[-1].reshape(1, -1)
        last_state = self.model.predict(last_value)[0]
        
        future_values = []
        for _ in range(h):
            next_value, _ = self.model.sample(1, currstate=last_state)
            future_values.append(next_value[0][0])
            last_state = self.model.predict(next_value)[0]
        
        future_values = self.scaler.inverse_transform(np.array(future_values).reshape(-1, 1)).flatten()
        
        prediction_dates = pd.date_range(start=df['ds'].values[-1], periods=h+1, freq='D')[1:]
        prediction_df = pd.DataFrame({
            'unique_id': '1',
            'ds': prediction_dates,
            'HMM': future_values
        })
        
        return prediction_df

def generate_forecasts(df, target_column, h=30):
    df_transformed = transform_dataframe(df[[target_column]], target_column=target_column)
    df_train = df_transformed[:-h]
    df_test = df_transformed[-h:]

    mstl = MSTL(season_length=[7, 12], trend_forecaster=AutoARIMA())
    hmm_model = HMMForecast()
    sf = StatsForecast(models=[mstl, SeasonalNaive(season_length=7), TBATS(seasonal_periods=7)], freq="D")
    sf = sf.fit(df=df_train)
    hmm_model.fit(df=df_train)

    forecasts_test = sf.predict(h=h)
    hmm_forecasts_test = hmm_model.predict(h=h, df=df_train)

    forecasts_test = forecasts_test.merge(hmm_forecasts_test, on=["unique_id", "ds"])
    return df_train, df_test, forecasts_test

def plot_forecasts(y_hist, y_true, y_pred, models, target_name):
    _, ax = plt.subplots(1, 1, figsize=(20, 7))
    y_true = y_true.merge(y_pred, how="left", on=["unique_id", "ds"])
    df_plot = pd.concat([y_hist, y_true]).set_index('ds').tail(30*5)
    df_plot[['y'] + models].plot(ax=ax, linewidth=2)
    
    ax.set_title(f"{target_name} Daily", fontsize=22)
    ax.set_ylabel(f"{target_name}", fontsize=20)
    ax.set_xlabel("Timestamp [t]", fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html', columns=columns_to_forecast)

@app.route('/forecast', methods=['POST'])
def forecast():
    column = request.form['column']
    df_train, df_test, forecasts_test = generate_forecasts(df, column)
    models = ["MSTL", "SeasonalNaive", "TBATS", "HMM"]
    plots = {model: plot_forecasts(df_train, df_test, forecasts_test, [model], target_name=column) for model in models}
    return render_template('forecast.html', plots=plots, column=column)

if __name__ == '__main__':
    app.run(debug=True)
