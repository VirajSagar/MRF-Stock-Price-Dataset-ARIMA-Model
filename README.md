# MRF-Stock-Price-Dataset-ARIMA-Model

About Dataset
Content
The date - "Date"

The opening price of the stock - "Open"

The high price of that day - "High"

The low price of that day - "Low"

The closed price of that day - "Close"

The number of stocks traded during that day - "Volume"

The stock's closing price has been amended to include any distributions/corporate actions that occur before the next day open - "Adj[usted] Close"

forecast = model.predict(n_periods=len(df_test))
forecast = pd.DataFrame(forecast,index=df_test.index,columns=['Prediction'])

sgt.plot_pacf(df_train,lags=40,zero=False, method=('ols'))

plt.plot(df_train, color="blue",label="Original Close")
plt.plot(df_test, color="green",label="Test")
plt.plot(forecast, color="red",label="forecast")
plt.title("MRF's stock prise")
plt.ylabel("Closing")
plt.xlabel("Date")
plt.legend(['train','test','forecast'])

from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(df_test,forecast))
print("RMSE: ", rms)

arima_model=ARIMA(df_train,order=(2,2,2))
result_arima=arima_model.fit()
result_arima.summary()

plt.plot(df_train, color="blue",label="MRF's stock price")
plt.plot(df_test, color="green",label="test")
plt.plot(forecast, color="red",label="Forecast")
plt.title("MRF'S stock price")
plt.ylabel("Close")
plt.xlabel("Date")
plt.legend(['train','test','forecast'])
