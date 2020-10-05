#!/usr/bin/env python
# coding: utf-8

# Mean Sea Level Time Series Analysis for Mumbai (India)
# Author: Javed Ali


# Import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from statsmodels.tsa.filters.hp_filter import hpfilter

import seaborn as sns
import plotly
from plotly import graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels

import scipy.stats as st
import scipy.io as sio

from datetime import datetime


# a.) Calculate the linear trend (in $mm/year$) and 99% confidence interval for the entire data record


# Load the time series data
data = pd.read_excel("Sea level data/Mumbai_India.xlsx", 
                               header=0, 
                               index_col=[0],
                               parse_dates=[0])



t = data.index.values #time
s = data['sea_level'] #sea level
L = len(data)



# Check the time series data
data.head()



# Summary of the data
data.describe(percentiles=[0.99, 0.25, 0.50, 0.75, 0.90])


# Plotting the original time series data

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index.values, 
    y=data.sea_level, 
    line_color='deepskyblue'))

fig.update_layout(
    template = "plotly_white",
    title_text="Original Mean Sea Level Time Series",
    xaxis_rangeslider_visible=True)

fig.update_xaxes(title_text="Time (years)")
fig.update_yaxes(title_text="Mean Sea Level (mm)")

plotly.offline.plot(fig, filename='0 Original MSL TS.html')

fig.show()


# Create a matrix with the time information (t is years)
matrix = (t, np.ones((L,1)))

# Build a model based on y = ax + b + Error
t_mean= np.mean(t)
s_mean= np.mean(s)


t_mean, s_mean


# METHOD 1: Linear regression fit using `Numpy's polynomial fit` function


# Numpy's polynomial fit function
parameters = np.polyfit(data.index.values, data['sea_level'], deg = 1) #Linear trend is a polynomial of degree 1
(a,b) = parameters
trend = a*data.index.values + b
print('The linear parameters are, a =', a, 'b =', b)

fig = plt.figure(figsize=(16,8))
plt.plot(data.index.values, data['sea_level'], color='blue')
plt.plot(data.index.values, trend, label='Linear Trend', color ='red')
plt.legend(loc='upper left')

plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Mean Sea Level (mm)", fontsize=14)
plt.title("Linear Trend Fit", fontsize=20)
plt.savefig('1 Linear trend fit', dpi=300)
plt.show()


print(trend)


# METHOD 2: Least Squares trend fit using `Numpy`

#Rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]

A = np.vstack([data.index.values, np.ones(L)]).T
m,c = np.linalg.lstsq(A,data['sea_level'], rcond=None)[0]
print('Least Squares parameters are, m =', m, 'c =', c)

line = m*data.index.values+c

fig = plt.figure(figsize=(16,8))
plt.plot(data.index.values, data['sea_level'])
plt.plot(data.index.values, line, label='Least Squares Trend Line', color='red')
plt.legend(loc='best')

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Mean Sea Level (mm)", fontsize=12)
plt.title("Least Squares Trend Fit", fontsize=20)

plt.savefig('2 Least Squares Trend Fit', dpi=300)


print(line)


# METHOD 3: Ordinary Least Square (OLS) trend fit using `plotly`


fig = px.scatter(data, x=data.index.values, y=data["sea_level"], trendline="ols", trendline_color_override='red', 
                 labels={"x": "Time (years)",  "sea_level": "Mean Sea Level (mm)"})

fig.update_layout(title='Ordinary Least Square (OLS) Trend Fit')

plotly.offline.plot(fig, filename='3 OLS linear trend fit.html')

fig.show()


# Check the results of OLS trend fit
results = px.get_trendline_results(fig)
print(results)

results.px_fit_results.iloc[0].summary()


# METHOD 4: Ordinary Least Squares (OLS) linear regression trend fit using `statsmodels`

# Ordinary Least Squares (OLS) linear regression model 

Y = data['sea_level']
X = data.index.values
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()


# Calculate confidence intervals (CI)


# define response variable
y = data['sea_level']

#define predictor variables
x = data.index.values

# add constant to predictor variables
x = sm.add_constant(x)

# fit OLS linear regression model
model = sm.OLS(y, x).fit()

# view model summary
print(model.summary())

print('\n')
# Calculate 99% confidence interval
CI99 = model.conf_int(0.01)
print(CI99)

ci_99 = np.min(CI99)[1] - a
print("The 99% confidence interval of the trend for the entire time series is +-", ci_99, "mm/year")


# b. Calculate the acceleration (in $mm/year^2$) for the entire data record (note that acceleration is defined as twice the quadratic coefficient!)

# Quadratic trend using `Numpy's polyfit` function

# Quadratic trend is a polynomial of degree 2: $ax^2 + bx + c$


# Quadratic trend 
para_quad = np.polyfit(t,s, deg = 2) 
(aq,bq,cq) = para_quad
acceleration = 2*aq
print('The acceleration of the sea level rise in Mumbai is', acceleration, 'mm/year^2')

quad_trend = aq*(t**2) + bq*t +cq
print('The quadratic parameters are', para_quad)

f = plt.figure(figsize=(16,8))

#plt.plot(data.index.values, data['sea_level'], label='Original', color='grey')

plt.plot(t, quad_trend, label ='Quadratic Trend', color='blue', linewidth=2)
plt.title('Quadratic trend fit', fontsize=20)
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Mean Sea Level (mm)', fontsize=12)
plt.legend()
plt.savefig('4 Quadratic trend', dpi=300)

plt.show()


#polynomial fit with degree = 2 (Quadratic fit)
model = np.poly1d(np.polyfit(data.index.values, data["sea_level"], 2))

print('The quadratic equation is: \n', model)

#add fitted polynomial line to scatterplot
f = plt.figure(figsize=(16, 12))
polyline = np.linspace(1880, 2000, 10)
plt.scatter(data.index.values, data["sea_level"], color="red", label='Original')
plt.plot(polyline, model(polyline), label='Quadratic Trend', linewidth=3)

plt.title('Quadratic trend fit', fontsize=20)
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Mean Sea Level (mm)', fontsize=12)
plt.legend(loc='best')
plt.savefig('5 Quadratic trend with original TS', dpi=300)
plt.show()


# c.) Calculate the linear trends (in $mm/year$) for the first and second halves of the time series

#first half of the time series
data_1 = data.iloc[0:696, 0:1]


t1 = data_1.index.values
s1 = data_1['sea_level']

(a1, b1) = np.polyfit(t1, s1, deg=1)
trend1 = a1*t1 + b1

print('The linear trend from 1878-1935 is', a1, 'mm/year')


#second half of the time series
data_2 = data.iloc[696:1393, 0:1]

t2 = data_2.index.values
s2 = data_2['sea_level']

(a2, b2) = np.polyfit(t2, s2, deg=1)
trend2 = a2*t2 + b2

print('The linear trend from 1936-1993 is', a2, 'mm/year')


# Plot the two halves of the time series data

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data_1.index.values, 
    y=data_1.sea_level, 
    name='TS-1 (1878-1935)',
    line_color='deepskyblue'))

fig.add_trace(go.Scatter(
    x=data_2.index.values, 
    y=data_2.sea_level,
    name='TS-2 (1936-1993)',
    line_color='red'))

fig.update_layout(
    template = "plotly_white",
    title_text="Mean Sea Level for Mumbai (India)",
    xaxis_rangeslider_visible=True)

fig.update_xaxes(title_text="Time (years)")
fig.update_yaxes(title_text="Mean Sea Level (mm)")

plotly.offline.plot(fig, filename='6 Two Halves TS.html')

fig.show()


# d.) Based on the 95% confidence levels determine if the trends derived in (c) are significantly different from each other (i.e. confidence levels do not overlap).

# Trend in TS-1: Confidence interval for TS-1 (1878-1935)


#define response variable
y1 = data_1['sea_level']

#define predictor variables
x1 = data_1.index.values

#add constant to predictor variables
x1 = sm.add_constant(x1)

#fit OLS linear regression model
model1 = sm.OLS(y1, x1).fit()

#view model summary
print(model1.summary())

#95% confidence interval
CI95_1 = model1.conf_int(0.05)
print(CI95_1)


# Confidence interval for TS-2 (1936-1993)

#define response variable
y2 = data_2['sea_level']

#define predictor variables
x2 = data_2.index.values

#add constant to predictor variables
x2 = sm.add_constant(x2)

#fit OLS linear regression model
model2 = sm.OLS(y2, x2).fit()

#view model summary
print(model2.summary())

#95% confidence interval
CI95_2 = model2.conf_int(0.05)
print(CI95_2)


# e.) Calculate the amplitude (in $cm$) of the average annual cycle for the entire period and identify the month when it peaks (remove the linear trend from (a) for detrending)


# Detrend Time Series Data: remove the linear trend from original time series
detrend = s - trend


detrend.head()


# Plotting detrended time series

f = plt.figure(figsize=(16,8))
detrend.plot()
plt.xlim(1860, 2000)
plt.xlabel('Time (years)', fontsize=14)
plt.ylabel('Mean Sea Level Change (mm)', fontsize=14)
plt.title('Detrended Mean Sea Level Time Series for Mumbai (India)', fontsize=20, loc='left')

plt.savefig('7 Detrended TS', dpi=300)


# Create a row for each month (12) and a column for each year
mo = detrend.values.reshape(12, int(L/12)) #float to integer


# Seasonal (Annual) Cycle by averaging by months (all the Januaries, Febs, etc.)
cycle = mo.mean(axis=1) #average the rows


max(cycle), min(cycle)


#Calculate the amplitude
amp = (max(cycle)-min(cycle))/2

print('The amplitude of the seasonal cycle is', amp/10, 'cm')
print('On average, the sea level in Mumbai peaks in August at', max(cycle)/10, 'cm')


max_month0 = np.argmax(cycle, axis=0)

print(max_month0)


# f.) Same as (e) but separately for the first 5 years of the record and last 5 years of the record (note: use the same detrended time series as in (e), DO NOT detrend the first/last five years of data again)

# First 5 years
f5cycle = mo[:,0:5].mean(axis=1) # all the rows:(months), first 5 columns (years)
f5amp = (max(f5cycle)-min(f5cycle))/2

print('The amplitude of the average annual cycle from 1878-1882 is', f5amp/10, 'cm')
print('The average sea level during the first 5 years of the record peaks in the month of May at', np.max(f5cycle), 'mm.')


max_month = np.argmax(f5cycle, axis=0)

print(max_month)


# Last 5 years
l5cycle = mo[:,-5:].mean(axis=1) #last 5 columns
l5amp = (max(l5cycle)-min(l5cycle))/2

print('The amplitude of the last 5 years of the sea level record in Mumbai is', l5amp/10, 'cm')
print('The average sea level during the last 5 years of the record peaks in August at', np.max(l5cycle), 'mm.')


max_month2 = np.argmax(l5cycle, axis=0)

print(max_month2)

print('First five years: ', f5cycle)

print('Last five years: ', l5cycle)


f = plt.figure(figsize=(16,8))

plt.plot(f5cycle, color='blue', label='First 5-year cycle')
plt.plot(l5cycle, color='red', label='Last 5-year cycle')

plt.legend(loc='best')
plt.title('First and last 5-year annual cycle', fontsize=20)
plt.xlabel('Time (month)', fontsize=14)
plt.ylabel('Mean Sea Level Change (mm)', fontsize=14)
plt.savefig('8 first & last 5-year cycle', dpi=300)


# g.) Identify the largest (positive or negative) monthly anomaly above or below the average seasonal cycle (Note: create a time series of the average seasonal cycle and subtract it from the raw data with linear trend removed)

# repeat the annual cycle for the entire record (to match the dimensions of t years)
# 116 years of data, so repeat seasonal cycle 116 times
rep_cycle = np.tile(cycle, int(L/len(cycle)))


fig=plt.figure(figsize=(18,10))

plt.plot(t, detrend,color='red', label='Detrended Data')
plt.plot(t, rep_cycle,'--',color='black', label='Seasonal Cycle')

plt.xlabel("Time (years)", fontsize=12)
plt.ylabel("Mean Sea Level Change (mm)", fontsize=12)

plt.legend()
plt.title('Detrended time series and seasonal cycle', fontsize=20)
plt.savefig('9 Detrended & seasonal cycle', dpi=300)

plt.show()

# Time Series of Average Sea Level Annual Cycle
anomaly = detrend - rep_cycle


anom_max = max(anomaly)
anom_min = min(anomaly)

print(anom_min, anom_max)


# Find index of the minimum value
np.argmin(anomaly)


# Year of minimum value
t[np.argmin(anomaly)]

print('The largest monthly anomaly is', min(anomaly)/10, 'cm, which occurs in September of 1982.')


f, (ax1, ax2) = plt.subplots(2, figsize=(16,15))

ax1.plot(t, detrend, color = 'purple')
ax1.set_title('Detrended Sea Level Data', fontsize=18)
ax1.set_ylabel("Mean Sea Level Change (mm)", fontsize=14)

ax2.plot(t, anomaly, color = 'blue')
ax2.set_title('Sea Level Anomaly', fontsize=20)

plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Mean Sea Level Change (mm)", fontsize=14)
plt.savefig('10 Detrended and anomaly', dpi=300)
#plt.tight_layout()
plt.show()


# Plot the anomaly


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, 
    y=anomaly, 
    name='Sea Level Anomaly'))

fig.update_layout(
    template = "plotly_white",
    title_text="Monthly Anomaly",
    xaxis_rangeslider_visible=True)

fig.update_layout(xaxis=dict(range=[1875,2000]))

fig.update_xaxes(title_text="Time (years)")
fig.update_yaxes(title_text="Mean Sea Level Change (mm)")

plotly.offline.plot(fig, filename='11 Monthly anomaly.html')

fig.show()


# h.) Calculate the range of decadal variability (defined as difference between max and min value after applying an 8-year moving average to the raw data with the linear trend removed; note that raw data is given at monthly resolution)
 

# Apply 8-year moving average on detrended time series data and calculate the range of decadal variability

# Create a function for applying 8-year moving average, calculating range of decadal variability and plotting the results

def moving_avg(TS):
    
    # Calculate moving average
    rolmean = TS.rolling(window = 96, center = False).mean()  # 96 months = 8 years
    rol_mean = rolmean.dropna()
    max_rol = max(rol_mean)
    min_rol = min(rol_mean)
    dec_var = max_rol - min_rol
    #print(rol_mean)
    print('The range of the decadel variability is:', dec_var, 'mm')
     
    
    #Plot moving average:
    fig = plt.figure(figsize=(15,8))
    orig = plt.plot(TS, color='blue',label='Detreded TS')
    mean = plt.plot(rolmean, color='red', label='8-Year Moving Average')
    plt.legend(loc='best')
    plt.title('Detrended Time Series with 8-Year Moving Average', fontsize=20) 
    plt.xlabel('Time (years)', fontsize=16)
    plt.ylabel('Mean Sea Level Change (mm)', fontsize=16)
    plt.tick_params(labelsize=12);
    plt.savefig('12 Detrended TS with 8-year moving average', dpi=300)
    plt.show(block=False)


moving_avg(detrend)


# Plotting moving average, confidence intervals and anomalies


def plot_moving_avg(TS, window, plot_intervals=False, scale=2.576): # scale = 2.576 for 99% CI
    
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
    
    rolling_mean = TS.rolling(window=window).mean()
    
    roll_mean = rolling_mean.dropna()
    
    plt.figure(figsize=(17,8))
    plt.title("Moving average (window size = {})".format(window), fontsize=20)
    plt.plot(rolling_mean, 'r', label='Moving average trend', linewidth=2)
    
    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(TS[window:], rolling_mean[window:])
        deviation = np.std(TS[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
    
    plt.plot(TS[window:], label='Actual values', color='grey')
    plt.legend(loc='best')
    plt.grid(True)


plot_moving_avg(detrend, 96);


# Plotting moving average
def plotMovingAverage(series, window, plot_intervals=False, scale=2.576, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
    
    rolling_mean = series.rolling(window=window, center = False).mean()

    plt.figure(figsize=(15,8))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend", linewidth=2)

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=12)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    #plt.savefig('0 Moving average, confidence intervals and anomalies', dpi=300)


# # Decompose the time series

result = seasonal_decompose(data['sea_level'], model='additive', period=365) 

trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Plot gathered statistics
#plt.rcParams['figure.figsize'] = (16, 9)
f = plt.figure(figsize=(16,14))

plt.suptitle("Decomposition of the Mean Sea Level Time Series for Mumbai (India)", fontsize=20)

plt.subplot(411)
plt.plot(data, label='Original', color='blue')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend', color="black")
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality', color="red")
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals', color="m")
plt.legend(loc='best')

plt.xlabel("Time (years)", fontsize=12)

plt.savefig('13 Decomposition of original TS', dpi=300)

#plt.tight_layout()


# ### Decomposition of detrended time series


result2 = seasonal_decompose(detrend, model='additive', period=365) 

trend2 = result2.trend
seasonal2 = result2.seasonal
residual2 = result2.resid

# Plot gathered statistics
#plt.rcParams['figure.figsize'] = (16, 10)
f = plt.figure(figsize=(16,14))

plt.suptitle("Decomposition of the Detrended Time Series for Mumbai (India)", fontsize=20)

plt.subplot(411)
plt.plot(data, label='Detrended TS', color='blue')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend2, label='Trend', color="black")
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal2,label='Seasonality', color="red")
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual2, label='Residuals', color="m")
plt.legend(loc='best')

plt.xlabel("Time (years)", fontsize=12)

plt.savefig('14 Decomposition of detrended TS', dpi=300)

#plt.tight_layout()


# i.) Produce a plot of the raw data with linear trend and quadratic trend in the same panel

# Linear trend
parameters = np.polyfit(data.index.values, data['sea_level'], deg = 1) 
(a,b) = parameters
trend = a*data.index.values + b


# Quadratic trend 
para_quad = np.polyfit(t,s, deg = 2) 
(aq,bq,cq) = para_quad
acceleration = 2*aq
quad_trend = aq*(t**2) + bq*t +cq


# Plot linear and quadratic trends
f = plt.figure(figsize=(16,10))

plt.plot(data.index.values, data['sea_level'], color='grey', label='Original TS')
plt.plot(data.index.values, trend, label='Linear Trend', color ='red', linewidth=2)

plt.plot(t, quad_trend, label ='Quadratic Trend', color='blue', linewidth=2)

plt.title('Mean Sea Level in Mumbai (India)', fontsize=20)
plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Mean Sea Level (mm)", fontsize=14)
plt.legend(loc='best')

plt.savefig('15 Linear & Quadratic trends with Original TS', dpi=300)

plt.show()


# Plot linear and quadratic trends
f = plt.figure(figsize=(16,10))

plt.plot(data.index.values, trend, label='Linear Trend', color ='red', linewidth=2)

plt.plot(t, quad_trend, label ='Quadratic Trend', color='blue', linewidth=2)

plt.title('Mean Sea Level in Mumbai (India)', fontsize=20)
plt.xlabel("Time (years)", fontsize=14)
plt.ylabel("Mean Sea Level (mm)", fontsize=14)
plt.legend(loc='best')

plt.savefig('16 Linear & Quadratic trends', dpi=300)

plt.show()


# j.) Produce a plot with the de-trended raw data (i.e. linear trend removed) and 8-year moving average in the same panel

moving_avg(detrend)





