#  Amanda_Barroso_HW_1.py
#  Created by Amanda Barroso on 9/10/20.

import numpy as np
import pandas as pd
import statistics
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as st
from scipy.stats import t

## a.) Calculate the linear trend (in mm/yr) and 99% confidence interval for the entire data record

# Load Data file
data = pd.read_excel('/Volumes/ExternalDrive/UCF_Data/StPete_USA.xlsx', header=None)
x = data.iloc[:,0] #first row: time
y = data.iloc[:,1] #second row: sea level
L = len(data)

# Make a matrix with the time information (x is years)
matrix = (x, np.ones((L,1)))

# Build Model based on y = ax + b + Error
x_mean= np.mean(x)
y_mean= np.mean(y)
#   METHOD 1: Compute Equation Manually
#numerator= 0
#denominator= 0
#for i in range(len(x)):
#    numerator   += (x[i] - x_mean)*(y[i] - y_mean)
#    denominator += (x[i] - x_mean)**2
#a = numerator/denominator
#b = y_mean - a*x_mean

#print(a,b)
#prediction = a*x + b
#plt.plot(x,y)
#plt.plot([min(x),max(x)], [min(prediction), max(prediction)], color='red') #trend line

#   METHOD 2: Use Numpy's polynomial fit function
parameters = np.polyfit(x,y, deg = 1) #Linear trend is a polynomial of degree 1
(a,b) = parameters
trend = a*x + b
print('polyfit parameters are', a,b)

fig = plt.figure(1)
plt.plot(x,y, color='grey')
plt.plot(x,trend, label='Polyfit Trend Line', color ='red')
#plt.legend()
#plt.show()
# remember data is given in mm/month so convert estimates to mm/yr

#   METHOD 3: Least Squares
# Rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]
#A = np.vstack([x, np.ones(L)]).T
#m,c = np.linalg.lstsq(A,y, rcond=None)[0]
#print('Least Squares parameters are', m,c)
#line = m*x+c
#plt.plot(x,y)
#plt.plot(x,line, label='Least Squares Trend Line', color='yellow')

# Confidence Intervals
CI95a = [np.percentile(a,2.5), np.percentile(a,97.5)] #97.5 - 2.5 = 95.
#print('CI95 of the slope is', CI95m) #95% chance that the parameter (coefficient of x the slope of trend line) lies within the 2 output values
CI99a = [np.percentile(a,1), np.percentile(a,100)] #100 - 1 = 99
#print('CI99 of the slope is', CI99a)
#CI99trend = [np.percentile(trend,1), np.percentile(trend,100)]
#print('CI99 of the trend line is', CI99trend)
#plt.legend()
#plt.show()

#st.t.interval(alpha=0.99, df=L-1, loc=np.mean(m), scale=st.sem(m))


## b. Calculate the acceleration (in mm/yr2) for the entire data record (note that acceleration is defined as twice the quadratic coefficient!)
para_quad = np.polyfit(x,y, deg = 2) #Quadratic trend is a polynomial of degree 2: ax^2 + bx + c
(aq,bq,cq) = para_quad
acceleration = 2*aq
print('The acceleration of SLR in St. Pete is', acceleration, 'mm/yr^2')
quad_trend = aq*(x**2) + bq*x +cq
print('The quadratic parameters are', para_quad)
plt.plot(x, quad_trend, label ='Quadratic Trend', color='green')
plt.xlabel('Time (years)')
plt.ylabel('Mean Sea Level (mm)')
plt.legend()
plt.show()

## c.) Calculate the linear trends (in mm/yr) for the first and second halves of the time series

x1 = x[:int(L/2)]    #first half
y1 = y[:int(L/2)]
(a1, b1) = np.polyfit(x1,y1, deg=1)
trend1 = a1*x1 + b1
print('Linear trend from 1947-1981 is', a1, 'mm/yr')

x2 = x[int(L/2):]    #second half
y2 = y[int(L/2):]
(a2, b2) = np.polyfit(x2,y2, deg=1)
trend2 = a2*x2 + b2
print('Linear trend from 1982-2016 is', a2, 'mm/yr')

## d.) Based on the 95% confidence levels determine if the trends derived in (c)are significantly different from each other (i.e. confidence levels do not overlap).
CI95_1 =[np.percentile(a1,2.5), np.percentile(a1,97.5)] #97.5 - 2.5 = 95.
print(CI95_1)
CI95_2 =[np.percentile(a2,2.5), np.percentile(a2,97.5)] #97.5 - 2.5 = 95.
print(CI95_2)

## e.) Calculate the amplitude (in cm) of the average annual cycle for the entire period and identify the month when it peaks (remove the linear trend from (a) for detrending)

# Detrend Data
detrend = y - trend

# Create a row for each month (12) & a column for each year
mo = detrend.values.reshape(12,int(L/12)) #float to integer

# Seasonal (Annual) Cycle by averaging by months (all the Januaries, Febs, etc.)
cycle = mo.mean(axis=1) #average the rows

#Amplitude
amp = (max(cycle)-min(cycle))/2
print('The amplitude of the seasonal cycle is', amp/10, 'cm')
print('On average, sea level in St. Pete peaks in December at', max(cycle)/10, 'cm')


## f.) Same as (e) but separately for the first 5 years of the record and last 5 years of the record (note: use the same detrended time series as in (e), DO NOT detrend the first/last five years of data again)

# First 5 years
f5cycle = mo[:,0:5].mean(axis=1) # all the rows:(months), first 5 columns (years)
f5amp = (max(f5cycle)-min(f5cycle))/2
print('The amplitude of the average annual cycle from 1947-1952 is', f5amp/10, 'cm')
print('During the first 5 years of the record, sea level peaks in the month of December')

# Last 5 years
l5cycle = mo[:,-5:].mean(axis=1) #last 5 columns
l5amp = (max(l5cycle)-min(l5cycle))/2
print('The amplitude of the last 5 years of the sea level record in St. Pete is', l5amp/10, 'cm')
print('During the last 5 years of the record, the average sea level peaks in November')

## g.) Identify the largest (positive or negative) monthly anomaly above or below the average seasonal cycle (note: create a time series of the average seasonal cycle and subtract it from the raw data with linear trend removed)

# repeat the annual cycle for the entire record (to match the dimensions of x years)
# 70 years of data, so repeat seasonal cycle 70 times
rep_cycle = np.tile(cycle,int(L/len(cycle)))

#fig=plt.figure(2)
#plt.plot(x,detrend,color='purple', label='Detrended Data')
#plt.plot(x,rep_cycle,'--',color='teal', label='Seasonal Cycle')
#plt.legend()
#plt.show()

# Time Series of Avg Sea Level Annual Cycle
anomaly = detrend - rep_cycle

print(max(anomaly),min(anomaly))
print('The largest monthly anomaly is', min(anomaly))
# -_--_-REMEMBER TO IDENTIFY WHERE IT TAKES PLACE-_--_-

#f, (ax1, ax2) =plt.subplots(2)
#ax1.plot(x, detrend, color='purple')
#ax1.set_title('Detrended Sea Level Data')
#ax2.plot(x,anomaly, color='black')
#ax2.set_title('Sea Level Anomaly')
#plt.tight_layout()
#plt.show()


## h.) Calculate the range of decadal variability (defined as difference between max and min value after applying an 8-year moving average to the raw data with the linear trend removed; note that raw data is given at monthly resolution)

# Moving Average of detrended data
window = 8
# Create an empty list
MA = []
# Loop to compute moving average for each time step
for i in range(len(detrend)-(window*12)):
    MA.append(np.mean(detrend[i:i+(window*12)]))
 
print(MA)

# Range of Decadal Variability
dv_range = max(MA) - min(MA) #or absolute value??

## i.) Produce a plot of the raw data with linear trend and quadratic trend in the same panel


## j.) Produce a plot with the de-trended raw data (i.e. linear trend removed) and 8-year moving average in the same panel

fig = plt.figure(5)
plt.plot(x,detrend, label='Detrended Data')
plt.plot(x[48:-48],MA, label='8-yr moving average')
plt.xlabel('Mean Sea Level Change (mm)')
plt.ylabel('Years')
plt.legend()
plt.show()
