# Basic Time Series Models - Lab

## Introduction

Now that you have some basic understanding of the white noise and random walk models, its time for you to implement them! 

## Objectives

In this lab you will: 

- Generate and analyze a white noise model 
- Generate and analyze a random walk model 
- Implement differencing in a random walk model 


## A White Noise model

To get a good sense of how a model works, it is always a good idea to generate a process. Let's consider the following example:
- Every day in August, September, and October of 2018, Nina takes the subway to work. Let's ignore weekends for now and assume that Nina works every day 
- We know that on average, it takes her 25 minutes, and the standard deviation is 4 minutes  
- Create and visualize a time series that reflects this information 

Let's import `pandas`, `numpy`, and `matplotlib.pyplot` using their standard alias. 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Do not change this seed
np.random.seed(12) 
```

Create the dates. You can do this using the `date_range()` function Pandas. More info [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html).


```python
# Create dates
dates = pd.date_range(start='2018-8-1', end='2018-10-31', freq='B')
```

Generate the values for the white noise process representing Nina's commute in August and September.


```python
# Generate values for white noise
commute = np.random.normal(25, 4, len(dates))
```

Create a time series with the dates and the commute times.


```python
# Create a time series
commute_series = pd.Series(commute, index=dates)
```

Visualize the time series and set appropriate axis labels.


```python
# Visualize the time series 
commute_series.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f886143dc18>




![png](index_files/index_13_1.png)


Print Nina's shortest and longest commute.


```python
# Shortest commute
commute_series.min(), commute_series.max()
```




    (12.41033391382408, 36.487277579955666)




```python
# Longest commute

```

Look at the distribution of commute times.


```python
# Distribution of commute times
commute_series.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f88c0862a90>




![png](index_files/index_18_1.png)


Compute the mean and standard deviation of `commute_series`. The fact that the mean and standard error are constant over time is crucial!


```python
# Mean of commute_series
commute_series.mean()
```




    24.648446172457277




```python
# Standard deviation of commute_series
commute_series.std()
```




    4.250874108206889



Now, let's look at the mean and standard deviation for August and October.  


```python
# Mean and standard deviation for August and October
aug_mean = commute_series['2018-8'].mean()
aug_std = commute_series['2018-8'].std()
oct_mean = commute_series['2018-10'].mean()
oct_std = commute_series['2018-8'].std()
```


```python
aug_mean, oct_mean, aug_std, oct_std
```




    (25.35433780425335, 25.687173311786953, 4.300977315435741, 4.300977315435741)



Because you've generated this data, you know that the mean and standard deviation will be the same over time. However, comparing mean and standard deviation over time is useful practice for real data examples to check if a process is white noise!

## A Random Walk model 

Recall that a random walk model has: 

- No specified mean or variance 
- A strong dependence over time 

Mathematically, this can be written as:

$$Y_t = Y_{t-1} + \epsilon_t$$

Because today's value depends on yesterday's, you need a starting value when you start off your time series. In practice, this is what the first few time series values look like:
$$ Y_0 = \text{some specified starting value}$$
$$Y_1= Y_{0}+ \epsilon_1 $$
$$Y_2= Y_{1}+ \epsilon_2 = Y_{0} + \epsilon_1 + \epsilon_2  $$
$$Y_3= Y_{2}+ \epsilon_3 = Y_{0} + \epsilon_1 + \epsilon_2 + \epsilon_3 $$
$$\ldots $$

Keeping this in mind, let's create a random walk model: 

Starting from a value of 1000 USD of a share value upon a company's first IPO (initial public offering) on January 1 2010 until end of November of the same year, generate a random walk model with a white noise error term, which has a standard deviation of 10.


```python
# Keep the random seed
np.random.seed(11)

# Create a series with the specified dates
dates = pd.date_range(start='2010-01-01', end='2010-11-30', freq='B')

# White noise error term
error = np.random.normal(0, 10, len(dates))

# Define random walk
def random_walk(start, error):
    y_0 = start
    cum_error = np.cumsum(error)
    y = y_0 + cum_error 
    return y

shares_value = random_walk(1000, error)

shares_series = pd.Series(shares_value, index=dates)
```

Visualize the time series with correct axis labels. 


```python
# Your code here
shares_series.plot(figsize=(16,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f886206fa58>




![png](index_files/index_29_1.png)


You can see how this very much looks like the exchange rate series you looked at in the lesson!

## Random Walk with a Drift

Repeat the above, but include a drift parameter $c$ of 8 now!


```python
# Keep the random seed
np.random.seed(11)

def random_drift(start, error):
    y_0 = start
    cum_error = np.cumsum(error + 8)
    y = y_0 + cum_error 
    return y

shares_drift = random_drift(1000, error)
shares_series_drift = pd.Series(shares_drift, index=dates)
```


```python
ax = shares_series_drift.plot(figsize=(14,6))
ax.set_ylabel('Stock value', fontsize=16)
ax.set_xlabel('Date', fontsize=16)
plt.show()
```


![png](index_files/index_33_0.png)


Note that there is a very strong drift here!

## Differencing in a Random Walk model

One important property of the random walk model is that a differenced random walk returns a white noise. This is a result of the mathematical formula:

$$Y_t = Y_{t-1} + \epsilon_t$$
which is equivalent to
$$Y_t - Y_{t-1} = \epsilon_t$$

and we know that $\epsilon_t$ is a mean-zero white noise process! 

Plot the differenced time series (time period of 1) for the shares time series (no drift).


```python
# Your code here
walk_diff = shares_series.diff(periods=1)
fig = plt.figure(figsize=(18,6))
plt.plot(walk_diff)
plt.title('Differenced shares series')
plt.plot(figsize=(16,6), block=False);
```


![png](index_files/index_38_0.png)


This does look a lot like a white noise series!

Plot the differenced time series for the shares time series (with a drift).


```python
# Your code here 
drift_diff = shares_series_drift.diff(periods=1)
fig = plt.figure(figsize=(18,6))
plt.plot(walk_diff)
plt.title('Differenced shares series')
plt.plot(figsize=(16,6), block=False);
```


![png](index_files/index_41_0.png)


This is also a white noise series, but what can you tell about the mean? 

The mean is equal to the drift $c$, so 8 for this example!

## Summary

Great, you now know how white noise and random walk models work!
