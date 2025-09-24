# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:35:52 2025

@author: Rupali
"""

#=================================================================================================================
1) Mean, Median, MAD & Standard Deviation – Real World
#=================================================================================================================
'''
Problem:
You are analyzing monthly sales data for two shops.
shop1_sales = [2200, 2250, 2300, 2350, 2400, 4000]
shop2_sales = [2000, 2100, 2300, 2500, 2700, 2800]

'''

----------------------------------------------------------------------------------------------------------------------------
import numpy as np

shop1_sales = [2200, 2250, 2300, 2350, 2400, 4000]
shop2_sales = [2000, 2100, 2300, 2500, 2700, 2800]

##a. Calculate the mean, median, MAD, and standard deviation for both shops.

##1) Mean (Average)
print("Mean of the Shop1:", np.mean(shop1_sales))
##Output: Mean of the Shop1: 2583.3333333333335

print("Mean of the shop2:", np.mean(shop2_sales))
##Output: Mean of the shop2: 2400.0

##2) Median (Central value)
print("Median of the shop1:", np.median(shop1_sales))
##Output: Median of the shop1: 2325.0

print("Median of the Shop2:", np.median(shop2_sales))
##Output: Median of the Shop2: 2400.0

##3) Mean Absolute Deviation(MAD)
print("MAD of Shop1:", np.mean(np.abs(shop1_sales - np.mean(shop1_sales))))
##Output: MAD of Shop1: 472.22222222222234

print("MAD of Shop2:", np.mean(np.abs(shop2_sales - np.mean(shop2_sales))))
##Output: MAD of Shop2: 266.6666666666667

##4) Standard Deviation(SD)
print("Standard Deviation of Shop1:", np.std(shop1_sales))
##Output: Standard Deviation of Shop1: 636.8324391514267

print("Standard Deviation of Shop2:", np.std(shop2_sales))
##Output: Standard Deviation of Shop2: 294.3920288775949

------------------------------------------------------------------------------------------------------------
##b) Which shop shows higher consistency in sales and why?
'''
Shop 2 has lower standard deviation and MAD, meaning less 
variation → it means Shop2 is more consistent.
'''
---------------------------------------------------------------------------------------------------------------------------
##Suppose the outlier in shop1 is corrected to 2450. 
##How do the metrics change?
shop1_corrected = np.array([2200, 2250, 2300, 2350, 2400, 2450])
shop1_corrected
##Output: array([2200, 2250, 2300, 2350, 2400, 2450])

print("Mean:", np.mean(shop1_corrected))
##Output: Mean: 2325.0

print("Median:",np.median(shop1_corrected))
##Output: Median: 2325.0

print("MAD:", np.mean(np.abs(shop1_corrected - np.mean(shop1_corrected))))
##MAD: 75.0

print("Standard Deviation:", np.std(shop1_corrected))
##Output: Standard Deviation: 85.39125638299666

#================================================================================================================================
2) Effect of Data Transformation on Spread
#===============================================================================================================================
'''
Problem:
Given the dataset:
data = [25, 30, 35, 40, 45, 50]
'''
##a)  Compute the mean and standard deviation.
import numpy as np

data = np.array([25, 30, 35, 40, 45, 50])

print("Mean:", np.mean(data))
##Mean: 37.5

print("Standard Deviation:", np.std(data))
##Output: Standard Deviation: 8.539125638299666
-----------------------------------------------------------------------
##b) Now apply:
    ##Addition: Add 5 to each value
print("After Adding 5, Mean is:", np.mean(data + 5)) 
##Output: After Adding 5, Mean is: 42.5

print("After Adding 5, Std Dev is:", np.std(data + 5))
##Output: After Adding 5, Std Dev is: 8.539125638299666

##Multiplication: Multiply each value by 2
print("After Multiplying by 2, Mean is:", np.mean(data * 2))
##Output: After Multiplying by 2, Mean is: 75.0

print("After Multiplying by 2, Std Dev:", np.std(data * 2))
##Output: After Multiplying by 2, Std Dev: 17.07825127659933
--------------------------------------------------------------------------------------------
##Log transformation: Apply np.log(data)
print("After Log Transformation, Mean is:", np.mean(np.log(data)))
##Output: After Log Transformation, Mean is: 3.5971643695553617

print("After Log Transformation, Std Dev is:", np.std(np.log(data)))
##Output: After Log Transformation, Std Dev is: 0.2361817685239452

--------------------------------------------------------------------------------------------
##c) Discuss the effect of each transformation on center and spread.
'''
Addition shifts the center but spread stays same. 
Multiplication increases both center and spread. 
Log compresses data, reducing the spread.
'''
--------------------------------------------------------------------------------------------
#=========================================================================================================================
3) Density Curve vs Histogram
#===========================================================================================================================
'''
Problem:
Generate 1000 height values assuming a normal distribution 
(mean=160 cm, std=10 cm).
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42) 
heights = np.random.normal(160, 10, 1000)

##a) Plot histogram and KDE using seaborn.
sns.histplot(heights, kde=True)
plt.show()
##This histogram shows the distribution of heights.
##and smooth line(KDE) shows the probability curve.

-------------------------------------------------------------------------------------------------------
##b) Manually create bins (intervals of 5 cm) and compute relative frequency.
bins = range(140, 181, 5)
freq, edges = np.histogram(heights, bins=bins)
rel_freq = freq/len(heights)
rel_freq
##Output: array([0.04 , 0.09 , 0.166, 0.177, 0.204, 0.151, 0.08 , 0.051])
----------------------------------------------------------------------------------------------------------------
##c) Approximate the area under the density curve between 150–170 cm. What does it represent?
area = norm.cdf(170, 160, 10) - norm.cdf(150, 160, 10)
print(area)
##output: 0.6826894921370859

#===================================================================================================================================
4) Skewness & Kurtosis Comparison
#===================================================================================================================================
from scipy.stats import skew, kurtosis
import numpy as np

##Create A left-skewed dataset
left_skew = np.random.beta(a=5, b=2, size=1000) * 50
print("Left Skewed:", skew(left_skew))
##Output: Left Skewed: -0.6490152230059311

print("Kurtosis of Left Skewed:", kurtosis(left_skew))
##Output: Kurtosis of Left Skewed: -0.02342323210329056

##Create A right-skewed dataset
right_skew = np.random.beta(a=2, b=5, size=1000) * 50
print("Right Skewed:", skew(right_skew))
##Output: Right Skewed: 0.5344386170606393

print("Kurtosis of Right Skewed:", kurtosis(right_skew))
##Output: Kurtosis of Right Skewed: -0.2926752931739154

##Create A symmetric dataset (normal)
normal_data = np.random.normal(50, 10, 1000)
print("Skewness of Symmetric Data:", skew(normal_data))
##Output: Skewness of Symmetric Data: 0

print("Kurtosis of Symmetric Data:", kurtosis(normal_data))
##Output: Kurtosis of Symmetric Data: 0
-------------------------------------------------------------

##Now plotting the histograms for each
import seaborn as sns
import matplotlib.pyplot as plt

#1)A left-skewed dataset
plt.subplot(1,3,1)
sns.histplot(left_skew, bins=20, kde=True, color="skyblue")
plt.title("Left Skewed")
'''
Shape: The peak is on the right side, and the tail stretches
        to the left.
Behavior: Most values are higher, but a few low values
          pull the tail left.
'''

##2) A right-skewed dataset
plt.subplot(1,3,2)
sns.histplot(right_skew, bins=20, kde=True, color="salmon")
plt.title("Right Skewed")
'''
Shape: The peak is on the left side, and the tail
       stretches to the right side.
Behavior: Most values are lower, but a few large values
          pull the tail right.
'''

##3)A symmetric dataset (normal)
plt.subplot(1,3,3)
sns.histplot(normal_data, bins=20, kde=True, color="lightgreen")
plt.title("Normal")
'''
Shape: Bell-Shaped curve, Balanced on both sides.
Behavior: Tails are even, with most values near the mean
          and fewer values at the extremes.
'''

#=================================================================================================================================
5) Chebyshev's Inequality on Unknown Distribution
#=================================================================================================================================
'''
Problem:
A dataset has mean income = ₹50,000 and standard deviation = ₹12,000.
''' 
'''
a. Without knowing the distribution, use Chebyshev’s theorem to estimate 
what percentage of individuals earn between ₹26,000 and ₹74,000.
'''
k = 74000 - 50000/12000 = 2
#At least 
1 - 1/k^2  
1-1/4
##Output: 0.75 i.e 75%
##So At least 75% of individuals earn between ₹26,000 and ₹74,000.

'''
b. Compare this with the Empirical Rule (assuming normal distribution).
'''
##For Normal Distribution:
##About 95% of values lie within 2 standard deviations of the mean.

'''
Chebyshev guarantees >=75% (very general), while the Empirical Rule suggests 
~95% (but only if the data is approximately normal).
'''

#===================================================================================================================================
6) Real-World Log Transformation
#===================================================================================================================================
'''
Load the csv file containing:
•	Population sizes of 1000 cities
•	Income distribution of households
'''
##Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##Load the file
data = pd.read_csv("R:/Sanjivani_Assignments_2/city_population_income.csv")

##Extracting the columns
populations = data["Population_Size"]
incomes = data["Household_Income"]

#a. Plot histograms for both features
##Population
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.histplot(populations, bins=30, kde=True)
plt.title("Population Size")
plt.show()
##Right-Skewed

##Income
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.histplot(incomes, bins=30, kde=True)
plt.title("Household Income")
plt.show()
##Right Skewed
-------------------------------------------------------------------------------------------------
#b) Apply log transformation and re-plot

##Applying log transformation
data["Log_Population"] = np.log(data["Population_Size"] + 1)
data["Log_Income"] = np.log(data["Household_Income"] + 1)

##Population
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.histplot(data["Log_Population"], bins=30, kde=True)
plt.title("Population Size")
plt.show()
##Balanced, Close to the normal curve

##Income
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.histplot(data["Log_Income"], bins=30, kde=True)
plt.title("Household Income")
plt.show()
## Normal shape of the curve means balanced.
------------------------------------------------------------------------------------------------------
#c. Explain how log helps in compressing skewed data
'''
In skewed data, a few very big values stretch the graph.
log function helps to makes big numbers smaller and closer to others.
log helps to reduces skewness and makes the graph more balanced.
'''

#d. Comment on interpretability after transformation
'''
After log transformation, we compare data in terms of percentages or 
ratios instead of raw numbers.
This makes patterns and comparisons easier to understand.
'''

#===================================================================================================================================
7) SciPy Applications – Linear Algebra and Interpolation
#===================================================================================================================================
##a. Create a 3x3 matrix and compute its determinant using scipy.linalg.det()
import numpy as np
from scipy.linalg import det

##Now creating a 3x3 matrix
A = np.array([[1,2,3],
              [4,5,6],
               [7,8,9],
               ])
det_A = det(A)
print("Determinant of A:", det_A)
##Output: Determinant of A: 0.0

##b. Use scipy.interpolate.interp1d() to interpolate the following points and estimate y at x = 3.5:
from scipy.interpolate import interp1d

##Data points
x = [1, 2, 4, 5]
y = [1, 4, 2, 5]

##Now creating the interpolation function
f = interp1d(x, y)

#estimating y at x = 3.5
y2 = f(3.5)
print("Estimated y at x=3.5:", y2)
##Output: Estimated y at x=3.5: 2.5
