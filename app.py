#import necessary packages
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

#create data to work with
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
[8.0, 1, 2.5, 4, 28.0]
x_with_nan
[8.0, 1, 2.5, nan, 4, 28.0]


math.isnan(np.nan), np.isnan(math.nan)
math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

#mean using imported python statistics function
mean_ = statistics.mean(x)
mean_

#weighted mean
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

#harmonic mean
hmean = len(x) / sum(1 / item for item in x)
#with statistics function
hmean = statistics.harmonic_mean(x)
#third method
scipy.stats.hmean(y)

#geometric mean
gmean = statistics.geometric_mean(x)
gmean
scipy.stats.gmean(y)

#median
median_ = statistics.median(x)
median_
statistics.median_low(x[:-1])
statistics.median_high(x[:-1])
median_ = np.median(y)
median_
median_ = np.median(y[:-1])
median_
np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])
z.median()
z_with_nan.median()

#mode
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

v = [12, 15, 12, 15, 21, 15, 12]
#single value, raises StatisticsError
statistics.mode(v) 
statistics.multimode(v)
#another method for mode
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

#measures of variability
var_ = statistics.variance(x)
var_

#standard dev
std_ = statistics.stdev(x)
std_
z.std(ddof=1)
z_with_nan.std(ddof=1)

#skewness
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)

#percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x, n=4, method='inclusive')

#finding specific percentiles
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

#range
np.ptp(y)
#interquartile range
quartiles = np.quantile(y, [0.25, 0.75])

#summary
result = scipy.stats.describe(y, ddof=1, bias=False)
result = z.describe()

#measures of correlation
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#covariance
cov_matrix = np.cov(x_, y_)
cov_xy = x__.cov(y__)

#correlation coefficient 
r, p = scipy.stats.pearsonr(x_, y_)
r
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

#create 2d NumPy array to work with
a = np.array([[1, 1, 1],
[2, 3, 1],
[4, 9, 2],
[16, 1, 1]])

np.median(a, axis=0)
np.median(a, axis=1)

#statistics for whole dataset
scipy.stats.gmean(a, axis=None)

#dataframes 
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df
df.describe().at['mean', 'A']

#visualizing data 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#boxplots
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
    labels=('x', 'y', 'z'), patch_artist=True,
    medianprops={'linewidth': 2, 'color': 'purple'},
    meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#histograms 
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges
#graphically 
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#pie chart 
x, y, z = 128, 256, 1024
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#bar chart 
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
#plotted
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#scatter plot
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
#plotted scatter plot 
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#heatmap
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()
