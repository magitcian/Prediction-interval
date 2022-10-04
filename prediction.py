import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table
import matplotlib.pyplot as plt
import pandas as pd

# get linearisation from extremity points :
def get_interval_equation(extremity_points):
    points = extremity_points
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, 1)[0]
    m = round(m,3)
    c = round(c,3)
    print("Prediction interval equation is: y = {m}x + {c}".format(m=m,c=c))
    return "y = {m}x + {c}".format(m=m,c=c)

#TODO: make another function to obtain linear regression from all interval points

dt = pd.read_csv('data.csv')
# defining the variables
x = dt['x'].tolist()
y = dt['y'].tolist()
# x =  data.iloc[0:len(data),0] 
# y =  data.iloc[0:len(data),1]
#X = sm.add_constant(x)

# performing the regression and fitting the model
result = sm.OLS(y, x).fit()
print(result.summary())
prstd, iv_l, iv_u = wls_prediction_std(result)
st, data, ss2 = summary_table(result, alpha=0.05)
fittedvalues = data[:, 2]
predict_mean_se  = data[:, 3]
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
predict_ci_low, predict_ci_upp = data[:, 6:8].T

#Upper prediction interval equation :
upi = "Top : " + get_interval_equation([(x[0], predict_ci_upp[0]), (x[len(x)-1], predict_ci_upp[len(predict_ci_upp)-1])])

#Lower prediction interval equation :
lpi = "Bottom : " + get_interval_equation([(x[0], predict_ci_low[0]), (x[len(x)-1], predict_ci_low[len(predict_ci_low)-1])])

#Obtain graph :
plt.plot(x, y, 'o')
plt.plot(x, fittedvalues, '-', lw=2)
plt.plot(x, predict_ci_low, 'r--', lw=2, label=upi) # prediction interval
plt.plot(x, predict_ci_upp, 'r--', lw=2, label=lpi) # prediction interval
plt.plot(x, predict_mean_ci_low, 'r--', lw=2, color='pink') # confidence interval
plt.plot(x, predict_mean_ci_upp, 'r--', lw=2, color='pink') # confidence interval
plt.legend()
plt.savefig("figure.png")
plt.show() 