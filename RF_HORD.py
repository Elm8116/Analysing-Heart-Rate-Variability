import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from termcolor import colored
from pySOT import check_opt_prob
from sklearn.model_selection import cross_val_score

from pySOT import *
from poap.controller import SerialController, BasicWorkerThread
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import metrics
path = './data-standard.csv'
features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation']

df = pd.read_csv(path)


RandomForestClassifier()
class RFOPT:
    def __init__(self, dim=5):
        self.xlow = np.array([1,1,2,1,0])
        self.xup = np.array([300,12,10,10,1])
        self.dim = dim
        self.info = "Our own " + str(dim) + "-dimensional function"  # info
        self.integer = np.arange(0, dim)  # integer variables
        self.continuous = np.array([])  # continuous variables

    def objfunction(self, x):

        kernel = {0: 'gini', 1: 'entropy'}
        criterion = kernel[x[-1]]
        X, Y = df[xfeatures].values, df['sleep'].values

        logo = LeaveOneGroupOut()
        grp = df['id'].values
        scores = []
        #  55.  11.   8.   3.   1.
        for train, test in logo.split(X, Y, grp):
            model = RandomForestClassifier(n_estimators=int(x[0]), criterion=criterion, max_depth=int(x[1]),
                                           min_samples_split=int(x[2]),
                                           min_samples_leaf=int(x[3]))
            x_train, x_test = X[train], X[test]
            y_train, y_test = Y[train], Y[test]
            model.fit(x_train, y_train)
            scores.append(metrics.accuracy_score(y_test, model.predict(x_test)))


        print(colored('Features:','blue'),colored(x,'green'))
        print(colored('Accuracy:', 'green'), colored(np.mean(scores) * 100, 'blue'))
        return 1 - np.mean(scores)


data = RFOPT(dim=5)
check_opt_prob(data)

# Decide how many evaluations we are allowed to use
maxeval = 500

# (1) Optimization problem
# Use our 10-dimensional function
print(data.info)

# (2) Experimental design
# Use a symmetric Latin hypercube with 2d + 1 samples
exp_des = SymmetricLatinHypercube(dim=data.dim, npts=2*data.dim+1)

# (3) Surrogate model
# Use a cubic RBF interpolant with a linear tail
surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

# (4) Adaptive sampling
# Use DYCORS with 100d candidate points
adapt_samp = CandidateDYCORS(data=data, numcand=100*data.dim)

# Use the serial controller (uses only one thread)
controller = SerialController(data.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))


# Extract function values from the controller
fvals = np.array([o.value for o in controller.fevals])

f, ax = plt.subplots()
ax.plot(np.arange(0,maxeval), fvals, 'bo')  # Points
ax.plot(np.arange(0,maxeval), np.minimum.accumulate(fvals), 'r-', linewidth=4.0)  # Best value found
plt.xlabel('Evaluations')
plt.ylabel('Function Value')
plt.title(data.info)
plt.show()