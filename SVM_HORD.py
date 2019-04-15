import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from termcolor import colored
from pySOT import check_opt_prob
from pySOT import *
from poap.controller import SerialController, BasicWorkerThread
import matplotlib.pyplot as plt


path = './data-standard.csv'
features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation']

df = pd.read_csv(path)
df = shuffle(df)


class SVMOPT:
    def __init__(self, dim=3):
        self.xlow = np.array([0, 0.01, 0.01])
        self.xup = np.array([2, 1, 1])
        self.dim = dim
        self.info = "Our own " + str(dim) + "-dimensional function"  # info
        self.integer = np.array([0])  # integer variables
        self.continuous = np.arange(1, dim)  # continuous variables

    def objfunction(self, x):
        kernel = {0: 'rbf', 1: 'linear', 2: 'sigmoid'}
        k = kernel[x[0]]
        X, Y = df[xfeatures].values, df['sleep'].values
        model = SVC(C=x[1], gamma=x[2], kernel=k)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=int(0.2 * len(df)))
        model.fit(x_train, y_train)
        acc = accuracy_score(y_test, model.predict(x_test))
        print(colored('Features:','blue'), colored(x,'green'))
        print(colored('Accuracy:', 'green'), colored(acc * 100, 'blue'))
        return 1 - acc


data = SVMOPT(dim=3)
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