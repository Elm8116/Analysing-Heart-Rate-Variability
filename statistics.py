import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import normaltest

from scipy import stats
features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation',]
path = '../data-standard.csv'


df = pd.read_csv(path )
df=df[features]

df_2=pd.read_csv(path2)


df.hist()
plt.show()

df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

df_2.hist()
plt.show()

# df_2.plot(kind='density', subplots=True, layout=(6,6), sharex=False)
# plt.show()


value, p = normaltest(df.values[:,0])
print(value, p)
if p >= 0.05:
	print('It is likely that result1 is normal')
else:
	print('It is unlikely that result1 is normal')
