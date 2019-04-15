import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tsne import bh_sne
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d

# *******TSNE********
path = './data-standard.csv'
features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation']

df = pd.read_csv(path)
X, y = np.array(df[xfeatures]), np.array(df['sleep'])
id_n=(np.array(df['id']))
vis_data = bh_sne(X, random_state = np.random.RandomState(0))

c= df['sleep'].values
color =[]


for i in range (len(c)):
    if c[i]==1:
        color.append('#008B8B')



    else:
        color.append('#00008B')



vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
plt.scatter(vis_x, vis_y, c= id_n, s=70, marker="^", alpha=0.3, label='exam')
plt.title("TSNE Exam")
plt.legend()
plt.show()

plt.scatter(vis_x,vis_y, c= color, marker='^', alpha=0.4 )
plt.title("TSNE Sleep")
plt.legend()
plt.show()



##############

X, y = np.array(data_stand[features]), np.array(data_stand['sleep'])
id_n=(np.array(data_stand['id']))


pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2','principal_component_3'])
finalDf = pd.concat([principalDf, df[['sleep']],df['id']], axis = 1)
finalDf.to_csv('../dataset/pca-bca.csv',index =False)
finalDf = pd.read_csv('../dataset/pca-bca.csv')
groups = finalDf.groupby(by = 'id')

c= df['sleep'].values
color =[]

for i in range (len(c)):
    if c[i]==1:
        color.append('#008B8B')


    else:
        color.append('#00008B')


vis_x = principalComponents[:,0]

vis_y = principalComponents[:,1]

vis_z = principalComponents[:,2]


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(vis_x, vis_y, vis_z, c = id_n,marker='^', alpha = 0.5, label = 'sleep')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(vis_x, vis_y, vis_z, c=color,marker = '^',label = 'sleep')
plt.show()