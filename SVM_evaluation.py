import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn_evaluation import plot
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.stats import sem
from tsne import bh_sne
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import metrics

data= pd.read_csv('..//data-standard.csv')

features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation']
data = shuffle(data)
X, Y = np.array(data[features]), np.array(data['sleep'])

grp = data['id'].values
model = model = SVC(gamma = 0.74373271 , C = 0.47289701 , kernel='rbf')
logo = LeaveOneGroupOut()
cv = LeaveOneGroupOut().split(X, Y, grp)
scores = []
scores_2 =[]
for train, test in logo.split(X, Y, grp):
    x_train, x_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    model.fit(x_train, y_train.ravel())

    scores.append(metrics.accuracy_score(y_test, model.predict(x_test)))

print ("Support Vector Machine (SVM) Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), np.std(scores))


scores_cross = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print('SVM Score: {0:.3f} (+/-{1:.3f})').format(np.mean(scores_cross), np.std(scores_cross))


def conf_mat_logo_aggregated():
    grp = data['id'].values

    logo = LeaveOneGroupOut()
    i = 0
    ytt = []
    yee = []
    for train, test in logo.split(X, Y, grp):
        model = SVC(gamma = 0.74373271 , C =   0.47289701 , kernel='rbf')
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        model.fit(x_train, y_train)
        ytt += list(y_test)
        yee += list(model.predict(x_test))
        print ("person ", i, "y_test",len(ytt))
        i += 1
        # Plot non-normalized confusion matrix
    cnf_matrix = confusion_matrix(ytt, yee)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['class 0', 'class 1'],
                          title='SVM matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['class 0', 'class 1'], normalize=True,
                          title='SVM Normalized confusion matrix ')

    plt.show()
    print(classification_report(ytt, yee))

    fpr, tpr, threshold = metrics.roc_curve(ytt, yee)
    auc = metrics.roc_auc_score(ytt, yee)
    i = float(("%0.4f" % auc))
    j = 'SVM AUC = ' + str(i)

    plt.plot(fpr, tpr)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(j)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()


conf_mat_logo_aggregated()
