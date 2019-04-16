import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle

import statsmodels.formula.api as smf
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
data= pd.read_csv('../data-standard.csv')
features = ['RMSSD', 'SDNN', 'SDANN', 'SDANNi', 'SDSD', 'pNN50', 'AutoCorrelation']


X, Y = np.array(data[features]), np.array(data['sleep'])
print 'Null accuracy sleep:',max(data['sleep'].mean(), 1-data['sleep'].mean())

model_stat = smf.Logit(Y, data[features])

result = model_stat.fit()
print(result.summary())
# print result.params

# print(list(zip(result.coef_, X)))


grp = data['id'].values
# model = LogisticRegression()

logo = LeaveOneGroupOut()
cv = LeaveOneGroupOut().split(X, Y, grp)
scores = []
scores_2 =[]
for train, test in logo.split(X, Y, grp):
    model=LogisticRegression()
    x_train, x_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    model.fit(x_train, y_train.ravel())

    scores.append(metrics.accuracy_score(y_test, model.predict(x_test)))

print ("Logistic Regression Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), np.std(scores))
scores_cross = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print('Logistic Regression Cross Validation Score: {0:.3f} (+/-{1:.3f})').format(np.mean(scores_cross), np.std(scores_cross))


def conf_mat_logo_aggregated():
    grp = data['id'].values

    logo = LeaveOneGroupOut()
    i = 0
    ytt = []
    yee = []
    for train, test in logo.split(X, Y, grp):
        model = LogisticRegression()
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
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['class 0', 'class 1'],
    #                       title='Logistic Regression matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['class 0', 'class 1'], normalize=True,
                          title='Logistic Regression Normalized confusion matrix ')

    plt.show()

    print(classification_report(ytt, yee))

    fpr, tpr, threshold = metrics.roc_curve(ytt, yee)
    auc = metrics.roc_auc_score(ytt, yee)
    i = float(("%0.4f" % auc))
    j = ' Logistic Regression AUC = ' + str(i)

    plt.plot(fpr, tpr)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(j)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()


conf_mat_logo_aggregated()
