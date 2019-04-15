
Classiﬁcation of Activity using Heart Rate Variability
===============================

Classiﬁeries based on analysis of HRV signal are developed to classify different activities including sleep, exam, and exercise.

Heart Rate Variability Analysis
------------------------

Heart rate variability is evaluated by a number of methods which are
categorized as time-domain, spectral or frequency domain, geometric, and
nonlinear methods. This study concentrates on time-domain measurements.The time-domain
measures the change in normal R wave to normal R wave (NN) intervals
over time and states the activity of circulation system.

Segmenting HRV Recordings by Activity
------------------------

Seven commonly used statistical time-domain parameters which are
calculated from HRV segmentation during 5-minute recording windows
including RMSSD, SDNN, SDANN, SDANNi, SDSD, PNN50, and
AutoCorrelation, are considered in this study. Each of these HRV
assessment techniques is described in the following table.
<pre>
  Parameter        Unit        Description
  ----------------- ---------- ------------------------------------------------------
  RMSSD             ms         The root-mean-square of successive differences
  SDNN              ms         The standard deviation
  SDANN             ms         The standard deviation of mean values of intervals
  SDANNi            ms         The mean standard deviation of intervals
  SDSD              ms         The standard deviation of differences
  PNN50             %          The percentage of differences greater than 50 (ms)
  AutoCorrelation              The correlation of successive intervals, called lags
 
</pre>

Binary Classiﬁcation of Activities
------------------------

The resulting 5-minute windows were then used for binary classiﬁcation of each activity by machine learning techniques.
Activities are labeled as sleep or not sleep, exam or not exam, exercise or not exercise.
For each individual, a unique identiﬁcation number (id) is assigned. Different statistical features are computed as discussed before and recorded in a comma separated value ﬁle.
The table consists of person id, features and the corresponding window id which indicates ﬁve minute intervals.
Logistic regression, Support vector machine, decision tree, and random forest are the methods that have been applied in this experiment to classify activities.

Logistic regression (LR), Support Vector Machines (SVM), Decision Trees (DT), and Random Forests (RF) were each applied to the dataset of 39 individuals with a total of 1832 5-minute activity 
windows. In all cases, the suite of features are used as described in Appendix A. All evaluation used leave-one-participant-out cross validation.

Standard LR was used with no regularization; hence no tuning was required.
For SVM, DT and RF, hyper-parameters were tuned with leave-one-participant out cross validation method in conjunction with the HORD algorithm which searches for the optimal hyperparameters using a combination of kernel regression and dynamic coordinate search.

