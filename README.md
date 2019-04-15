---
author:
- 
---

Heart Rate Variability Analysis
-------------------------------

Heart rate variability is evaluated by a number of methods which are
categorized as time-domain, spectral or frequency domain, geometric, and
nonlinear methods. This study concentrates on time-domain measurements,
which are used in most research studies of HRV analysis. The time-domain
measures the change in normal R wave to normal R wave (NN) intervals
over time and states the activity of circulation system
[@achten2003heart].

Statistical HRV Features {#features}
------------------------

Seven commonly used statistical time-domain parameters
[@camm1996heart][@stein1994heart] which are calculated from HRV
segmentation during 5-minute recording windows as proposed in previous
study [@malik1996heart], comprised of RMSSD, SDNN, SDANN, SDANNi, SDSD,
PNN50, and AutoCorrelation, are considered in this study. Each of these
HRV assessment techniques is described in Table
[\[table:time-domain\]](#table:time-domain){reference-type="ref"
reference="table:time-domain"} and the detail formula of them is
described in more detail by the following equations.

  **Parameter**     **Unit**   **Description**
  ----------------- ---------- ------------------------------------------------------
  RMSSD             ms         The root-mean-square of successive differences
  SDNN              ms         The standard deviation
  SDANN             ms         The standard deviation of mean values of intervals
  SDANNi            ms         The mean standard deviation of intervals
  SDSD              ms         The standard deviation of differences
  PNN50             \%         The percentage of differences greater than 50 (ms)
  AutoCorrelation              the correlation of successive intervals, called lags

  : Time domain measures of HRV[]{label="table:time-domain"}

Suppose that $R_i, i = 1, 2, .., N$ be the time intervals between
successive R points of a heartbeat signal. (I.e., $R_i$ is the interval
between the $i$th R point and the $i+1$st R point.) Each of the measures
below is typically computed over a fixed-size window, e.g. 5 minutes.

1.  RMSSD refers to the root mean square differences of adjacent R-R
    intervals in a window. $$\label{rmssd}
       \mathit{RMSSD}  =  \sqrt{ \frac{1}{N-1}  \sum_{i=1} ^{N -1} (R_{i+1} - R_{i})^2   }$$

2.  SDNN refers to the standard deviation of the R-R intervals in a
    window.

    $$\mathit{SDNN} =  \sqrt{ \frac{1}{N}   \sum_{i=1}^N (R_i -  \overline{R})^2  }$$
    where $\overline{R}$ (ms) is the arithmetic mean value of the normal
    R-R intervals computed as follow:
    $$\overline{R} = \frac{1}{N} \sum_{i=1}^N R_i$$

3.  SDANN is the standard deviation of average values of consecutive
    R---R intervals in a window.

4.  SDANNi is defined by the mean standard deviation of consecutive
    R---R intervals within a window.

5.  SDSD refers to the standard deviation of differences between the
    successive R---R intervals within a window. $$\label{sdsd}
      \mathit{SDSD} =   \sqrt{ \frac{1}{N}   \sum_{i=1}^N (dR_i -  \overline{d}R)^2  }$$
    where $dR_i= R_{i+1}-R_i$ and $\overline{d}R$ is the mean value of
    all $dR_i$

6.  PNN50 calculates percentage of the differences between consecutive
    R-R intervals which are greater than a $50$ ms.

7.  AutoCorrelation $$\label{corr}
         \mathit{CORR} ( \tau ) =  \frac{ \sum_{i=1}^{N-\tau} (R_i - \overline{R})  (R_{i+\tau} -  \overline{R}) }{ \sum_{i=1}^N (R_i -  \overline{R})^2  }$$
    where $\tau$ is a time lag
