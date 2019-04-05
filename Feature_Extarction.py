import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import math
from sklearn import preprocessing ,model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import random
import seaborn as sns
import os
import glob
import csv
from statistics import median
from itertools import chain, product
import warnings
from itertools import islice

Const_GrpMinutes = 5

def get_data(filename):

    data = pd.read_csv(filename,skiprows=2)
    data.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
    data = data.fillna({'RR': 0}).dropna()

    return data

def CreateOneMin(data):
    Comulative_time_list = []
    RR_list = data['Artifact_corrected_RR']

    sum = 0
    i = 0

    while (i < len(RR_list)):
        sum = (sum + RR_list[i])
        Comulative_time_list.append(sum / 60000)
        i += 1

    data['Comulative_time'] = pd.Series(Comulative_time_list)
    data['minute'] = np.floor(data['Comulative_time'].values)

    DummyMin = []
    LastIndx = data.minute.last_valid_index()
    # print("LastIndx = ", LastIndx)
    lastMin = data.get_value(LastIndx, 'minute') + 1
    lastMin = lastMin.astype(int)
    print("lastMin=", lastMin)
    if (lastMin % Const_GrpMinutes != 0):
        for i in range(lastMin - (lastMin % Const_GrpMinutes), lastMin):
            DummyMin.append(i)
            # print("i= ", i)

    print("dummy = ", DummyMin)
    data = data.query('minute != @DummyMin')
    # data.to_csv('./dataset/time4.csv')
    return data


def comulative_time(data):

    data['five_minute'] = np.floor(data.minute / Const_GrpMinutes)
    seq_id = len(data.groupby('five_minute'))
    data['caption_five_min']= (data.five_minute )* Const_GrpMinutes
    # data.to_csv('./dataset/time402.csv')
    return data.five_minute


def Sequenc_id(data):

    s = []
    len_seq_id = len(data.groupby('five_minute'))
    for i in range(len_seq_id):
        s.append(i+1)
    data['sequence_id']=pd.Series(s)
    return data.sequence_id,len_seq_id

def patient_id_activity(data,split_name):


    data['patient_id'] =split_name
    return  data.patient_id

def RMSSD_Calculation(data):
    Comulative_time_list = []
    RR_list = data['Artifact_corrected_RR']
    dif_RR = []
    RR_sqdiff = []
    sum = 0
    i = 0
    while (i < len(RR_list) - 1):
        dif_RR.append(abs(RR_list[i + 1] - RR_list[i]))
        RR_sqdiff.append(math.pow(dif_RR[i], 2))
        i += 1

    sqrt = lambda x: np.sqrt(np.mean(x))
    data['RR_sqdiff'] = pd.Series(RR_sqdiff)
    data['RMSSD'] = pd.Series(data.groupby('five_minute').RR_sqdiff.apply(sqrt))
    #print("RMSSD: ", data.RMSSD)
    return data.RMSSD.mean()

# /Standard Deviation of 5 minutes intervals/
def SDNN_Calculation(data):

    sdnn=[]
    #RR_list = data['Artifact_corrected_RR']

    sdnn=pd.Series(data.groupby('five_minute').Artifact_corrected_RR.std())
    data['SDNN']=pd.Series(sdnn)
    return data.SDNN


def SDANN_Calculation(data):
    sdnn = []
    sdann = []
    sdnn = pd.Series(data.groupby('minute').Artifact_corrected_RR.mean())
    data['Sdnn_one_min'] = pd.Series(sdnn)
    index = data.Sdnn_one_min.last_valid_index() + 1
    print("index in SDANN",index)

    j=Const_GrpMinutes
    for i in range(0, index, Const_GrpMinutes):
        if (i + Const_GrpMinutes < index):
            j = Const_GrpMinutes
            print("test  SDNN", data.Sdnn_one_min[i:i + j])
        else:
            j = index - i
            print("test  SDNN", data.Sdnn_one_min[i:i + j])
        sdann.append(data.Sdnn_one_min[i:i + j].std())
    print("sdann",sdann)
    data['SDANN'] = pd.Series(sdann)
    return data.SDANN

def SDNNi_Calculation(data):
    sdnn = []
    sdnni = []
    sdnn = pd.Series(data.groupby('minute').Artifact_corrected_RR.std())
    data['mean_one_min'] = pd.Series(sdnn)
    index = data.mean_one_min.last_valid_index() + 1
    print("index in SDNNi",index)
    # Const_GrpMinutes = 5
    j = Const_GrpMinutes

    for i in range(0, index, Const_GrpMinutes):
        if (i + Const_GrpMinutes < index):
            j = Const_GrpMinutes
            print("test  SDNNi", data.mean_one_min[i:i + j])
        else:
            j = index - i
            print ("test SDNNi",data.mean_one_min[i:i + j])
        sdnni.append(data.mean_one_min[i:i + j].mean())
    print("SDNNi",sdnni)
    data['SDANNi'] = pd.Series(sdnni)

    return data.SDANNi

# /Standard Deviation differences in 5 minutes intervals/
def SDSD_Calculation(data):
    RR_list = data['Artifact_corrected_RR']
    diff_RR = []

    i = 0
    while (i < len(RR_list) - 1):
        diff_RR.append(abs(RR_list[i + 1] - RR_list[i]))
        i += 1
    data['diff_RR']=pd.Series(diff_RR)
    sdsd=pd.Series(data.groupby('five_minute').diff_RR.std())
    data['SDSD']=pd.Series(sdsd)

    return data.SDSD


#/ the percentage of differences greater than x (pNNx) calculates
def PNN_Calculation(data):

    RR_list = data['Artifact_corrected_RR']
    diff_RR = []
    i = 0
    while (i < len(RR_list) - 1):
        diff_RR.append(abs(RR_list[i + 1] - RR_list[i]))
        i += 1
    data['diff_RR'] = pd.Series(diff_RR)
    pnn50 = data.groupby('five_minute')['diff_RR'].apply(lambda x: x.gt(50).sum() / len(x))
    data['pNN50'] = pd.Series(pnn50)
    return data.pNN50


def autocorrelation_Calculation(data):
    seq_id = data['five_minute']
    # calculate number of each sequence id


    SeqIdCnt = np.bincount(np.array(seq_id.astype(int)))
    # remove sequence ID which the value is equal to zero
    SeqIdCnt = list(filter(lambda x: x != 0, SeqIdCnt))

    auto = []
    # calculate auto corrolation based on through
    for lag in SeqIdCnt:
        # print(lag)
        auto.append(data['Artifact_corrected_RR'].autocorr(lag))
    data['AutoCorrelation']=pd.Series(auto)
    return data.AutoCorrelation


path_sleep="./dataset/sleep(new)/*.csv"
allFiles_sleep = glob.glob(path_sleep)
i=0





for fname in allFiles_sleep:

    data = get_data(fname)

    print(fname)
    split_name = fname.split("/")
    activity = split_name[0].split("\\")

    data = CreateOneMin(data)

    minutes = comulative_time(data)
    Rmssd = RMSSD_Calculation(data)
    sDNN = SDNN_Calculation(data)
    sDANN = SDANN_Calculation(data)
    sDNNi = SDNNi_Calculation(data)
    sDsD = SDSD_Calculation(data)
    pNN50 = PNN_Calculation(data)
    auto = autocorrelation_Calculation(data)
    seq_id ,len_seq = Sequenc_id(data)



    split_n = patient_id_activity(data,fname)



    data = data.loc[0:len_seq-1 , ["patient_id","sequence_id",
                      "RMSSD","SDNN","SDANN","SDANNi","SDSD","pNN50","AutoCorrelation"]]
    data.to_csv('./dataset/test8/file'+ str(i) +'.csv', index=False)
    i+=1
    # break





