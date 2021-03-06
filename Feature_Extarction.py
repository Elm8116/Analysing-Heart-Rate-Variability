import pandas as pd
import numpy as np

Const_GrpMinutes = 5


def get_data(filename):
    data = pd.read_csv(filename, skiprows=2)
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

    dummyMin = []
    LastIndx = data.minute.last_valid_index()
    lastMin = data.get_value(LastIndx, 'minute') + 1
    lastMin = lastMin.astype(int)
    if (lastMin % Const_GrpMinutes != 0):
        for lsm in range(lastMin - (lastMin % Const_GrpMinutes), lastMin):
            dummyMin.append(lsm)
    data_n = data.query('minute != @dummyMin')
    # data_n.to_csv('./dataset/time.csv')
    return data_n


def comulative_time(data):

    data['five_minute'] = np.floor(data.minute / Const_GrpMinutes)
    data['caption_five_min']= (data.five_minute )* Const_GrpMinutes
    return data.five_minute


def Sequenc_id(data):
    s = []
    len_seq_id = len(data.groupby('five_minute'))
    for i in range(len_seq_id):
        s.append(i+1)
    data['sequence_id'] = pd.Series(s)
    return data.sequence_id, len_seq_id


def individual_id_activity(data,split_name):
    data['individual_id'] = split_name
    return data.individual_id

def RMSSD_Calculation(data):
    RR_list = data['Artifact_corrected_RR']
    dif_RR = []
    RR_sqdiff = []
    i = 0
    while (i<len(RR_list) - 1):
        dif_RR.append(abs(RR_list[i + 1] - RR_list[i]))
        RR_sqdiff.append(math.pow(dif_RR[i], 2))
        i += 1
    sqrt = lambda x: np.sqrt(np.mean(x))
    data['RR_sqdiff'] = pd.Series(RR_sqdiff)
    data['RMSSD'] = pd.Series(data.groupby('five_minute').RR_sqdiff.apply(sqrt))

    return data.RMSSD

# /Standard Deviation of 5 minutes intervals/
def SDNN_Calculation(data):
    sdnn=pd.Series(data.groupby('five_minute').Artifact_corrected_RR.std())
    data['SDNN']=pd.Series(sdnn)
    return data.SDNN


def SDANN_Calculation(data):
    sdann = []
    sdnn = pd.Series(data.groupby('minute').Artifact_corrected_RR.mean())
    data['Sdnn_one_min'] = pd.Series(sdnn)
    index = data.Sdnn_one_min.last_valid_index() + 1
    print("index in SDANN",index)

    j = Const_GrpMinutes
    for i in range(0, index, Const_GrpMinutes):
        if (i + Const_GrpMinutes < index):
            j = Const_GrpMinutes
        else:
            j = index - i

        sdann.append(data.Sdnn_one_min[i:i + j].std())
    print("sdann",sdann)
    data['SDANN'] = pd.Series(sdann)
    return data.SDANN


def SDNNi_Calculation(data):
    sdnni = []
    sdnn = pd.Series(data.groupby('minute').Artifact_corrected_RR.std())
    data['mean_one_min'] = pd.Series(sdnn)
    index = data.mean_one_min.last_valid_index() + 1
    print("index in SDNNi",index)
    for i in range(0, index, Const_GrpMinutes):
        if (i + Const_GrpMinutes < index):
            j = Const_GrpMinutes
            # print("test  SDNNi", data.mean_one_min[i:i + j])
        else:
            j = index - i
            # print ("test SDNNi",data.mean_one_min[i:i + j])
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


# /The percentage of differences greater than x (pNNx) calculates
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
    # calculate auto corrolation
    for lag in SeqIdCnt:
        # print(lag)
        auto.append(data['Artifact_corrected_RR'].autocorr(lag))
    data['AutoCorrelation']=pd.Series(auto)
    return data.AutoCorrelation


path_sleep="./dataset/sleep/*.csv"
allFiles_sleep = glob.glob(path_sleep)
i = 0

for fname in allFiles_sleep:
    data = get_data(fname)
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
    seq_id, len_seq = Sequenc_id(data)

    split_n = individualt_id_activity(data, fname)

    data = data.loc[0:len_seq-1, ["individual_id", "sequence_id",
                                   "RMSSD", "SDNN", "SDANN", "SDANNi", "SDSD", "pNN50", "AutoCorrelation"]]
    data.to_csv('./dataset/sleep_features/file' + str(i) + '.csv', index=False)
    i += 1






