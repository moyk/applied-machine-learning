import csv
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from numpy import linalg as LA
import bisect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def nothing():
    filePath = "/Users/kellywang/Documents/CornellTech/AppliedMachineLearning/hm1/train.csv"
    trainLabel = []
    trainData = []
    questionelist=[]
    with open(filePath, 'rb') as csvfile:
        traintxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
        j = 0
        for row in traintxt:
            row = row[0].split(",")
            label = row[0]
            pixelVal = row[1:]
            if j != 0:
                trainLabel.append(int(label))
                pixelVal = [float(i) for i in pixelVal]
                pixelVal = np.asarray(pixelVal)
                digit = pixelVal.reshape(28,28)
                trainData.append(digit)
                # fig = plt.figure()
                # plt.imshow(digit,cmap="gray")
                # plt.show()
                # fig.savefig("digit"+ label +".png")
            if label=='1'or label=='0':
                questionelist.append(j-1)
            j +=1
    digitDict = defaultdict(int)
    for each in trainLabel:
        digitDict[each] +=1

    # histogram(trainLabel,digitDict)
    #findBestMatch(trainLabel,trainData)
    questionE(trainLabel,trainData, questionelist)


def histogram(trainLabel, digitDict):

    normArray  = [0 for i in range(10)]
    for digit in digitDict:
        norm_num = digitDict[digit]/float(42000)
        normArray[int(digit)] = norm_num

    # trainLabel=[0,1,2,3,1,2,3,3,3,4,5,6,7,8,8,8,7,5,6,2,9]
    normArray = np.asarray(normArray)
    fig = plt.figure()
    plt.hist(np.asarray(trainLabel), bins = 10, normed = True,ec='black')
    plt.title("histogram of digit counts")
    plt.show()
    fig.savefig("histogram.png")

def findBestMatch(trainLabel,trainData):
    samplesDict = {}
    for i, label in enumerate(trainLabel[0:30]):
        if label not in samplesDict:
            samplesDict[label] = i
    print samplesDict
    res = []
    for eachDig in range(10):
        min_dist = 100000
        min_index = -1
        pixels = trainData[samplesDict[str(eachDig)]]
        # tempData =trainData[:samplesDict[str(eachDig)]] + trainData[samplesDict[str(eachDig)]+1:]
        for i, compare in enumerate(trainData):
            if i==samplesDict[str(eachDig)]:
                continue
            dist = LA.norm(pixels - compare)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        res.append(min_index)
    print res
    plotNearSample(res,samplesDict,trainData)


def plotNearSample(res,samplesDict,trainData):
    for i in range(10):
        sample = samplesDict[str(i)]
        min_dist_sample = res[i]
        a = trainData[sample]
        b = trainData[min_dist_sample]
        fig = plt.figure()
        kelly = fig.add_subplot(1, 2, 1)
        plt.imshow(np.asarray(a),cmap='gray')
        kelly = fig.add_subplot(1, 2, 2)
        plt.imshow(np.asarray(b),cmap='gray')
        plt.show()
        # res




def questionE(trainLabel,trainData,questionelist):

    GenuineDis=[]
    ImposterDis=[]
    for i in range(len(questionelist)):
        for j in range(i+1, len(questionelist)):
            a = trainData[questionelist[i]]
            b = trainData[questionelist[j]]
            dist = LA.norm(a - b)
            # print trainLabel[questionelist[i]], trainLabel[questionelist[j]]
            if trainLabel[questionelist[i]]==trainLabel[questionelist[j]]:
                GenuineDis.append(dist)
                # plt.imshow()
            else:
                ImposterDis.append(dist)
    print("guess rate: ", len(ImposterDis) / (len(ImposterDis) + len(GenuineDis)))
    # fig = plt.figure()
    # plt.title("histogram of Digit 0 and 1")
    #
    # plt.hist(np.asarray(ImposterDis), bins=200, label='imposter', alpha=0.5)
    # plt.hist(np.asarray(GenuineDis), bins=200, label='genuine', alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.show()
    # fig.savefig("histogram2.png")
    generate_roc(ImposterDis,GenuineDis)



def generate_roc(ImposterDis,GenuineDis):
    tpr, fpr = [], []
    ImposterDis.sort()
    GenuineDis.sort()

    maxDistance = max(ImposterDis[len(ImposterDis)-1],GenuineDis[len(GenuineDis)-1])
    for th in range(0,int(maxDistance),1):
        tpr.append(bisect.bisect_left(GenuineDis,th) / float(len(GenuineDis)))
        fpr.append( bisect.bisect_left(ImposterDis,th) / float(len(GenuineDis)))

    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    eer = get_eer(tpr, fpr)


def get_eer(tpr, fpr):
    for i in range(0,len(tpr),20):
        if round(1-tpr[i],2) == round(fpr[i],2):
            return fpr[i]


def questionHversion():
    filePath = "train.csv"
    trainLabel = []
    trainData = []
    with open(filePath, 'rb') as csvfile:
        traintxt = csv.reader(csvfile, delimiter=' ', quotechar='|')
        j = 0
        for row in traintxt:
            row = row[0].split(",")
            label = row[0]
            pixelVal = row[1:]
            if j != 0:
                trainLabel.append(label)
                pixelVal = [float(i) for i in pixelVal]
                pixelVal = np.asarray(pixelVal)
                # digit = pixelVal.reshape(28,28)
                trainData.append(pixelVal)
            j += 1
    neigh = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(neigh, trainData, trainLabel)
    print scores.mean()
    conf_matrix = obtain_confusion_matrix(neigh, trainData, trainLabel)
    print conf_matrix

def obtain_confusion_matrix(neigh,trainData,trainLabel):
    predicted = cross_val_predict(neigh, trainData, trainLabel)
    return confusion_matrix(trainLabel, predicted)


nothing()
# questionHversion()
