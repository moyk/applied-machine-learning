import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
from scipy.ndimage import imread
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression

def loadData():
    train_labels, train_data = [], []
    for line in open('./faces/train.txt'):
        im = misc.imread(line.strip().split()[0])
        train_data.append(im.reshape(2500, ))
        train_labels.append(line.strip().split()[1])
    train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
    # print train_data.shape, train_labels.shape
    # plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
    # plt.show()


    test_labels, test_data = [], []
    for line in open('./faces/test.txt'):
        im = misc.imread(line.strip().split()[0])
        test_data.append(im.reshape(2500, ))
        test_labels.append(line.strip().split()[1])
    test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)
    print test_data.shape, test_labels.shape
    return train_data,train_labels,test_data,test_labels


def lowRankApproxi(train_data,U,s,V):
    rRange = [i for i in range(1,200)]
    errorList = []
    sDiag = np.diag(s)
    for r in rRange:
        tmp = np.matmul(sDiag[:r,:r],V[:r,:])
        XR = np.matmul(U[:,:r],tmp)
        error = LA.norm(np.subtract(train_data,XR), 'fro')
        errorList.append(error)
    plt.plot(range,errorList)
    plt.show()


def generateFeature(r, X,V):
    F = np.matmul(X, np.transpose(V[:r,:]))
    return F


def hw1():
    train_data, train_labels, test_data, test_labels = loadData()
    sumFace = np.sum(train_data, axis=0)
    averageFace = sumFace / 540
    meanSubTrain=np.subtract(train_data, averageFace)
    meanSubTest = np.subtract(test_data, averageFace)
    plt.imshow(meanSubTrain[10, :].reshape(50,50), cmap = cm.Greys_r)
    plt.show()
    plt.imshow(meanSubTest[10, :].reshape(50,50), cmap = cm.Greys_r)
    plt.show()
    U, s, V = np.linalg.svd(train_data, full_matrices=True)
    for i in range(10):
        plt.imshow(V[i, :].reshape(50, 50), cmap=cm.Greys_r)
        plt.show()
        plt.savefig("eigenFace"+str(i)+".png")
    lowRankApproxi(train_data,U,s,V)
    accuracyList = []
    for r in range(10,11):
        Ftrain = generateFeature(r,train_data,V)
        Ftest = generateFeature(r, test_data, V)
        res = hw1H(Ftrain,Ftest,train_labels,test_labels)
        if r == 10:
            print res
        accuracyList.append(res)
    plt.plot([i for i in range(1,200)],accuracyList)


def hw1H(Ftrain,Ftest,train_labels,test_labels):
    LogReg = LogisticRegression(multi_class='ovr')
    LogReg.fit(Ftrain, train_labels)
    accuracy = LogReg.score(Ftest,test_labels)
    return accuracy


hw1()
