import numpy as np
from os import listdir

def createData():
    grades = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return grades, labels


def classify0(X, dataset, labels, k):
    size = dataset.shape[0]
    dis = np.sqrt(np.sum(np.square(dataset-X), axis=1))
    sort_index = np.argsort(dis)

    label_count = {}
    for i in range(k):
        label = labels[sort_index[i]]
        label_count[label] = label_count.get(label, 0)+1
    sort_label = sorted(label_count.items(), key=lambda d: d[1], reverse=True)
    
    return sort_label[0][0]

def getLabel(label):
    if label == 'didntLike':
        return 0
    elif label == 'smallDoses':
        return 1
    elif label == 'largeDoses':
        return 2
    else:
        return int(label)

def setLabel(label):
    if label == 0:
        return 'didntLike'
    elif label == 1:
        return 'smallDoses'
    else:
        return 'largeDoses'

def readFile(path):
    f = open(path)
    lines = f.readlines()
    num = len(lines)
    mat = np.zeros((num, 3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        mat[index, :] = line[0:3]
        labels.append(getLabel(line[-1]))
        index += 1
    return mat, labels

def autoNorm(data):
    minVals = np.min(data, axis=0)
    maxVals = np.max(data, axis=0)
    ranges = maxVals - minVals
    data = (data - minVals) / ranges
    return data, minVals, ranges

def datingLabelTest(filePath, rate, k, printPredict):
    data, labels = readFile(filePath)
    normMat, minVals, ranges = autoNorm(data)
    n = normMat.shape[0]
    newIndexs = np.random.permutation(n)
    data = data[newIndexs, :]
    labels = [labels[newIndexs[i]] for i in newIndexs]
    
    train_n = int(n*rate)
    test_X = data[:train_n]
    test_y = labels[:train_n]
    train_X = data[train_n:]
    train_y = labels[train_n:]
    
    err_count = 0
    for i in range(train_n):
        result = classify0(test_X[i], train_X, train_y, k)
        if printPredict:
            print('predict: %s, real: %s'%(setLabel(result), setLabel(test_y[i])))
        if result != test_y[i]: err_count += 1
    print('k: %d, rate: %.2f, error rate: %f%%'%(k, rate, err_count/n*100))
    
def classifyPerson():
    labelList = ['not at all', 'in small doses', 'in large doses']
    game = float(input('percentage of time spent playing video games?'))
    fly = float(input('frequent flier miles earned per years?'))
    ice = float(input('liters of ice cream consumed per years?'))
    data, labels = readFile('01KNN_dating2.txt')
    data, minVals, ranges = autoNorm(data)
    per = np.array([fly, game, ice])
    result = classify0((per - minVals)/ranges, data, labels, 3)
    print('You will probably like this person: ', labelList[result])

def digVector(filename):
    vect = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lines = f.readline()
        for j in range(32):
            vect[0, 32*i+j] = int(lines[j])
    return vect

def classifyDigits():
    trainList = listdir('trainingDigits')
    nTrain = len(trainList)
    train_X = np.zeros((nTrain, 1024))
    train_y = []
    for i in range(nTrain):
        filename = trainList[i]
        train_X[i, :] = digVector('trainingDigits/%s'%filename)
        train_y.append(int(filename.split('.')[0].split('_')[0]))

    testList = listdir('testDigits')
    nTest = len(testList)
    error_count = 0
    for i in range(nTest):
        filename = testList[i]
        label = int(filename.split('.')[0].split('_')[0])
        test = digVector('testDigits\%s'%filename)
        result = classify0(test, train_X, train_y, 3)
        #print('predict: %d, real: %d'%(result, label))
        if result != label: error_count += 1
    print('the error rate: ', error_count/float(nTest))
    










