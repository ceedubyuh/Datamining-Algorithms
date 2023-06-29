import operator
import os
import numpy as np

#Pseudocode for classifying
#For each inputted vector:
#1. Calculate distance between a new vector (x) and current vector 
#2. Sort all distances in ascending order
#3. Take K with lowest distances to x
#4. Find the majority classes
#5. Return that majority as the prediction for x
def classify(x, data, labels, k):
    dataSize = data.shape[0]
    difference = np.tile(x, (dataSize,1)) - data
    sqDiff = difference**2
    sqDistances = sqDiff.sum(axis=1)
    distances = sqDistances**0.5
    sortedDist = distances.argsort()     
    classes={}          
    for i in range(k):
        votes = labels[sortedDist[i]]
        classes[votes] = classes.get(votes,0) + 1
    sortedClass = sorted(classes.items(), 
        key=operator.itemgetter(1), 
        reverse=True)
    return sortedClass[0][0]
    
def vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingTest():
    trainingLabels = []
    trainingFileList = os.listdir('trainingData')
    trainLength = len(trainingFileList)
    trainingMat = np.zeros((trainLength,1024))
    for x in range(trainLength):
        #Get file names in directory
        fileName = trainingFileList[x]
        #Open the directory
        for file in os.listdir('trainingData'):
            #Open and read every file
            with open(os.path.join('trainingData', file), 'r') as f:
                #Read file into 'text'
                text = f.read()
                #Set correctAnswer to the first character
                correctAnswer = text[0]
                #Append each correctAnswer to array of training labels
                trainingLabels.append(correctAnswer)
        trainingMat[x,:] = vector('trainingData/{}'.format(fileName))
    testFileList = os.listdir('testData')
    error = 0.0
    answerLabels = []
    testLength = len(testFileList)
    for y in range(testLength):
        #Get file names in directory
        fileName = testFileList[y]
        #Open the directory
        for file in os.listdir('testData'):
            #Open and read every file
            with open(os.path.join('testData', file), 'r') as f:
                #Read file into 'text'
                text = f.read()
                #Set correctAnswer to the first character
                correctAnswer = text[0]
                #Append each correctAnswer to array of testing labels
                answerLabels.append(correctAnswer)
        testVector = vector('testData/{}'.format(fileName))
        result = classify(testVector, trainingMat, trainingLabels, 4)
        #Print results from each and compare if trainingLabels and answerLabels match
        print("The KNN algorithm guessed: {}, the actual number was: {}".format(result, answerLabels[y]))
        if (result != answerLabels[y]): error += 1.0
    print("\nThe total accuracy is:", 1-(error/float(testLength)))

handwritingTest()