import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = pd.read_csv('training.csv', delimiter=',')
iris2 = pd.read_csv('testing.csv', delimiter=',')
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']
X = X.to_numpy()
y = y.to_numpy()

class Node():
    def __init__(self, featureIndex=None, threshold=None, left=None, right=None, infoGain=None, value=None):
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.infoGain = infoGain
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, minSamplesSplit=2, maxDepth=2):
        self.root = None
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        
    def buildTree(self, dataset, currDepth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        numSamples, numFeatures = np.shape(X)
        if numSamples>=self.minSamplesSplit and currDepth<=self.maxDepth:
            bestSplit = self.getBestSplit(dataset, numSamples, numFeatures)
            if bestSplit["infoGain"]>0:
                leftSubtree = self.buildTree(bestSplit["datasetLeft"], currDepth+1)
                rightSubtree = self.buildTree(bestSplit["datasetRight"], currDepth+1)
                return Node(bestSplit["featureIndex"], bestSplit["threshold"], 
                            leftSubtree, rightSubtree, bestSplit["infoGain"])
        
        leafValue = self.calculateLeafValue(Y)
        return Node(value=leafValue)
    
    def getBestSplit(self, dataset, numSamples, numFeatures):
        bestSplit = {}
        maxInfoGain = -float("inf")
        for featureIndex in range(numFeatures):
            featureValues = dataset[:, featureIndex]
            possibleThresholds = np.unique(featureValues)
            for threshold in possibleThresholds:
                datasetLeft, datasetRight = self.split(dataset, featureIndex, threshold)
                if len(datasetLeft)>0 and len(datasetRight)>0:
                    y, leftY, rightY = dataset[:, -1], datasetLeft[:, -1], datasetRight[:, -1]
                    currInfoGain = self.informationGain(y, leftY, rightY)
                    if currInfoGain>maxInfoGain:
                        bestSplit["featureIndex"] = featureIndex
                        bestSplit["threshold"] = threshold
                        bestSplit["datasetLeft"] = datasetLeft
                        bestSplit["datasetRight"] = datasetRight
                        bestSplit["infoGain"] = currInfoGain
                        maxInfoGain = currInfoGain
                        
        return bestSplit
    
    def split(self, dataset, featureIndex, threshold):        
        datasetLeft = np.array([row for row in dataset if row[featureIndex]<=threshold])
        datasetRight = np.array([row for row in dataset if row[featureIndex]>threshold])
        return datasetLeft, datasetRight
    
    def informationGain(self, parent, lChild, rChild):        
        weightL = len(lChild) / len(parent)
        weightR = len(rChild) / len(parent)
        gain = self.entropy(parent) - (weightL*self.entropy(lChild) + weightR*self.entropy(rChild))
        return gain
    
    def entropy(self, y):        
        classLabels = np.unique(y)
        entropy = 0
        for cls in classLabels:
            pCls = len(y[y == cls]) / len(y)
            entropy += -pCls * np.log2(pCls)
        return entropy

    def calculateLeafValue(self, Y):        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def printTree(self, tree=None, indent=" "):        
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.featureIndex), "<=", tree.threshold, "?", tree.infoGain)
            print("%sleft:" % (indent), end="")
            self.printTree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.printTree(tree.right, indent + indent)
    
    def fit(self, X, Y):        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)
    
    def predict(self, X):        
        preditions = [self.makePrediction(x, self.root) for x in X]
        return preditions
    
    def makePrediction(self, x, tree):        
        if tree.value!=None: return tree.value
        featureVal = x[tree.featureIndex]
        if featureVal<=tree.threshold:
            return self.makePrediction(x, tree.left)
        else:
            return self.makePrediction(x, tree.right)


X = iris.iloc[:, :-1].values
Y = iris.iloc[:, -1].values.reshape(-1,1)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=.5, random_state=50)

classifier = DecisionTreeClassifier(minSamplesSplit=3, maxDepth=3)
classifier.fit(xTrain,yTrain)
print('Decision Tree: ')
classifier.printTree()

yPred = classifier.predict(xTest) 

print("\nTraining: ")
for x in yTrain:
   print(*x, end='')
print('\n')
ints = [int(x) for x in yPred]
print("Predicted: \n", *ints, sep='')
print("\nAccuracy: ", accuracy_score(yTest, yPred))

print('-----------------------------')
print('New Testing Data:')
exampleTestingData = iris2.values
examplePredicitions = classifier.predict(exampleTestingData)
intPredictions = [int(x) for x in examplePredicitions]
print(*intPredictions, sep='')


