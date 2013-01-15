from numpy import *
import operator
from os import listdir

def classify(x, dataSet, labels, k):
    numRows = dataSet.shape[0]
    difference = tile(x, (numRows,1)) - dataSet
    squareOfDifference = difference**2
    squareOfDistances = squareOfDifference.sum(axis=1)
    distances = squareOfDistances**0.5
    distanceSort = distances.argsort()     
    classCount={}          
    for i in range(k):
        label = labels[distanceSort[i]]
        classCount[label] = classCount.get(label,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def loadData(filename):
    fr = open(filename)
    lines = fr.readlines()[1:]
    numberOfLines = len(lines)
    matrix = zeros((numberOfLines,6))
    labels = []              
    index = 0
    for line in lines:
      # clean input
      line = line.strip()
      # remove name
      q1 = line.find('"')
      q2 = line.find('"', q1+1)
      line = line[:q1] + line[q2+1:]

      listFromLine = line.split(',')
      survival,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked = listFromLine
      if not age:
        continue #skip age when null
      sex = (1 if sex == 'male' else 0)
      listFromLine = [int(pclass),sex,float(age),int(sibsp),int(parch),float(fare)]
      matrix[index,:] = listFromLine
      labels.append(int(survival))
      index += 1
    return matrix,labels
    
def normalize(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalized = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalized = dataSet - tile(minVals, (m,1))
    normalized = normalized/tile(ranges, (m,1))   #element wise divide
    return normalized, ranges, minVals
   
def translateLabel(indicator):
  return ('survived' if indicator == 1 else 'died')

def test():
    hoRatio = 0.50      #hold out 10%
    mat,labels = loadData('csv/train.csv')       #load data setfrom file
    normalized, ranges, minVals = normalize(mat)
    size = normalized.shape[0]
    numTests = int(size*hoRatio)
    errorCount = 0.0
    k = 8 
    for i in range(numTests):
        classifierResult = classify(normalized[i,:],normalized[numTests:size,:],labels[numTests:size],k)
        print "classifier thinks: %s, real answer is: %s" % (translateLabel(classifierResult), translateLabel(labels[i]))
        if (classifierResult != labels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTests) * 100)
    print errorCount
