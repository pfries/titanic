from numpy import *
import operator
import csv as csv

def loadDataSet():
  csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) 
  header = csv_file_object.next() #Skip the header
  data=[]
  labels=[]
  for row in csv_file_object: 
    r = []
    r.append(row[0])
    r.append(row[1])
    r.append((1,0)[row[3] == 'male'])
    r.append(row[4])
    r.append(row[5])
    r.append(row[6])
    r.append(row[8])
    data.append(r) 
    labels.append(row[0])
  data = array(data) #convert from a list to an array
  return data, labels

def file2matrix(filename):
  fr = open(filename)
  nlines = len(fr.readlines())
  returnMat = zeros((nlines,3))
  classLabelVector = []
  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip()
    listFromLine = line.split(',')
    returnMat[index,:] = listFromLine[0:3]
    classLabelVector.append(int(listFromLine[-1]))
    index += 1
  return returnMat,classLabelVector

def normalize(dataSet):
  minVals = dataSet.min(0)
  maxVals = dataSet.max(0)
  ranges = maxVals - minVals
  normalized = zeros(shape(dataSet))
  m = dataSet.shape[0]
  normalized = dataSet - tile(minVals, (m,1))
  normalized = normalized/tile(ranges, (m,1))
  return normalized, ranges, minVals

def classify(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  diffMatrix = tile(inX, (dataSetSize,1)) - dataSet
  sqDiffMatrix = diffMatrix**2
  sqDistances = sqDiffMatrix.sum(axis=1)
  distances = sqDistances**0.5
  sortedDistanceIndices = distances.argsort()
  classCount={}
  for i in range(k):
    voteILabel = labels[sortedDistanceIndices[i]]
    classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
  sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]


