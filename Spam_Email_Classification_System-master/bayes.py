'''
classifying spam email with naive Bayes
@author: Evan Sun
'''
import numpy as np

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) 
    numWords = len(trainMatrix[0]) 
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords) 
    p1Num = np.ones(numWords)   
    #denominator
    p0Denom = 2.0  
    p1Denom = 2.0                        
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          
    p0Vect = np.log(p0Num/p0Denom)          
                                           
    return p0Vect,p1Vect,pAbusive  

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)   
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList) 
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

#hold-out cross validation

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        #read spam email
        wordList = textParse(open('email/spam/%d.txt' % i,"rb").read().decode('GBK','ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #mark spam email
        wordList = textParse(open('email/ham/%d.txt' % i,"rb").read().decode('GBK','ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) 
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet))) 
        testSet.append(trainingSet[randIndex]) 
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []  
    for docIndex in trainingSet: 
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:     
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print( "classification error",docList[docIndex]) 
    print ('the accuracy rate is: ',1.0-float(errorCount)/len(testSet)) 
spamTest();
