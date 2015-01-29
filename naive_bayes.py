# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import math


class NaiveBayes(Classifier):
    """A naïve Bayes classifier."""
    def __init__(self, model = {}):
        super(NaiveBayes, self).__init__(model)
        self.wordPro = {}
        self.dicCount = {}
        self.wordCount = {}
        self.attri = set()
        self.priorScore = {}
        
    def get_model(self): pass
    def set_model(self, model): pass
    model = property(get_model, set_model, "navie bayes classifier")
    
    #calculate the label
    def getdicCount(self, label):
        if label in self.dicCount.keys():
            self.dicCount[label] = self.dicCount[label] + 1
        else:
            self.dicCount[label] = 1
            
    #(male,feature),(female,feature)
    def calculateWordPro(self, label, feature):
        
        for f in feature.items():
            #pro = {[label, word], num}
            if (label, f) in self.attri:
                self.wordPro[(label, f)] = self.wordPro[(label, f)] + 1
            else:
                self.attri.add((label, f))
                self.wordPro[(label, f)] = 1
            
            if label in self.wordCount:
                self.wordCount[label] = self.wordCount[label] + 1
            else:
                self.wordCount[label] = 1
                
    #得到的{[label, word], probability}
    def getWordPro(self):
        for key in self.wordPro.keys():
            self.wordPro[key] = float(self.wordPro[key])/(self.wordCount[key[0]])
    
    def getPrior(self):
        for key in self.dicCount.keys():
            self.priorScore[key] = math.log(self.dicCount[key]/float(self.totalNum))
            
    #abstractmethod
    def train(self, instances):
        self.totalNum = 0
        
        for instance in instances:
            self.totalNum =self.totalNum + 1        
            label = instance.label                  
            self.getdicCount(label)                 
            feature = instance.features()           
            self.calculateWordPro(label, feature)        
        self.getPrior()                                
        self.getWordPro()                               
        self.sumValue = sum(self.wordCount.itervalues())     
        self.smooth = math.log((float(1))/self.sumValue)     
        
        print self.wordCount
   
        
    #abstractmethod
    def classify(self, instance): 
        result = []
        for key in self.dicCount.keys():
            tempScore = self.priorScore[key]
            for f in instance.features().items():
                if (key, f) in self.attri:
                    tempScore = tempScore +math.log(self.wordPro[key, f]) 
                else:
                    tempScore = tempScore + self.smooth             #Laplace
            result.append((tempScore, key))
        return max(result)[1]
                    
                
    
