from collections import defaultdict
import math
from classifier import Classifier

class maxent(Classifier):
    def __init__(self, model = {}):
        super(maxent, self).__init__(model)
        self.label_feature_pair = defaultdict(int)         
        self.all_instances = []                          
        self.labels = []
        self.max_instance = 0
    
    def get_model(self):
        return self.seen
    def set_model(self, model):
        self.seen = model
    model = property(get_model, set_model)
        
    def initial(self, instances):
        for instance in instances:
            label = instance.label
            if label not in self.labels:
                self.labels.append(label)
            for feature in instance.features():
                self.label_feature_pair[(label, feature)] += 1
            label_features = []
            label_features.append(label)
            label_features += instance.features()
            self.all_instances.append(label_features)
        self.size = len(self.all_instances)
        self.current_lambda = [0.0] * len(self.label_feature_pair)
        for instance in self.all_instances:
            if len(instance) - 1 > self.max_instance:
                self.max_instance = len(instance) - 1
        self.real_expectation = [0.0] * len(self.label_feature_pair)
        for i, f in enumerate(self.label_feature_pair):
            self.real_expectation[i] = float(self.label_feature_pair[f])/self.size
            self.label_feature_pair[f] = i
        
          
    def cal_numerator(self, features, label):
        numerator = 0
        for f in features:
            if(label, f) in self.label_feature_pair:
                numerator += self.current_lambda[self.label_feature_pair[(label, f)]]
        return math.exp(numerator)
    
    def cal_posterior(self, features):
        numerators_label = []
        for label in self.labels:
            numerator = self.cal_numerator(features, label)
            numerators_label.append((numerator, label))
        denominator = 0;
        for numerator, label in numerators_label:
            denominator += numerator;
        posterior_label = []
        for numerator,label in numerators_label:
            posterior_label.append((numerator/denominator, label))
        return posterior_label
    
    def cal_expectation(self):
        expectation = [0.0] * len(self.label_feature_pair)
        for instance in self.all_instances:
            features = instance[1:]
            probability = self.cal_posterior(features)
            for feature in features :
                for posterior,label in probability:
                    if(label, feature) in self.label_feature_pair:
                        index = self.label_feature_pair[(label, feature)]
                        expectation[index] += posterior * (1.0/self.size)
        return expectation
    
    def is_convergence(self, last_lambda, current_lambda):
        for i in range(len(last_lambda)):
            if abs (last_lambda[i] - current_lambda[i])>=0.0005:
                return False
        return True
    
    #calculate the nth lambda and (n+1)th lambda  
    def cal_step(self, index):
        step_size = 1.0/self.max_instance * math.log(self.real_expectation[index]/self.expectation[index])
        return step_size
    
    def GIS(self):
        while 1 :
            last_lambda = self.current_lambda[:]
            self.expectation = self.cal_expectation()
            for index in range(len(self.current_lambda)):
                step_size = self.cal_step(index)
                self.current_lambda[index] += step_size
            if self.is_convergence(last_lambda, self.current_lambda):
                break
    
    def train(self, instances):
        self.initial(instances)
        self.GIS()
    
    def classify(self, instance):
        probability = self.cal_posterior(instance.features())
        probability.sort(reverse=True)
        return probability[0][1]
    
