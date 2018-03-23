# HackerRank Problem --> https://www.hackerrank.com/challenges/document-classification/problem
# Classify documents based on trainingdata.txt
## Developed by Nathan Shepherd

import numpy as np
import operator
import random

file = open("trainingdata.txt")

num_docs = int(file.readline()[:-1])

train_y = []
train_x = []
for i in range(num_docs):
    line = file.readline()[:-1]
    if (int(line[0]) <= 8):# max 8; 89%@2, 80%@3, 75%@4
        train_y.append(int(line[0]))
        train_x.append(line[2:].split(' '))

test_y = train_y[2*len(train_y)//3:]
test_x = train_x[2*len(train_x)//3:]

train_y = train_y[:2*len(train_y)//3]
train_x = train_x[:2*len(train_x)//3]
                 
freq_dict = {}
# freq_dict == {class: {word:freq, word:freq:, [..., ...], ...}, class: {...}, ...}
# Contains set of all unique words and their relative frequency for each class
for domain in set(train_y):
    freq_dict[domain] = []

#compute the list of all words in each class
for i, domain in enumerate(train_y):
    for word in train_x[i]:
        freq_dict[domain].append(word)

#contains (class: synonymn} pairs
classifier = {}

#take the set of each list
for domain in freq_dict:
    print("\nComputing statistics on Domain:",domain)
    all_words = freq_dict[domain]
    set_all_words = list(set(all_words))
    
    #correlate the set to their relative frequencies
    corr_dict = {}
    for root in set_all_words:
        corr_dict[root] = 0

    all_words = sorted(all_words)
    for word in all_words:
        corr_dict[word] += 1
        
    #replace ordered words with summary of unique words relative to each word's frequency
    freq_dict[domain] = sorted(corr_dict.items(), key=operator.itemgetter(1))[::-1]

    #print statistics
    mean = int(np.mean(list(corr_dict.values())))
    stddev = int(np.std(list(corr_dict.values()), axis=0))
    print('Mean:',np.mean(list(corr_dict.values()),axis=0),
          'Stdev:',np.std(list(corr_dict.values()), axis=0),
          'Count:',len(set_all_words))
    
    #print('Words that occur most frequently:',[(pair[0], pair[1]) for pair in freq_dict[domain] if pair[1] == mean])

    spread = 30
    classifier[domain] = []
    for pair in freq_dict[domain]:
        if pair[1] in range(mean - spread, mean + spread):
            classifier[domain].append([pair[0], pair[1]])

[print(domain, len(classifier[domain]),
       len(classifier[domain])*100//len(freq_dict[domain]),
           '%')for domain in classifier]

#convert classifier into probabilities of each word given the domain
for domain in classifier:
    total = 0
    #add the freq of each word to the total
    for pair in classifier[domain]:
        total += pair[1]

    #divide frequencies through the total to get probabilies
    for i, pair in enumerate(classifier[domain]):
        classifier[domain][i] = [pair[0], pair[1]/total]


comparitor = {}
for domain in classifier:
    #convert to hash lookup (dict) to optimize naive bayes
    comparitor[domain] = dict(classifier[domain])

def naive_bayes(doc, y):# y == domain, yi = classifier[domain]
    #argmax( P(doc | y) * P(y))
    #argmax( P(x1, x2, ...| y) * P(y))
    # --> P(x1,x2,...|y) == P(x1|y)*P(x2|y)* ...
    #Thus, y = argmax(P(yi) * (P(x1|yi)*P(x2|yi)* ...))
    #(in this case, P(yi) is the same for all y's in yi)

    _max = [0, -1]# [argmax(P(yi)), yi]
    for domain in comparitor:
        product = 1
        for word in doc:
            try:
                product *= comparitor[domain][word]
            except KeyError as e:
                pass
        if product > _max[1]:
            _max = [domain, product]

    return _max[0], y

num_success = 0
len_test = len(test_x)
for i in range(len(test_x)):
    pair = naive_bayes(test_x[i], test_y[i])
    if pair[0] == pair[1]:
        num_success += 1

accuracy = num_success*100 / len_test
print('\nAccuracy of classifier: {} %'.format(accuracy))


'''
Darwinian Natural Selection
--> Variation: There is a variety of traits and or a means of mutation
--> Selection: Threshold of fitness for specific traits
--> Heredity: Children recieve parent's genetic information

Genetic Algorithm
1.) Create a random population of N genetic objects
     ex.) Predict "cat" from (tar, aat, ase, etc.)

2.) Calculate Fitness for all N elements

3.) Selection of M genetic objects
     ex.) Assign probability of selecting each object relative to Fitness
     ex.) Probability of selection for cat >> tar == .33, aas == .66, ase == .01

4.) Reproduction via some means
     ex.) tar + aas >> t|ar + a|as >> (probablilistic mutation) >> tas (del ase)
'''

import numpy as np
import operator
import random
import math

domain_words = [[word for word in comparitor[domain]] for domain in comparitor]
ALL_WORDS = []
for group in domain_words:
    for word in group:
        ALL_WORDS.append(word)
ALL_WORDS = list(set(ALL_WORDS))

class DNA():
    def __init__(self, test_x, test_y, target_fitness=.80):
        # testing data used to calculate fitness
        self.test_x = test_x
        self.test_y = test_y
        
        # desired accuracy after training
        self.target_fitness = target_fitness

        #weight to each word in ALL_WORDS
        self.weights = {}
        sample = random.randint(0, 1) + 0.5
        for word in ALL_WORDS:
            self.weights[word] = sample
        
        self.reject = False
        self.calc_fitness()
        self.mating_index = None

    def genetic_bayes(self, doc, y):# y == domain, yi = classifier[domain]
        #argmax( P(doc | y) * P(y))
        #argmax( P(x1, x2, ...| y) * P(y))
        # --> P(x1,x2,...|y) == P(x1|y)*P(x2|y)* ...
        #Thus, y = argmax(P(yi) * (P(x1|yi)*P(x2|yi)* ...))
        #(in this case, P(yi) is the same for all y's in yi)

        _max = [0, -1]# [argmax(P(yi)), yi]
        for domain in comparitor:
            product = 1
            for word in doc:
                try:
                    product *= self.weights[word] * comparitor[domain][word]
                except KeyError as e:
                    pass
            if product > _max[1]:
                _max = [domain, product]

        return _max[0], y

    def most_informative_features(self, length=5):
        features = ['Null' for i in range(length)]
        sorted_features = sorted(self.weights.items(), key=operator.itemgetter(1))[::]
        
        for i in range(length):
            features[i] = [sorted_features[i][0], sorted_features[i][1]]

        return features
        

    def calc_fitness(self):
        self.fitness = 0
        num_success = 0
        for i in range(len(self.test_x)):
            pair = self.genetic_bayes(self.test_x[i], self.test_y[i])
            if pair[0] == pair[1]:
                num_success += 1

        #accuracy
        self.fitness = num_success*100 / len(self.test_x)
        #self.fitness = np.exp(self.fitness)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    ############################################
    def reproduce(self, population):
        # use population.mating_index
        #combine DNA in a meaningful way
        child_dna = self.weights
        keep_prob = self.sigmoid(self.fitness)

        if self.fitness > population[self.mating_index].fitness:
            for word in self.weights:
                random_sample = random.randint(0, 1)
                
                if random_sample < keep_prob:# keep self words
                    child_dna[word] = self.weights[word] + .1
                    
                else:#subtract random_sample to emulate mutation
                    child_dna[word] = population[self.mating_index].weights[word]
        else:
            for word in self.weights:
                random_sample = random.randint(0, 1)
                
                if random_sample > keep_prob:# keep mates words
                    child_dna[word] = population[self.mating_index].weights[word] + .1
                    #add random_sample to emulate mutation
                else:
                    child_dna[word] = self.weights[word]
                
        self.weights = child_dna

class Population():
    def __init__(self, test_x, test_y, size=50, target_fitness=80):
        self.target_fitness = target_fitness
        self.population = []
        for i in range(size):
            random_variable = DNA(test_x, test_y, target_fitness)
            self.population.append(random_variable)

    def fittest_of(self, all_members):
        maximal = 0
        for dna in all_members:
            if dna.fitness > maximal:
                maximal = dna.fitness
        return maximal

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def selection(self):
        for i, dna in enumerate(self.population):
            #parent_index == i
            if random.randint(0,1) < self.sigmoid(np.exp(dna.fitness)/np.exp(self.fittest_of(self.population))) or dna.fitness > self.target_fitness:
                dna.reject = True
            else:
                dna.reject = False
                dna.mating_index = math.floor(random.randint(0, len(self.population)-1))

        for dna in self.population:
            if not dna.reject:
                dna.reproduce(self.population)

    def main(self):
        generation = 0
        while self.fittest_of(self.population) < self.target_fitness:
            self.selection()
            generation += 1
            fittest = self.fittest_of(self.population)
            mi_features = []
            printed = False
            for dna in self.population:
                if dna.fitness == fittest and not printed:
                    mi_features = dna.most_informative_features(5)
                    self.best_weights = dna.weights
                    printed = True
            acc = fittest
            print('\nGeneration: {} || Accuracy: {}'.format(generation, acc))
            print('Most Informative Features: {}'.format([str(mi[0]) + ' : ' + str(mi[1]) for mi in mi_features]))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

            if fittest == self.target_fitness:
                print("\nEvolved to target phenotype")

    def get_weights(self):
        return self.best_weights
                

print('Genetic Classifier:')
pop = Population(test_x, test_y, size=30)
pop.main()
    



















