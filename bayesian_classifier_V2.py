# HackerRank Problem --> https://www.hackerrank.com/challenges/document-classification/problem
# Classify documents based on trainingdata.txt
# % % % % % % %  % % %'''
'''
Using bayesian methods to classify text documents

USE Z-SCORE when num_classes is >= 8

Develop Class to make predictions of class given any text
    For all words in doc FIND:
    --> P(words | doc)
    --> P(doc | words)
    --> P(words) = term freq * inverse class freq
'''
# % % % % % % %  % % %'''
## Developed by Nathan Shepherd

from scipy import stats
import numpy as np
import operator
import random
import math

class Bayesian_Classifier():
    def __init__(self):
        pass

    def evaluate_model(self):
        num_success = 0
        len_test = len(self.x)
        for i in range(len_test):
            _, pred = self.classify(self.x[i])

            if pred == self.y[i]:
                num_success += 1
                
        accuracy = num_success*100 / len_test
        print('\nAccuracy of classifier: {} %'.format(accuracy))

        for domain in self.print_dict:
            print('Words that occur most frequently: domain',domain,"{}")
            arr = []
            for pair in self.print_dict[domain]:
                if pair[1] == self.domain_means[domain]:
                    arr.append([pair[0], pair[1]])

            printed = 0
            random.shuffle(arr)
            while (not printed > 5 and not printed > len(arr)):
                print(arr[printed])
                printed += 1
            

    def test_accuracy(self, novel_x, novel_y):
        if type(novel_x) != type(list()):
            print("Input DataType must be list of strings")

        num_success = 0
        len_test = len(novel_x)
        for i in range(len_test):
            _, pred = self.classify(novel_x[i])

            if pred == novel_y[i]:
                num_success += 1

        accuracy = num_success*100 / len_test
        print('\nAccuracy of classifier: {} %'.format(accuracy))
            

    def classify(self, input_data):
        # y == domain, yi = classifier[domain]
        #argmax( P(doc | y) * P(y))
        #argmax( P(x1, x2, ...| y) * P(y))
        # --> P(x1,x2,...|y) == P(x1|y)*P(x2|y)* ...
        #Thus, for all y's in y:
        # --> yi = argmax(P(yi) * (P(x1|yi)*P(x2|yi)* ...))
        #(in this case, P(yi) is the same for all y's in yi)

        _max = [0, None]# [argmax(P(yi)), yi]
        all_products = []
        for domain in self.freq_dict:
            product = self.prior_dict[domain]
            for word in (input_data):
                try:
                    product += self.freq_dict[domain][word]
                except KeyError as e:
                    product += self.sigmoid(np.log(1 / len(self.y)))/10

            if _max[1] == None:
                _max = [domain, product]
                
            if product < _max[1]:
                _max = [domain, product]

            all_products.append(product)

        #print(all_products)
        '''
        z_score = ((_max[1] - np.mean(all_products)) / (np.std(all_products)))
        accuracy_of_prediction = 1 - np.exp(-(z_score ** 2) / 2)#~N(0, 1)

        decision = "Fail to reject"
        if accuracy_of_prediction <= 0.95:
            decision = "reject"
        '''
        decision =""
        return decision, _max[0]
        

    def fit(self, x, y):
        self.x = x
        self.y = y

        #compute priors
        prior_dict = {}
        for domain in y:
            if domain not in prior_dict:
                prior_dict[domain] = 0
            else:
                prior_dict[domain] += 1

        self.prior_dict = {}
        for domain in prior_dict:
            self.prior_dict[domain] = prior_dict[domain] / len(y)

        #compute the list of all words in each class
        # freq_dict == {class: {word:freq}, class: {...}, ...}
        # Contains set of all unique words and their relative frequency for each class
        freq_dict = {}
        for domain in set(y):
            freq_dict[domain] = []

        for i, domain in enumerate(y):
            for word in x[i]:
                freq_dict[domain].append(word)

        #contains probability of all words given the class
        #-->{class: p(words | class}
        classifier = {}
        self.domain_means = {}
        self.print_dict = {}

        #take the set of each list
        for domain in freq_dict:
            print("\nComputing statistics on Domain:",domain)
            all_words = freq_dict[domain]
            set_all_words = list(set(all_words))
    
            #correlate the set to their relative frequencies
            corr_dict = {}
            for word in set_all_words:
                corr_dict[word] = 0

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
            self.domain_means[domain] = mean
            self.print_dict[domain] = freq_dict[domain]

            #keep words within certain range of the mean word freq
            spread = stddev*2 #~98% of all words
            classifier[domain] = []
            for pair in freq_dict[domain]:
                if pair[1] in range(mean - spread, mean + spread):
                    classifier[domain].append([pair[0], pair[1]])

        [print('Percentage of target words kept:',domain, len(classifier[domain]),
                len(classifier[domain])*100//len(freq_dict[domain]),
                    '%')for domain in classifier]


        #determine inverse document frequency weighting
        doc_freq = {}
        for domain in classifier:
            for word, freq in classifier[domain]:
                if word not in doc_freq:
                    doc_freq[word] = 1
                else:
                    doc_freq[word] += 1
        self.doc_freq =sorted(doc_freq.items(), key=operator.itemgetter(1))[::-1]
        #convert classifier into probabilities of each word given the domain
        #  word_freq = (domain_freq * (num_domains / domain_freq))
        #  word_prob = log(word_freq / num_of_domains)
        for domain in classifier:
            total = 0
            #add the freq of each word to the total
            for pair in classifier[domain]:
                total += pair[1]

            total = np.log(total)
            #divide frequencies through the total to get probabilies
            # --> P( class | word )
            for i, pair in enumerate(classifier[domain]):
                word_freq = np.log(pair[1]) * np.log(len(classifier) / doc_freq[word])
                classifier[domain][i] = [pair[0], self.sigmoid(word_freq/total)]

        self.freq_dict = {}
        #freq_dict = {domain:[ [word, P(word | domain)], ...]}
        for domain in classifier:
            #convert to hash lookup (dict) to optimize naive bayes
            self.freq_dict[domain] = dict(classifier[domain])

        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

file = open("trainingdata.txt")

num_docs = int(file.readline()[:-1])

train_y = []
train_x = []
for i in range(num_docs):
    line = file.readline()[:-1]
    if (int(line[0]) <= 3):# max 8; 89%@2, 80%@3, 75%@4
        train_y.append(int(line[0]))
        train_x.append(line[2:].split(' '))

test_y = train_y[4*len(train_y)//5:]
test_x = train_x[4*len(train_x)//5:]

train_y = train_y[:4*len(train_y)//5]
train_x = train_x[:4*len(train_x)//5]

# Build a classifier for every pair of classes
# Compare all pairs to get best class given the words
'''
classes = [1, 2, 3, 4]

argmax (
    bayes(1, 2); bayes(2, 3); bayes(3, 4);
    bayes(1, 3); bayes(2, 4);
    bayes(1, 4); 
    )
'''
'''
tree_classifier = {}
Y = list(set(train_y))
for i in range(len(Y)):
    tree_classifier[i] = []
    for j in range(i, len(Y) - 1):
        temp_x = []
        temp_y = []
        for index, val in enumerate(train_y):
            if val in [i, j + 1]:
                temp_x.append(train_x[index])
                temp_y.append(train_y[index])

        bayes = Bayesian_Classifier()
        bayes.fit(temp_x, temp_y)
        tree_classifier[i].append(bayes)
        print(i, j + 1)

num_success = 0
for index, test in enumerate(test_x):
    mode = []
    for i in tree_classifier:
        for bayes in tree_classifier[i]:
            mode.append(bayes.classify(test))

    mode = stats.mode(mode)[0]
    #print(mode, test_y[index])
    if mode == test_y[index]:
        num_success += 1

print("Accuracy of model by tree:",(num_success*100)/len(test_x))
'''

bayes = Bayesian_Classifier()
bayes.fit(train_x, train_y)
#bayes.evaluate_model()
bayes.test_accuracy(test_x, test_y)
'''
num_decisions = len(test_x)
num_incorrect_decisions = 0
for i, ins in enumerate(test_x[:num_decisions]):
    #print(bayes.classify(ins), train_y[i])
    dec, pred = bayes.classify(ins)

    #H0: correct classification
    if dec == 'Fail to reject':#H0 is true
        if pred != test_y[i]:
            num_incorrect_decisions += 1
            
    if dec == 'reject':#h0 is false
        if pred == test_y[i]:
            num_incorrect_decisions += 1
            
print('Percentage of incorrect decisions:', 100*num_incorrect_decisions/num_decisions)
'''









        
