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
    # --> P( class | word )
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

    


#remove words that co-occur in each domain

'''
for domain in freq_dict:
    all_words = freq_dict[domain]
    for sub_domain in freq_dict:
        for word in freq_dict[sub_domain]:
            for root_word in all_words:
                if word == root_word:
                    del freq_dict[domain][word]
'''


















