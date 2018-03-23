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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
import tensorflow as tf
from math import floor
import numpy as np
import operator
import sklearn
import random
import math
import time

style.use('fivethirtyeight')

class Bayesian_Classifier():
    def __init__(self):
        self.average_sparcity = [0, 0]
        self.doc_space_init = False

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

    def supervised_training(self):
        acc_y = []; acc_x = [];
        acc_y = self.y[4*len(train_y)//5:]
        acc_x = self.x[4*len(train_x)//5:]

        input_x = []; input_y = [];
        input_y = self.y[:4*len(train_y)//5]
        input_x = self.x[:4*len(train_x)//5]

        ''' Feature basis length will be at least argmax(len(doc) for each doc)
        1 8632
        2 9073
        3 4049
        4 2555
        5 1449
        6 4250
        7 2135
        8 2893
        '''

        document_space = {}#13750 for two documents
        for domain in self.freq_dict:
            for word in self.freq_dict[domain]:
                if word[0] not in document_space:
                    document_space[word[0]] = len(document_space)

        self.document_space = document_space
        print("Length of document space:",len(document_space))

        #transform documents into vector space
        self.doc_matrix = [[] for i in range(len(self.freq_dict))]
        for domain in self.freq_dict:
            doc_vect = np.zeros(len(document_space))
            
            for word_freq in self.freq_dict[domain]:
                doc_vect[document_space[word_freq[0]]] = word_freq[1]

            assert(domain > 0)#domain must start at one for subtraction
            self.doc_matrix[domain - 1] = doc_vect

        #transform training data into document space
        for i in range(len(input_x)):
            sample = np.zeros(len(document_space))
            for word in input_x[i]:
                try:
                    sample[document_space[i]] += 1
                except KeyError as e:pass

            input_x[i] = sample

        self.input_x = input_x
        
        for i in range(len(acc_x)):
            sample = np.zeros(len(document_space))
            for word in acc_x[i]:
                try:
                    sample[document_space[i]] += 1
                except KeyError as e:pass
            acc_x[i] = sample

        #transform y's into vector space
        for i in range(len(input_y)):
            assert(domain > 0)#domain must start at one for subtraction
            vect = np.zeros(len(self.freq_dict))
            vect[input_y[i] - 1] += 1
            input_y[i] = vect

        for i in range(len(acc_y)):
            vect = np.zeros(len(self.freq_dict))
            vect[acc_y[i] - 1] += 1
            acc_y[i] = vect

        self.neural_network(self.doc_matrix, input_x, input_y)

    def make_doc_matrix(self):
        if not self.doc_space_init:
            _ = self.transform_to_doc_space('intializer')
        
        #transform documents into vector space
        self.doc_matrix = [[] for i in range(len(self.freq_dict))]
        for domain in self.freq_dict:
            doc_vect = np.zeros(len(self.document_space))
            
            for word_freq in self.freq_dict[domain]:
                doc_vect[self.document_space[word_freq[0]]] = word_freq[1]

            assert(domain > 0)#domain must start at one for subtraction
            self.doc_matrix[domain - 1] = doc_vect
        '''
        color_map = {0:'purple', 1:'blue', 2:'lightgreen', 3:'purple',
                     4:'cyan',   5:'red',  6:'magenta',    7:'k'}

        fig = plt.figure()
        
        ax1 = plt.subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((3,1), (1,0), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid((3,1), (2,0), rowspan=1, colspan=1)

        ax1.plot([i for i in range(len(bayes.doc_matrix[0]))], bayes.doc_matrix[0])
        ax2.plot([i for i in range(len(bayes.doc_matrix[1]))], bayes.doc_matrix[1])
        ax3.plot([i for i in range(len(bayes.doc_matrix[2]))], bayes.doc_matrix[2])

        plt.show()
        '''
        #for j in range(len(bayes.doc_matrix)):
        #    plt.plot([i for i in range(len(bayes.doc_matrix[j]))],
        #             bayes.doc_matrix[j], color=color_map[j])
        #    plt.show()
        
        

    def transform_to_doc_space(self, text_data):
        if not self.doc_space_init:
            self.doc_space_init = True
            document_space = {}#13750 for two documents
            all_words = []
            for domain in self.freq_dict:
                for word in self.freq_dict[domain]:
                    if word[0] not in all_words:
                        all_words.append(word[0])
                        #document_space[word[0]] = len(document_space)

            random.shuffle(all_words)
            for word in set(all_words):
                document_space[word] = len(document_space)    

            self.document_space = document_space
            
        sample = np.zeros(len(self.document_space))
        for word in text_data:
            try:
                sample[self.document_space[word]] += 1
            except KeyError as e:pass

        return sample

    def convert_to_feature_space(self, domain, text_data):
        sample_space = np.zeros(len(self.freq_dict[domain]))
        
        feature_space = {}#[word[0] for word in self.freq_dict[domain]]
        for i, word in enumerate(self.freq_dict[domain]):
            feature_space[word[0]] = i
                
        num_irrelevent = 0
        
        for word in text_data:
            try:
                sample_space[feature_space[word]] += 1
            except KeyError as e:
                num_irrelevent += 1

                        
        self.average_sparcity[0] += 100*num_irrelevent / len(text_data)
        self.average_sparcity[1] += 1
        
        return sample_space#/(10*sum(sample_space)**2+1))

    def neural_network(self, initial_matrix, inputs, labels,
                       num_epochs=10, batch_size=1, learning_rate=1):
        W0 = np.array(initial_matrix)

        #add batching to improve convergance
        for i in range(num_epochs):
            for batch in range(len(inputs)):
                ##### CHECK this dot operation #####
                l1 = self.sigmoid(np.dot(W0, np.array(inputs[batch])))

                #### ADD more layers and regularization
                l1_error = labels[batch] - l1
                l1_delta = l1_error * self.sigmoid(l1, deriv=True)

                #apply update (MeanSqrErr) over average of inputs in a batch
                W0 += np.dot(l1.T, l1_delta) * learning_rate

                print("Cost:",str(np.mean(np.absolute(l1_error))))

        self.W0 = W0    
        
        

    def classify(self, input_data):
        # y == domain, yi = classifier[domain]
        #argmax( P(doc | y) * P(y))
        #argmax( P(x1, x2, ...| y) * P(y))
        # --> P(x1,x2,...|y) == P(x1|y)*P(x2|y)* ...
        #Thus, for all y's in y:
        # --> yi = argmax(P(yi) * (P(x1|yi)*P(x2|yi)* ...))
        #(in this case, P(yi) is the same for all y's in yi)

        _max = [0, -1]# [argmax(P(yi)), yi]
        all_products = []
        for domain in self.freq_dict:
            sample = self.convert_to_feature_space(domain, input_data)
            features = [freq[1] for freq in self.freq_dict[domain]]            

            product = 1
            summ = 1
            for i in range(len(features)):
                #product = np.dot(sample, features) #VERY SLOW
                #if sample[i] != 0:
                #    product *= features[i]# + .01#bias to
                summ += (features[i] * sample[i])
            #print('product: {}, summation: {}'.format(product, summ/sum(sample)))

            product = summ/(100*sum(sample)+1)
            if product > _max[1]:
                _max = [domain, product]

            all_products.append(product)
        '''
        z_score = ((_max[1] - np.mean(all_products)) / (np.std(all_products)))
        accuracy_of_prediction = 1 - np.exp(-(z_score ** 2) / 2)#~N(0, 1)
        '''
        accuracy_of_prediction = 0
        decision = "Fail to reject"
        if accuracy_of_prediction <= 0.95:
            decision = "reject"

        return decision, _max[0]
        

    def fit(self, x, y):
        self.x = x
        self.y = y

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

            #####
            ''' INCLUDE N-GRAMS for all words '''
            #####
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
            spread = 30 #~98% of all words
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
        #  word_prob = sigmoid(word_freq / num_of_domains)
        for domain in classifier:
            total = 0
            #add the freq of each word to the total
            for pair in classifier[domain]:
                total += pair[1]

            total = np.log(total)
            #divide frequencies through the total to get probabilies
            # --> P( class | word )
            for i, pair in enumerate(classifier[domain]):
                word_freq = np.log(pair[1]) * (len(classifier) / doc_freq[word])
                classifier[domain][i] = [pair[0], self.sigmoid(word_freq/total)]

        self.freq_dict = {}
        #freq_dict = {domain:[ [word, P(word | domain)], ...]}
        for domain in classifier:
            #convert to hash lookup (dict) to optimize naive bayes
            self.freq_dict[domain] = classifier[domain]#dict(classifier[domain])



    def sigmoid(self, x, deriv=False):
        if(deriv==True):
            return x * (1 - x)

        return 1/(1 + np.exp(-x))

file = open("trainingdata.txt")

num_docs = int(file.readline()[:-1])

train_y = []
train_x = []
num_classes = 8# <<----
for i in range(num_docs):
    line = file.readline()[:-1]
    if (int(line[0]) <= num_classes):# max 8; 89%@2, 80%@3, 75%@4
        train_y.append(int(line[0]))
        train_x.append(line[2:].split(' '))

test_y = train_y[1*len(train_y)//5:]
test_x = train_x[1*len(train_x)//5:]

train_y = train_y[:4*len(train_y)//5]
train_x = train_x[:4*len(train_x)//5]

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

class NeuralComputer:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.in_dim = len(x[0])
        self.out_dim = len(y[0])
        
    def Perceptron(self, tensor):
        #with tf.name_scope('softmax_linear'):
            
        V0 = tf.Variable(tf.truncated_normal([self.in_dim, 1000]))
        b0 = tf.Variable(tf.truncated_normal([1000]))
        l0 = tf.sigmoid(tf.matmul(tensor, V0) + b0)

        V1 = tf.Variable(tf.truncated_normal([1000, 500]))
        b1 = tf.Variable(tf.truncated_normal([500]))
        l1 = tf.sigmoid(tf.matmul(l0, V1) + b1)

        V2 = tf.Variable(tf.truncated_normal([500, 300]))
        b2 = tf.Variable(tf.truncated_normal([300]))
        l2 = tf.sigmoid(tf.matmul(l1, V2) + b2)
        
        V3 = tf.Variable(tf.truncated_normal([300, 200]))
        b3 = tf.Variable(tf.truncated_normal([200]))
        l3 = tf.sigmoid(tf.matmul(l2, V3) + b3)

        V4 = tf.Variable(tf.truncated_normal([200, 50]))
        b4 = tf.Variable(tf.truncated_normal([50]))
        l4 = tf.sigmoid(tf.matmul(l3, V4) + b4)
        
        weights = tf.Variable( tf.zeros([50, self.out_dim]),name='weights')
        biases = tf.Variable(tf.zeros([self.out_dim]),name='biases')

        logits = tf.nn.softmax(tf.matmul(l4, weights) + biases)
        
        return logits, weights, biases

    def init_placeholders(self, n_classes, batch_size):
        #init Tensors: fed into the model during training
        x = tf.placeholder(tf.float32, shape=(None, self.in_dim))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

        #Neural Network Model
        y, W, b = self.Perceptron(x)

        return y, W, b, x, y_

    def train(self, test_x, in_str, training_epochs=10,learning_rate=.5,display_step=1):
        batch_size = len(test_x)
        test_size = batch_size* floor(len(self.y)/batch_size)

        self.x = self.x[:test_size]
        self.y = self.y[:test_size]
        
        # Train W, b such that they are good predictors of y
        self.out_y, W, b, self.in_x, y_ = self.init_placeholders(self.out_dim, batch_size)

        # Cost function: Mean squared error
        loss = tf.reduce_sum(tf.pow(y_ - self.out_y, 2))/(batch_size)

        # Regularization of loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        loss = loss + reg_constant * sum(reg_losses)
        
        # Gradient descent: minimize cost via Adam Delta Optimizer (SGD)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=.99,epsilon=3e-08).minimize(loss)

        # Initialize variables and tensorflow session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

        start_time = time.time()
        print_time = True
        for i in range(training_epochs):
            j=0
            while j < len(self.x):
                start = j
                end = j + batch_size
                
                self.sess.run([optimizer, loss], feed_dict={self.in_x: self.x[start:end],
                                                            y_: self.y[start:end]})
                j += batch_size
            # Display logs for epoch in display_step
            if (i) % display_step == 0:
                if print_time:
                    print_time = False
                    elapsed_time = time.time() - start_time
                    print('Predicted duration of this session:',(elapsed_time*training_epochs//60) + 1,'minute(s)')
                cc = self.sess.run(loss, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})
                print("Training step: {} || cost= {}".format(i,cc))
                        
        print("\nOptimization Finished!\n")
        training_loss = self.sess.run(loss, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})
        print("Training cost=",training_loss,"\nW=", self.sess.run(W)[:1],"\nb=",self.sess.run(b),'\n')
        correct_prediction = tf.equal(tf.argmax(self.out_y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy for predictions of {}'.format(in_str),
                self.sess.run(accuracy, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})*100,'%')
        
        #str(self.sess.run(accuracy, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})*100//1) + ' %'

    def save(self, in_str):
        self.saver.save(self.sess, in_str)

    def load(self, graph):
        #out_y, W, b, in_x, y_ = self.init_placeholders(self.out_dim, batch_size)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(graph + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
    def predict(self, test_x):
        predictions = []
        for matrix in test_x:
            predictions.append(self.sess.run(self.out_y, feed_dict={self.in_x:matrix}))

        self.sess.close()
        return predictions

    def max_of_predictions(self, predictions):
        out_arr = []
        for pred in predictions:
            #print('\n========')
            _max = [0, 0]# [index, value]
            for matrix in pred:
                for i, vect in enumerate(matrix):
                    if _max[1] <  vect:
                        _max[1] = vect
                        _max[0] = i
                    #print('{}::{}'.format(i,vect))
                #print(':MAX:', _max[0], _max[1])
            out_arr.append(_max[0])

        #indecies of max values in one-hot arrays
        return out_arr
    
features = []
for i in range(num_classes):
    features.append([freq[1] for freq in bayes.freq_dict[i + 1]])

sample_matrix = []
#bayes.make_doc_matrix()
for i, text in enumerate(train_x):
    #sample_space = []
    #for i in range(num_classes):
    #    sample_space.append(np.dot(bayes.convert_to_feature_space(i + 1, text), features[i]))
        
    sample_matrix.append(bayes.transform_to_doc_space(text))

sample_y = []
for i in range(len(train_y)):
    distro = np.zeros(num_classes)
    distro[train_y[i] - 1] += 1
    sample_y.append(distro)

print('\nInitializing Perceptron ...')
nn = NeuralComputer(sample_matrix, sample_y)

nn.train([[bayes.transform_to_doc_space(text)] for text in test_x],
          "Document Classifier", training_epochs=5, learning_rate=.5, display_step=1)

'''
# The Best Way to Visualize a Dataset Easily
# @Siraj Raval -> https://youtu.be/yQsOFWqpjkE
# t distribution stochastic neighbor embedding (t-SNE) visualization

tsne = TSNE(n_components=2, random_state=0)
train_x_2d = tsne.fit_transform(sample_matrix)

#numpy.linalg.svd()

#scatter plot individual classes


markers = ('s','d','o', 'p', '^', 'v', '<', '>')
color_map = {0:'purple', 1:'blue', 2:'lightgreen', 3:'purple',
             4:'cyan', 5:'red', 6:'magenta', 7:'k'}
plt.figure()
for index, freq in enumerate(train_x_2d):
    
    plt.scatter(x=train_x_2d[index][0], y=train_x_2d[index][1],
                c=color_map[train_y[index] - 1],# % number of classes
                marker=markers[train_y[index] - 1],)# % number of classes
                #label=train_y[index])
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()

# Build a classifier for every pair of classes
# Compare all pairs to get best class given the words
'''
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








        
