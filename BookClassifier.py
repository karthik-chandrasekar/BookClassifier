#!/usr/bin/env python
#coding=utf-8

import operator, string, re, sys
import nltk, os, logging, ConfigParser
import nltk.classify.util
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

class BookClassifier:

    def __init__(self):
    
        config = ConfigParser.ConfigParser()
        config.read("BookClassifier.config")

        cur_dir = os.getcwd()
        
        #Config parameters
        data_dir = config.get('GLOBAL', 'data_dir')
        op_dir = config.get('GLOBAL', 'output_dir')
        train_file = config.get('GLOBAL', 'train_file_name')
        train_file_2 = config.get('GLOBAL', 'train_file_name_2')
        self.bigram_threshold = int(config.get('GLOBAL', 'bigram_threshold'))
        self.k_fold = int(config.get('GLOBAL', 'k_fold'))
        self.unigram_threshold = int(config.get('GLOBAL', 'unigram_threshold'))

        self.data_dir = os.path.join(cur_dir, data_dir)
        self.output_dir = os.path.join(cur_dir, op_dir)
        self.train_file = os.path.join(self.data_dir, train_file)
        self.train_file_2 = os.path.join(self.data_dir, train_file_2)        
        self.logger_file = os.path.join(self.output_dir, "BookClassifier.log") 
        self.mode = int(sys.argv[1])        
        
        if self.mode == 1:
            output_file = config.get('GLOBAL', 'output_file_1') 
        elif self.mode  ==2:
            output_file = config.get('GLOBAL', 'output_file_2') 
        self.output_file = os.path.join(self.output_dir, output_file)
        
        #Data structures 
        self.stopwords_set = set(stopwords.words('english'))    
        self.toc_list = []
        self.training_feats = []
        self.test_cases = []
        self.book_instances = []
        self.selected_features = []
        self.book_category_set = set()
        self.bookid_to_toc_dict = {}   #toc - table of contents
       
        self.train_file_fd = None 
        self.train_file_2_fd = None
        self.output_file_fd = None

        #classifiers
        self.nb_classifier = None
        self.svm_classifier = None

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")
        self.logging = logging

    def run_main(self):
        self.preprocessing()
        self.feature_selection()
        self.feature_extraction()       
        self.classification()
        self.testing()
        self.cross_validation() 
        self.close_files()        

    def clean_book_title(self, title):
        return nltk.word_tokenize(title.translate(None, string.punctuation))

    def clean_author_name(self, author):
        return author.split(";")

    def feature_extraction(self):
        for instance in self.book_instances:
            try:
                raw_data = instance and instance.strip() and instance.strip().split("\t")
                if raw_data and len(raw_data) == 4:
                    bookid = raw_data[0]
                    features = []
                    features.extend(self.clean_book_title(raw_data[2]))
                    features.extend(self.clean_author_name(raw_data[3]))
                    features.extend(self.bookid_to_toc_dict.get(raw_data[1], []))
                    train_feats_list = []
                    for feat in features:
                        if feat and feat.lower() in self.selected_features and feat.lower() not in self.stopwords_set:
                            train_feats_list.append((feat.lower(), True))
                    train_feats_list.extend(self.get_bigram([pair[0] for pair in train_feats_list if pair]))
                elif raw_data and len(raw_data) == 3:
                    self.test_cases.append(instance)
                else:
                    continue
                self.training_feats.append((dict(train_feats_list), bookid))            
            except:
                self.logging.info("Exception while running this instance %s\n" % instance)
                continue

    def get_bigram(self, features_list):
        score = BigramAssocMeasures.chi_sq
        all_bigrams = BigramCollocationFinder.from_words(features_list)
        best_bigrams = all_bigrams.nbest(score, self.bigram_threshold)
        selected_bigrams = [(bigram, True) for bigram in best_bigrams]
        return selected_bigrams
        
    def classification(self):
        #Training NB classifier
        self.nb_classifier = NaiveBayesClassifier.train(self.training_feats)         
        
        #Training SVM classifier
        self.svm_classifier = SklearnClassifier(LinearSVC()) 
        self.svm_classifier.train(self.training_feats)
        
    def testing(self):
        for instance in self.test_cases:
            try:
                raw_data = instance.strip() and instance.strip() and instance.strip().split("\t")
                if raw_data:
                    features = []
                    train_feats_list = []
                    features.extend(self.clean_book_title(raw_data[1]))
                    features.extend(self.clean_author_name(raw_data[2]))
                    for feat in features:
                        if feat and feat.lower() not in self.stopwords_set and feat.lower() in self.selected_features:
                            train_feats_list.append((feat.lower(),True)) 
                    train_feats_list.extend(self.get_bigram([pair[0] for pair in train_feats_list if pair]))
                
                label = self.svm_classifier.classify(dict(train_feats_list))
                self.output_file_fd.write("%s\t%s\n" % (raw_data[0], label))
            except:
                self.logging.info("Exception while running this instance %s\n" % instance)
                

    def cross_validation(self):
        train_feats_count = int(len(self.training_feats))
        fold_size = int(train_feats_count / self.k_fold)
        nb_accuracy_list = []
        svm_accuracy_list = []
        nb_f_val_list = []
        svm_f_val_list = []

        for a in range(self.k_fold):
            start_index = a * fold_size
            end_index = start_index + fold_size

            train_features = self.training_feats[:start_index] + self.training_feats[end_index:]
            test_features  = self.training_feats[start_index:end_index] 
            
            self.nb_classifier = NaiveBayesClassifier.train(train_features)         
            nb_acc = nltk.classify.util.accuracy(self.nb_classifier, test_features) 
            nb_accuracy_list.append(nb_acc)
       
            self.svm_classifier = SklearnClassifier(LinearSVC()) 
            self.svm_classifier.train(train_features)
            svm_acc = nltk.classify.util.accuracy(self.svm_classifier, test_features) 
            svm_accuracy_list.append(svm_acc)

            #Find F-Measure
            nb_f_val = self.compute_measures(test_features, self.nb_classifier, "NB")
            nb_f_val_list.append(nb_f_val)
            svm_f_val = self.compute_measures(test_features, self.svm_classifier, "SVM")
            svm_f_val_list.append(svm_f_val)

        self.logging.info('Average accuracy of Naive Bayes Classifier %s\n' % (float(sum(nb_accuracy_list)/len(nb_accuracy_list))))
        self.logging.info('Average accuracy of SVM Classifier %s\n' % (float(sum(svm_accuracy_list)/len(svm_accuracy_list))))
        self.logging.info('Average F measure of Naive Bayes Classifier %s\n' % (float(sum(nb_f_val_list)/len(nb_f_val_list))))
        self.logging.info('Average F measure of SVM Classifier %s\n' % (float(sum(svm_f_val_list)/len(svm_f_val_list))))

    def compute_measures(self, test_features, classifier, classifier_name):
        actual_labels, predicted_labels = self.get_actual_and_predicted_labels(test_features, classifier)
        precision = self.find_precision(actual_labels, predicted_labels)
        recall = self.find_recall(actual_labels, predicted_labels)
        f_val = self.find_f_measure(precision, recall)
        return f_val

    def find_precision(self, actual_labels, predicted_labels):
        if not actual_labels and not predicted_labels:
            return 0
        precision_list = []
        for category in self.book_category_set:
            if not predicted_labels.get(category):
                continue
            precision = nltk.metrics.precision(actual_labels.get(category, set()), predicted_labels.get(category, set()))
            precision_list.append(precision)
        return float(sum(precision_list)/len(precision_list))

    def find_recall(self, actual_labels, predicted_labels):
        if not actual_labels and not predicted_labels:
            return 0
        recall_list = []
        for category in self.book_category_set:
            if not actual_labels.get(category):
                continue
            recall = nltk.metrics.recall(actual_labels.get(category, set()), predicted_labels.get(category, set()))
            recall_list.append(recall)
        return float(sum(recall_list)/len(recall_list))
         
    def find_f_measure(self, precision, recall):
        if precision == 0 and recall == 0:
            return 0
        f_val = 2 * (precision * recall) / float(precision + recall)
        return f_val

    def get_actual_and_predicted_labels(self, test_features, classifier):
        actual_labels = {}
        predicted_labels = {}
        for i, (features, label) in enumerate(test_features):
            actual_labels.setdefault(label, set()).add(i)
            labels = classifier.classify(features)
            predicted_labels.setdefault(labels, set()).add(i)
        return (actual_labels, predicted_labels)

    def preprocessing(self):
        self.initialize_logger()
        self.open_files()
        self.load_data()

    def feature_selection(self):
        self.clean_and_structure_toc_data()
        self.clean_train_data_and_find_best_features()
    
    def clean_train_data_and_find_best_features(self):
        freq_dist_obj = FreqDist()
        cond_freq_dist_obj = ConditionalFreqDist()
        self.book_category_set = set() 

        for instance in self.book_instances:
            try:
                raw_data = instance and instance.strip() and instance.strip().split("\t") 
                if not raw_data or len(raw_data) != 4 : continue  
                bookid  = raw_data[0]
                self.book_category_set.add(bookid)
                features = []
                features.extend(self.clean_book_title(raw_data[2]))
                features.extend(self.clean_author_name(raw_data[3]))
                features.extend(self.bookid_to_toc_dict.get(raw_data[1], []))
                for feat in features:
                    freq_dist_obj.inc(feat)
                    cond_freq_dist_obj[bookid].inc(feat)
            except:
                self.logging.info("Exception while running this instance %s \n" % instance)
                
        total_word_count = 0    
        for bookid in self.book_category_set:
            total_word_count += cond_freq_dist_obj[bookid].N()

        word_score_dict = {}
        for word, freq in freq_dist_obj.iteritems():
            score = 0
            if word and word.lower() in self.stopwords_set:continue
            for bookid in self.book_category_set:
                score += BigramAssocMeasures.chi_sq(cond_freq_dist_obj[bookid][word], (freq, cond_freq_dist_obj[bookid].N()), total_word_count)
            word_score_dict[word] = score
        
        self.selected_features =  sorted(word_score_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        total_select_count = int(len(self.selected_features) * self.unigram_threshold/float(100))
        self.selected_features = self.selected_features[:total_select_count]
        self.selected_features = set([pair[0].lower() for pair in self.selected_features if pair[0]])

    def clean_book_toc(self, toc):
        return [word  for word in re.sub("[^a-zA-Z]"," ", toc).split(" ") if word]

    def clean_and_structure_toc_data(self):
        for instance in self.toc_list:
            raw_data = instance and instance.strip() and instance.strip().replace("↵","")
            if not raw_data:continue
            bookid = raw_data.split("\t")[0]
            clean_data = self.clean_book_toc(raw_data)
            self.bookid_to_toc_dict.setdefault(bookid, []).extend(clean_data[1:])
            
    def open_files(self):
        self.train_file_fd = open(self.train_file, 'r') 
        self.train_file_2_fd = open(self.train_file_2, 'r')
        self.output_file_fd = open(self.output_file, 'w')

    def load_data(self):
        self.load_train_data()
        if self.mode == 2:   #Load more train data only when it run as problem 2. 
            self.load_more_train_data()

    def load_train_data(self):
        self.book_instances = []
        for instance in self.train_file_fd.readlines():
            self.book_instances.append(instance) 
        self.book_instances = self.book_instances[1:]

    def load_more_train_data(self):
        for instance in self.train_file_2_fd.readlines():
            self.toc_list.append(instance)
        self.toc_list = self.toc_list[1:]

    def close_files(self):
        self.train_file_fd.close() 
        self.train_file_2_fd.close()
        self.output_file_fd.close()

if __name__ == "__main__":
    bc = BookClassifier()
    bc.run_main()
