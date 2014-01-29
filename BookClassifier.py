#coding=utf-8

import operator, string
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
import nltk, os, logging, json, ConfigParser, codecs
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.metrics import classification_report
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
import nltk;
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

class BookClassifier:

    def __init__(self):
    
        self.config = ConfigParser.ConfigParser()
        self.config.read("BookClassifier.config")

        cur_dir = os.getcwd()
        rel_dir_path = self.config.get('GLOBAL', 'data_dir')

        self.data_dir = os.path.join(cur_dir, rel_dir_path)
        self.train_file_name = self.config.get('GLOBAL', 'train_file_name')
        self.train_file = os.path.join(self.data_dir, self.train_file_name)
        self.second_train_file_name = self.config.get('GLOBAL', 'second_train_file_name')
        self.second_train_file = os.path.join(self.data_dir, self.second_train_file_name)        

        self.logger_file = os.path.join("OUTPUT", "BookClassifier.log") 
        self.output_file = os.path.join("OUTPUT", "output.txt")
        self.stopwords_set = set(stopwords.words('english'))    


    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")

    def run_main(self):
        self.preprocessing()
        self.feature_extraction()       
        self.classification()
        self.testing()
        self.cross_validation() 

    def feature_extraction(self):
        self.all_feature_selection()

    def clean_book_title(self, title):
        return nltk.word_tokenize(title.translate(None, string.punctuation))

    def clean_book_author(self, author):
        return author.split(";")

    def all_feature_selection(self):
        selected_feat_list = []
        category_dict = {}
        self.selected_feats = []
        self.test_instances_list = []

        for instance in self.instance_list:
            if instance and instance.strip():
                temp = instance.strip().split("\t")
                if temp and len(temp) == 4:
                    key = temp[0]
                    new_temp = []
                    new_temp.extend(self.clean_book_title(temp[2]))
                    new_temp.extend(self.clean_book_author(temp[3]))
                    new_temp.extend(self.bookid_features_dict.get(temp[1], []))
                    toc_set = set(self.bookid_features_dict.get(temp[1]))
                    temp_list = []
                    selected_temp = []
                    for feat in new_temp:
                        if feat and feat in self.best and feat:
                            temp_list.append((feat, True))
                            if feat in toc_set:
                                selected_temp.append(feat)        
                    #temp_list.extend(self.get_bigram(selected_temp))
                    self.selected_feats.append((dict(temp_list), key))            
                if temp and len(temp) == 3:
                    self.test_instances_list.append(instance)                  

    def get_bigram(self, features_list):
        score = BigramAssocMeasures.chi_sq
        n = 10
        all_bigrams = BigramCollocationFinder.from_words(features_list)
        best_bigrams = all_bigrams.nbest(score, n)
        selected_bigrams = [(bigram, True) for bigram in best_bigrams]
        return selected_bigrams
        
    def classification(self):
        #Training NB classifier
        self.nb_classifier = NaiveBayesClassifier.train(self.selected_feats)         
        
        #Training SVM classifier
        self.svm_classifier = SklearnClassifier(LinearSVC()) 
        self.svm_classifier.train(self.selected_feats)
        

    def testing(self):
        
        self.output_file_fd  = open(self.output_file, 'w')
        for instance in self.test_instances_list:
            temp = instance.strip().split("\t")
            if temp and len(temp) == 3:
                new_temp = []
                temp_list = []
                new_temp.extend(self.clean_book_title(temp[1]))
                new_temp.extend(self.clean_book_author(temp[2]))
                for feat in new_temp:
                    if feat and feat not in self.stopwords_set:
                        temp_list.append((feat,True)) 
            temp.insert(0, self.nb_classifier.classify(dict(temp_list)))
            
            self.output_file_fd.write("%s\n" % "\t".join(temp))


    def cross_validation(self):

        train_feats_count = int(len(self.selected_feats))
        fold_size = int(train_feats_count / 10)
        acc_list = []
        svm_acc_list = []
        bnb_acc_list = []
        dtc_acc_list = []
        rf_acc_list = []

        for a in range(10):
            start_index = a * fold_size
            end_index = start_index + fold_size

            train_features = self.selected_feats[:start_index] + self.selected_feats[end_index:]
            test_features  = self.selected_feats[start_index:end_index] 
            
            self.nb_classifier = NaiveBayesClassifier.train(train_features)         
    
            acc = nltk.classify.util.accuracy(self.nb_classifier, test_features) 
            print "\n ACCURACY - NAIVE BAYE CLASSIFIER: %s \n" % acc
            acc_list.append(acc)
       
            self.svm_classifier = SklearnClassifier(LinearSVC()) 
            self.svm_classifier.train(train_features)
            svm_acc = nltk.classify.util.accuracy(self.svm_classifier, test_features) 
            print "\n ACCURACY - SVM  CLASSIFIER: %s \n" % svm_acc
            
            svm_acc_list.append(svm_acc)

            #self.bnb_classifier = SklearnClassifier(BernoulliNB()).train(train_features)
            #bnb_acc = nltk.classify.util.accuracy(self.bnb_classifier, test_features) 
            #bnb_acc_list.append(bnb_acc)
            #print "\n ACCURACY - BNB  CLASSIFIER: %s \n" % bnb_acc
            

            #self.dtc = DecisionTreeClassifier.train(train_features)
            #dtc_acc = nltk.classify.util.accuracy(self.dtc, test_features)
            #print "\n ACCURACY - DTC  CLASSIFIER: %s \n" % dtc_acc
            #dtc_acc_list.append(dtc_acc)        
    

            self.rf = RandomForestClassifier.train(train_features)            
            rf_acc = nltk.classify.util.accuracy(self.rf, test_features)
            print "\n ACCURACY - DTC  CLASSIFIER: %s \n" % rf_acc
            rf_acc_list.append(dtc_acc)        


        print 'Average acc %s' % (float(sum(acc_list)/len(acc_list)))
        print 'Average svm acc %s' % (float(sum(svm_acc_list)/len(svm_acc_list)))
        print 'Bnb acc %s' % (float(sum(bnb_acc_list)/len(bnb_acc_list)))
        print 'Dtc acc %s' % (float(sum(dtc_acc_list)/len(dtc_acc_list)))
        print 'Rf acc %s' % (float(sum(rf_acc_list)/len(rf_acc_list)))

    def preprocessing(self):
        self.initialize_logger()
        self.open_files()
        self.load_data()
        self.close_files()        
        self.compute_word_scores()


    def compute_word_scores(self):
      
        self.second_train_data()
        self.train_data()

    
    def train_data(self):
 
        freq_dist_obj = FreqDist()
        cond_freq_dist_obj = ConditionalFreqDist()
        key_set = set() 
        key_count_dict = {}

        for instance in self.instance_list:
            temp = instance and instance.strip().split("\t") 
            if not temp:continue;  
            if not len(temp) == 4:continue;
            key_set.add(temp[0])
            key  = temp[0]
            features = self.clean_book_title(temp[2])
            features.extend(self.bookid_features_dict.get(temp[1], []))
            for feat in features:
                freq_dist_obj.inc(feat)
                cond_freq_dist_obj[key].inc(feat)
            
        for key in key_set:
            key_count_dict[key] = cond_freq_dist_obj[key].N()
        total_word_count  = sum(key_count_dict.values())

        word_score_dict = {}
        for word, freq in freq_dist_obj.iteritems():
            score = 0
            for key in key_set:
                 score += BigramAssocMeasures.chi_sq(cond_freq_dist_obj[key][word], (freq, key_count_dict.get(key,0)), total_word_count)
            word_score_dict[word] = score
        
        self.best =  sorted(word_score_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

        #total_select_count = int(len(self.best) * 0.85)
        total_select_count = 2000
        self.best = self.best[:total_select_count]
        self.best = [pair[0] for pair in self.best]


    def clean_book_toc(self, toc):
        return [word  for word in re.sub("[^a-zA-Z]"," ", toc).split(" ") if word]

    def second_train_data(self):
        
        freq_dist_obj = FreqDist()
        cond_freq_dist_obj = ConditionalFreqDist()
        self.bookid_features_dict = {}              

        for instance in self.s_instance_list:
            temp = instance and instance.strip().replace("↵","")
            if not temp:continue
            key = temp.split("\t")[0]
            temp = self.clean_book_toc(temp)
            self.bookid_features_dict.setdefault(key, []).extend(temp[1:])
            
    def open_files(self):
        self.train_file_fd = open(self.train_file, 'r') 
        self.second_train_file_fd = open(self.second_train_file, 'r')

    def load_data(self):
        self.load_train_data()
        self.load_second_train_data()


    def load_train_data(self):
        self.instance_list = []
        for instance in self.train_file_fd.readlines():
            self.instance_list.append(instance) 
        self.instance_list = self.instance_list[1:]

    def load_second_train_data(self):
        self.s_instance_list = []
        for instance in self.second_train_file_fd.readlines():
            self.s_instance_list.append(instance)
        self.s_instance_list = self.s_instance_list[1:]

    def close_files(self):
        self.train_file_fd.close() 

if __name__ == "__main__":
    bc = BookClassifier()
    bc.run_main()
