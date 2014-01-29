import operator
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk, os, logging, json, ConfigParser, codecs
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.metrics import classification_report
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords

class BookClassifier:

    def __init__(self):
    
        self.config = ConfigParser.ConfigParser()
        self.config.read("BookClassifier.config")

        cur_dir = os.getcwd()
        rel_dir_path = self.config.get('GLOBAL', 'data_dir')

        self.data_dir = os.path.join(cur_dir, rel_dir_path)
        self.train_file_name = self.config.get('GLOBAL', 'train_file_name')
        self.train_file = os.path.join(self.data_dir, self.train_file_name)

        self.logger_file = os.path.join("OUTPUT", "BookClassifier.log") 

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")

    def run_main(self):
        self.preprocessing()
        self.feature_extraction()       
        self.classification()
        self.cross_validation() 

    def feature_extraction(self):
        self.all_feature_selection()

    def all_feature_selection(self):
        selected_feat_list = []
        category_dict = {}
        self.selected_feats = []

        for instance in self.instance_list:
            if instance and instance.strip():
                temp = instance.strip().split("\t")
                if temp and len(temp) == 4:
                    key = temp[0]
                    temp = temp[1:]
                    temp_list = []
                    for feat in temp:
                        if feat:
                            temp_list.append((feat, True))
                    self.selected_feats.append((dict(temp_list), key))            
        

    def classification(self):

        train_feats_count = int(len(self.selected_feats) * 0.75 )

        train_features = self.selected_feats[:train_feats_count]
        test_features = self.selected_feats[train_feats_count:]

        self.nb_classifier = NaiveBayesClassifier.train(train_features)         
 
        print '\n ACCURACY - NAIVE BAYE CLASSIFIER: %s \n' % (nltk.classify.util.accuracy(self.nb_classifier, test_features))
        self.nb_classifier.show_most_informative_features()


    def cross_validation(self):

        train_feats_count = int(len(self.selected_feats))
        fold_size = int(train_feats_count / 10)

        for a in range(10):
            start_index = a * fold_size
            end_index = start_index + fold_size

            train_features = self.selected_feats[:start_index] + self.selected_feats[end_index:]
            test_features  = self.selected_feats[start_index:end_index] 
            
            self.nb_classifier = NaiveBayesClassifier.train(train_features)         
     
            print '\n ACCURACY - NAIVE BAYE CLASSIFIER: %s \n' % (nltk.classify.util.accuracy(self.nb_classifier, test_features))
            #self.nb_classifier.show_most_informative_features()
            


    def preprocessing(self):
        self.initialize_logger()
        self.open_files()
        self.load_data()
        self.close_files()        

    def open_files(self):
        self.train_file_fd = open(self.train_file, 'r') 

    def load_data(self):
        self.instance_list = []
        for instance in self.train_file_fd.readlines():
            self.instance_list.append(instance) 
        self.instance_list = self.instance_list[1:]

    def close_files(self):
        self.train_file_fd.close() 

if __name__ == "__main__":
    bc = BookClassifier()
    bc.run_main()
