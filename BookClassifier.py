import operator, string
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
import nltk;

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
        self.output_file = os.path.join("OUTPUT", "output.txt")

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_file, level=logging.INFO)
        logging.info("Initialized logger")

    def run_main(self):
        self.preprocessing()
        self.feature_extraction()       
        self.classification()
        self.cross_validation() 
        self.testing()

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
                    temp_list = []
                    for feat in new_temp:
                        if feat:
                            temp_list.append((feat, True))
                    self.selected_feats.append((dict(temp_list), key))            
                if temp and len(temp) == 3:
                    self.test_instances_list.append(instance)                  

    def classification(self):

        self.nb_classifier = NaiveBayesClassifier.train(self.selected_feats)         
         

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
                    if feat:
                        temp_list.append((feat,True)) 
            temp.insert(0, self.nb_classifier.classify(dict(temp_list)))
            
            self.output_file_fd.write("%s\n" % "\t".join(temp))


    def cross_validation(self):

        train_feats_count = int(len(self.selected_feats))
        fold_size = int(train_feats_count / 10)
        acc_list = []

        for a in range(10):
            start_index = a * fold_size
            end_index = start_index + fold_size

            train_features = self.selected_feats[:start_index] + self.selected_feats[end_index:]
            test_features  = self.selected_feats[start_index:end_index] 
            
            self.nb_classifier = NaiveBayesClassifier.train(train_features)         
    
            acc = nltk.classify.util.accuracy(self.nb_classifier, test_features) 
            print "\n ACCURACY - NAIVE BAYE CLASSIFIER: %s \n" % acc
            #self.nb_classifier.show_most_informative_features()
            acc_list.append(acc)
        print 'Average acc %s' % (float(sum(acc_list)/len(acc_list)))

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
