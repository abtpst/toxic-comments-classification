'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import os
import pickle
import pandas as pd
import progressbar
progressbar.streams.flush()
from processor.TextProcessors import TextAnalyzer

class Features(object):
    
    _train=None
    _test=None
    _save_path=None

    def __init__(self,params):
        
        if 'train' in params and 'test' in params:
            self._train = pd.read_csv(params['train'])
            self._test = pd.read_csv(params['test'])
            
class DirectFeatures(Features):
    '''
    Class for extracting features from text
    '''
  
    _features_data_frame=None
    nasties=set()
    
    def __init__(self, params):
    
        super(DirectFeatures, self).__init__(params)
        
        self._save_path='../../data/cleaned/direct_features.csv'
        if os.path.exists(self._save_path):
            if 'purge' in params:
                if not params['purge']:
                    self._features_data_frame=pd.read_csv(self._save_path)
                else:
                    os.remove(self._save_path)
            else:
                self._features_data_frame=pd.read_csv(self._save_path)
                
        if 'nasty_path' in params:
            with open(params['nasty_path'],'rb') as nasty:
                self.nasties = pickle.load(nasty)
        
        if 'nasties' in params:
            self.nasties=params['nasties']
            
    def generate_features_data_frame(self):  
        
        if self._features_data_frame is None:  
            self._features_data_frame = pd.concat([self._train.iloc[:,1:3],self._test.iloc[:,1:3]])
            self._features_data_frame=self._features_data_frame.reset_index(drop=True)
    
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        ta=TextAnalyzer({})
        
        for index in bar(range(len(self._features_data_frame))):
            row=self._features_data_frame.loc[index]
            comment=row.comment_text
            num_sentences,num_words,num_unique_words,num_letters,num_punctuations,num_uppers,num_titles,num_stops,mean_length,num_nasties,num_emojis = ta.get_metrics(comment,self.nasties)
            self._features_data_frame.loc[index, 'number_of_sentences'] = num_sentences
            self._features_data_frame.loc[index, 'number_of_words'] = num_words
            self._features_data_frame.loc[index, 'number_of_unique_words'] = num_unique_words
            self._features_data_frame.loc[index, 'number_of_letters'] = num_letters
            self._features_data_frame.loc[index, 'number_of_punctuations'] = num_punctuations
            self._features_data_frame.loc[index, 'number_of_uppercase_words'] = num_uppers
            self._features_data_frame.loc[index, 'number_of_title_words'] = num_titles
            self._features_data_frame.loc[index, 'number_of_stop_words'] = num_stops
            self._features_data_frame.loc[index, 'mean_length_of_words'] = mean_length
            self._features_data_frame.loc[index, 'number_of_nasty_words'] = num_nasties
            self._features_data_frame.loc[index, 'number_of_emojis'] = num_emojis
            self._features_data_frame.loc[index, 'percentage_of_unique_words'] = (100*num_unique_words/num_words)
            self._features_data_frame.loc[index, 'percentage_of_punctuations'] = (100*num_punctuations/num_words)
            self._features_data_frame.loc[index, 'percentage_of_nasty_words'] = (100*num_nasties/num_words)
            self._features_data_frame.loc[index, 'percentage_of_emojis'] = (100*num_emojis/num_words)
            self._features_data_frame.loc[index, 'processed'] = True

        self._features_data_frame.to_csv(self._save_path)
    
    def get_features_data_frame(self):
        return self._features_data_frame
    
    def get_train(self):
        return self._train
    
    def get_test(self):
        return self._test

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DerivedFeatures(Features):
    '''
    Class for extracting derived features from text
    '''
   
    corpus=None
    _ngram_ranges={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3),'quadgram':(4,4),'pentagram':(5,5)}
    _merged_data_frame=None
    
    def __init__(self, params):
    
        super(DerivedFeatures, self).__init__(params)
        
        self._corpus_save_path='../../data/cleaned/corpus.csv'
        self.feature_path='../../data/features/'

        if os.path.exists(self._corpus_save_path):
            if 'purge' in params:
                if not params['purge']:
                    self.corpus=pd.read_csv(self._corpus_save_path)
                else:
                    os.remove(self._corpus_save_path)
            else:
                self.corpus=pd.read_csv(self._corpus_save_path)
        
    def generate_corpus(self): 
        if self._merged_data_frame is None:  
            self._merged_data_frame = pd.concat([self._train.iloc[:,1:3],self._test.iloc[:,1:3]])
            self._merged_data_frame=self._merged_data_frame.reset_index(drop=True)
            self.corpus=pd.DataFrame({'id':self._merged_data_frame['id'],'comment_text':self._merged_data_frame['comment_text']})
    
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        ta=TextAnalyzer({})
        for index in bar(range(len(self.corpus))):
            row=self._merged_data_frame.loc[index]
            comment=row.comment_text
            self.corpus.loc[index, 'sanitized_comment'] = ta.get_sanitized(comment)
        
        self.corpus=self.corpus.dropna(axis=0,how='any')
        self.corpus.to_csv(self._corpus_save_path)
    
    def generate_features(self): 
         
        if os.path.exists(self._corpus_save_path):
            self.corpus=pd.read_csv(self._corpus_save_path)
            self.corpus = self.corpus[self.corpus['sanitized_comment'].notnull()]
        else:
            print('Corpus not found')
            return
        
        clean_text=self.corpus.sanitized_comment
        for key, range_ngram in self._ngram_ranges.items():
            print('Generating for ',key)
            
            tf_idf_vectorizer = TfidfVectorizer(min_df=200,  max_features=10000, 
                strip_accents='unicode', analyzer='word',ngram_range=range_ngram,
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
            
            tf_idf_vectorizer.fit(clean_text)
            features = np.array(tf_idf_vectorizer.get_feature_names())
            
            with open(self.feature_path+key+'Features.pkl','wb') as feature_out:
                pickle.dump(features,feature_out,pickle.HIGHEST_PROTOCOL)
            
            with open(self.feature_path+key+'Vectorizer.pkl','wb') as vectorizer_out:
                pickle.dump(tf_idf_vectorizer,vectorizer_out,pickle.HIGHEST_PROTOCOL)
  
    def get_ngram_ranges(self):
        return self.get_ngram_ranges
    
    def get_feature_path(self):
        return self.feature_path
    
    def get_train(self):
        return self._train
    
    def get_test(self):
        return self._test