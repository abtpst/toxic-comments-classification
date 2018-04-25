'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

class Plotter(object):
    '''
    Class for plotting features
    '''
    
    features_object=None
    save_path=None
    color = sns.color_palette()
    
    def __init__(self,params):
        self.save_path='../../results/figures/'
        if 'features' not in params:
            print('Must provide a feature_engineering.FeaturesFromContent.Features object in params. Use \'features\' as key')
            raise StopIteration
        self.features_object=params['features']
        
class DirectFeaturesPlotter(Plotter):
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(Plotter, self).__init__(params)
        if type(self.features_object) != 'DirectFeatures':
            print('Must provide a DirectFeatures object in params. Use \'features\' as key')
            raise StopIteration
        
    def generate_plots(self):
        
        direct_df = self.features_object.get_features_data_frame()
        direct_df=direct_df.loc[direct_df['processed'] == True]
        train = self.features_object.get_train()
       
        train_feats=direct_df.iloc[0:len(train),]

        #join the tags
        train_tags=train.iloc[:,1:]
        train_tags=train_tags[train_tags['id'].isin(train_feats['id'].tolist())]
        train_feats=pd.concat([train_feats,train_tags],join='inner',axis=1)
        train_feats=train_feats.dropna()
        train_feats['number_of_sentences'].loc[train_feats['number_of_sentences']>10] = 10 
        
        plot_data = train_feats[['number_of_sentences','clean']]
        plt.figure(figsize=(12,6))

        plt.subplot(121)
        plt.suptitle("Toxicity With Respect To Number Of Words",fontsize=20)
        sns.violinplot(y='number_of_sentences',x='clean', data=plot_data,split=True)
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of sentences', fontsize=12)
        plt.title("Number of sentences in each comment", fontsize=15)
        # words
        train_feats['number_of_words'].loc[train_feats['number_of_words']>200] = 200
        
        plot_data=train_feats[['number_of_words','clean']]
        plt.subplot(122)
        sns.violinplot(y='number_of_words',x='clean', data=plot_data,split=True,inner="quart")
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of words', fontsize=12)
        plt.title("Number of words in each comment", fontsize=15)
        plt.savefig(self.save_path+'Toxicity_With_Respect_To_Number_Of_Words')
        plt.show()
        
        train_feats['number_of_unique_words'].loc[train_feats['number_of_unique_words']>200] = 200
        plot_data=train_feats[['number_of_words','number_of_unique_words','clean','percentage_of_unique_words']]
        #prep for split violin plots
        #For the desired plots , the data must be in long format
        temp_df = pd.melt(train_feats, value_vars=['number_of_words', 'number_of_unique_words'], id_vars='clean')
        
        #spammers - comments with less than 40% unique words
        spammers=plot_data[train_feats['percentage_of_unique_words']<30]
        
        plt.figure(figsize=(16,12))
        plt.suptitle("What's so unique ?",fontsize=20)
        gridspec.GridSpec(2,2)
        plt.subplot2grid((2,2),(0,0))
        sns.violinplot(x='variable', y='value', hue='clean', data=temp_df,split=True,inner='quartile')
        plt.title("Absolute wordcount and unique words count")
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        plt.subplot2grid((2,2),(0,1))
        plt.title("Percentage of unique words of total words in comment")
        #sns.boxplot(x='clean', y='percentage_of_unique_words', data=train_feats)
        ax=sns.kdeplot(plot_data[plot_data.clean == 0].percentage_of_unique_words, label="Bad",shade=True,color='r')
        ax=sns.kdeplot(plot_data[plot_data.clean == 1].percentage_of_unique_words, label="Clean")
        plt.legend()
        plt.ylabel('Number of occurrences', fontsize=12)
        plt.xlabel('Percent unique words', fontsize=12)
        
        x=spammers.iloc[:,-7:].sum()
        plt.subplot2grid((2,2),(1,0),colspan=2)
        plt.title("Count of comments with low(<30%) unique words",fontsize=15)
        ax=sns.barplot(x=x.index, y=x.values,color=self.color[3])
        
        #adding the text labels
        rects = ax.patches
        labels = x.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
        
        plt.xlabel('Threat class', fontsize=12)
        plt.ylabel('# of comments', fontsize=12)
        plt.savefig(self.save_path+'What\'s_so_unique_?')
        plt.show()

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DerivedFeaturesPlotter(Plotter):
    
    def __init__(self, params):
        '''
        Constructor
        '''
        super(Plotter, self).__init__(params)
        if type(self.features_object) != 'DerivedFeatures':
            print('Must provide a DerivedFeatures object in params. Use \'features\' as key')
            raise StopIteration
    
    def generate_plots(self):
        
        clean_text=self.features_object.get_corpus().sanitized_comment
        
        for key in self.features_object.get_ngram_ranges():
            print('Plotting for ',key)
            with open(self.features_object.get_feature_path()+key+'Vectorizer.pkl','rb') as vect:
                tf_idf_vectorizer=pickle.load(vect)
            with open(self.features_object.get_feature_path()+key+'Features.pkl','rb') as feature_in:
                features=pickle.load(feature_in)
                
            train_unigrams =  tf_idf_vectorizer.transform(clean_text.iloc[:self.features_object.get_train().shape[0]])
            test_unigrams = tf_idf_vectorizer.transform(clean_text.iloc[self.features_object.get_test().shape[0]:])
            
            tfidf_top_n_per_class=self.top_feats_by_class(train_unigrams,features)
            
            self.plot_and_save(tfidf_top_n_per_class,key)

    def plot_and_save(self,tfidf_top_n_per_class,moniker,num_bars=9):
        plt.figure(figsize=(16,22))
        plt.suptitle("TF_IDF Top "+moniker+" per Class",fontsize=20)
        gridspec.GridSpec(4,2)
        plt.subplot2grid((4,2),(0,0))
        sns.barplot(tfidf_top_n_per_class[0].feature.iloc[0:num_bars],tfidf_top_n_per_class[0].tfidf.iloc[0:num_bars],color=self.color[0])
        plt.title("class : Toxic",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        plt.subplot2grid((4,2),(0,1))
        sns.barplot(tfidf_top_n_per_class[1].feature.iloc[0:num_bars],tfidf_top_n_per_class[1].tfidf.iloc[0:num_bars],color=self.color[1])
        plt.title("class : Severe toxic",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        
        plt.subplot2grid((4,2),(1,0))
        sns.barplot(tfidf_top_n_per_class[2].feature.iloc[0:num_bars],tfidf_top_n_per_class[2].tfidf.iloc[0:num_bars],color=self.color[2])
        plt.title("class : Obscene",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        
        plt.subplot2grid((4,2),(1,1))
        sns.barplot(tfidf_top_n_per_class[3].feature.iloc[0:num_bars],tfidf_top_n_per_class[3].tfidf.iloc[0:num_bars],color=self.color[3])
        plt.title("class : Threat",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        
        plt.subplot2grid((4,2),(2,0))
        sns.barplot(tfidf_top_n_per_class[4].feature.iloc[0:num_bars],tfidf_top_n_per_class[4].tfidf.iloc[0:num_bars],color=self.color[4])
        plt.title("class : Insult",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        
        plt.subplot2grid((4,2),(2,1))
        sns.barplot(tfidf_top_n_per_class[5].feature.iloc[0:num_bars],tfidf_top_n_per_class[5].tfidf.iloc[0:num_bars],color=self.color[5])
        plt.title("class : Identity hate",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        
        plt.subplot2grid((4,2),(3,0),colspan=2)
        sns.barplot(tfidf_top_n_per_class[6].feature.iloc[0:num_bars],tfidf_top_n_per_class[6].tfidf.iloc[0:num_bars])
        plt.title("class : Clean",fontsize=15)
        plt.xlabel('Word', fontsize=12)
        plt.ylabel('TF-IDF score', fontsize=12)
        
        plt.show()
        
    def top_tfidf_feats(self,row, features, top_n=25):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        if self.features is None:
            print('Could not load features')
            return
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(self.features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df

    def top_feats_in_doc(self,Xtr, features, row_id, top_n=25):
        ''' Top tfidf features in specific document (matrix row) '''
        row = np.squeeze(Xtr[row_id].toarray())
        return self.top_tfidf_feats(row, features, top_n)
    
    def top_mean_feats(self,Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
        ''' Return the top n features that on average are most important amongst documents in rows
            indentified by indices in grp_ids. '''
        
        D = Xtr[grp_ids].toarray()
    
        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)
        return self.top_tfidf_feats(tfidf_means, features, top_n)
    
    '''
    Return a list of dfs, where each df holds top_n features and their mean tfidf value
            calculated across documents with the same class label. 
    '''
    # modified for multilabel milticlass
    def top_feats_by_class(self,Xtr, features, min_tfidf=0.1, top_n=20):
       
        dfs = []
        cols=self._train.columns
        for col in cols:
            ids = self._train.index[self._train[col]==1]
            feats_df = self.top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
            #feats_df.label = label
            dfs.append(feats_df)
        return dfs