'''
Created on Apr 5, 2018

@author: abhijit.tomar
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 

class FeaturePlotter(object):
    '''
    Class for plotting features
    '''

    direct=None
    save_path=None
    
    def __init__(self, params):
        '''
        Constructor
        '''
        self.save_path='../../results/figures/'
        if 'direct' not in params:
            print('Must provide DirectFeatures obeject in params. Use \'direct\' as key')
            raise StopIteration
        self.direct=params['direct']
    
    def generate_plots_for_direct_features(self):
        
        color = sns.color_palette()
        direct_df = self.direct.get_direct_features()
        train = self.direct.get_train()
       
        train_feats=direct_df.iloc[0:len(train),]
        
        test_feats=direct_df.iloc[len(train):,]
        #join the tags
        train_tags=train.iloc[:,2:]
        train_feats=pd.concat([train_feats,train_tags],axis=1)
        
        train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
        plt.figure(figsize=(12,6))

        plt.subplot(121)
        plt.suptitle("Toxicity With Respect To Number Of Words",fontsize=20)
        sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of sentences', fontsize=12)
        plt.title("Number of sentences in each comment", fontsize=15)
        # words
        train_feats['count_word'].loc[train_feats['count_word']>200] = 200
        plt.subplot(122)
        sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of words', fontsize=12)
        plt.title("Number of words in each comment", fontsize=15)
        plt.savefig(self.save_path+'Toxicity_With_Respect_To_Number_Of_Words')
        plt.show()
        
        train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
        #prep for split violin plots
        #For the desired plots , the data must be in long format
        temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
        #spammers - comments with less than 40% unique words
        spammers=train_feats[train_feats['word_unique_percent']<30]
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
        #sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)
        ax=sns.kdeplot(train_feats[train_feats.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')
        ax=sns.kdeplot(train_feats[train_feats.clean == 1].word_unique_percent, label="Clean")
        plt.legend()
        plt.ylabel('Number of occurances', fontsize=12)
        plt.xlabel('Percent unique words', fontsize=12)
        
        x=spammers.iloc[:,-7:].sum()
        plt.subplot2grid((2,2),(1,0),colspan=2)
        plt.title("Count of comments with low(<30%) unique words",fontsize=15)
        ax=sns.barplot(x=x.index, y=x.values,color=color[3])
        
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