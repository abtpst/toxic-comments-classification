'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import numpy.core.multiarray as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import progressbar
progressbar.streams.flush()
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image

class Clouder(object):
    
    train = None
    test = None
    mystops=None
    categories=None
    nasties=None
    nasty_path=None
    
    def __init__(self,params):
        
        self.train = pd.read_csv(params['train'])
        self.test = pd.read_csv(params['test'])
        if 'stops' in params:
            self.mystops = params['stops']
        else:
            self.mystops=set(STOPWORDS)
        if 'categories' in params:
            self.categories=params['categories']
        else:
            self.categories=['toxic','severe_toxic','obscene','threat','insult','identity_hate','clean']
        self.nasties=set()
        self.nasty_path='../../data/nasties.pkl'
    
    def generate_word_clouds(self):
        
        bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])
        nasty_list=[]
        clean_set=set()
        for index in bar(range(len(self.categories))):
            cat=self.categories[index]
            current_nasty=set()
            clean_mask=np.array(Image.open('../../data/images/mask.jpg'))
            #clean_mask=clean_mask[:,:,1]
            #wordcloud for clean comments
            subset=self.train.loc[self.train[cat]==1]
            text=subset.comment_text.values
            for sentence in text:
                for word in sentence.split():
                    if cat!='clean':
                        current_nasty.add(word.lower())
                    else:
                        clean_set.add(word.lower())
                if len(current_nasty)>0:
                    nasty_list.append(current_nasty)
                
            wc= WordCloud(background_color="white",max_words=2000,mask=clean_mask,stopwords=self.mystops)
            wc.generate(" ".join(text))
            plt.figure(figsize=(20,10))
            plt.axis("off")
            plt.title('Most Frequent Words in '+cat+' Comments', fontsize=20)
            plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
            plt.savefig('../../results/figures/'+cat+'_wordcloud.jpg')
            #plt.show()
        
        self.nasties=set.intersection(*nasty_list).difference(clean_set)
        with open(self.nasty_path,'wb') as nasty_out:
            pickle.dump(self.nasties,nasty_out,pickle.HIGHEST_PROTOCOL)
    
    def get_nasties(self):
        return self.nasties