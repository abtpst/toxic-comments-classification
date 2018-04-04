'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''
import numpy.core.multiarray as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image

if __name__ == '__main__':
    
    train = pd.read_csv('../../data/cleaned/train.csv')
    test = pd.read_csv('../../data/cleaned/test.csv')
    mystops=set(STOPWORDS)
    categories=['toxic','severe_toxic','obscene','threat','insult','identity_hate','clean']
    
    for cat in categories:
        clean_mask=np.array(Image.open('../../data/images/base.png'))
        clean_mask=clean_mask[:,:,1]
        #wordcloud for clean comments
        subset=train.loc[train[cat]==1]
        text=subset.comment_text.values
        wc= WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=mystops)
        wc.generate(" ".join(text))
        plt.figure(figsize=(20,10))
        plt.axis("off")
        plt.title('Most Frequent Words in '+cat+' Comments', fontsize=20)
        plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
        plt.savefig('../../results/figures/'+cat+'_wordcloud.jpg')
        #plt.show()