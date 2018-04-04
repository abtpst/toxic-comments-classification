'''
Created on Apr 4, 2018

@author: abhijit.tomar
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    
    color = sns.color_palette()
    sns.set_style("dark")
    
    train = pd.read_csv('../../data/cleaned/train.csv')
    test = pd.read_csv('../../data/cleaned/test.csv')
    
    #Create slice that only has the category columns
    train_df_category_columns_only=train.iloc[:,2:]
    
    records_per_category=train_df_category_columns_only.sum(axis=0)
    #plot
    plt.figure(figsize=(8,4))
    ax= sns.barplot(records_per_category.index, records_per_category.values, alpha=0.8)
    plt.title("Records Per Category")
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Categories ', fontsize=12)
    #adding the text labels
    rects = ax.patches
    labels = records_per_category.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    plt.show(block=True)
    plt.savefig('../../results/figures/Records_Per_Category.jpg')