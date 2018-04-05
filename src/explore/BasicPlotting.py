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
    
    train = pd.read_csv('../../data/train.csv')
 
    #Create slice that only has the category columns
    train_df_category_columns_only=train.iloc[:,2:]
    
    records_per_category=train_df_category_columns_only.sum(axis=0)
    sum_of_all_category_values_per_record=train_df_category_columns_only.sum(axis=1)
    train['clean']=(sum_of_all_category_values_per_record==0)
    #mapping to 1 or 0 for ease in summation and keeping consistency with other categories
    train['clean'] = train['clean'].map({True:1,False:0})
    
    #count number of clean entries
    number_of_clean_records=train['clean'].sum()

    records_per_category=records_per_category.append(pd.Series({'clean':number_of_clean_records}))
 
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
    
    #plt.show()
    plt.savefig('../../results/figures/Records_Per_Category.png')
    
    #plot
    sum_of_all_category_values_per_record=sum_of_all_category_values_per_record.value_counts()
    plt.figure(figsize=(8,4))
    ax = sns.barplot(sum_of_all_category_values_per_record.index, sum_of_all_category_values_per_record.values, alpha=0.8,color=color[2])
    plt.title("Count of Multiple Categories")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('# of tags ', fontsize=12)
    
    #adding the text labels
    rects = ax.patches
    labels = sum_of_all_category_values_per_record.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.savefig('../../results/figures/Count_of_Multiple_Categories.png')
    #plt.show()