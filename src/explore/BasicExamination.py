'''
Created on Apr 4, 2018

@author| abhijit.tomar
'''
import pandas as pd
if __name__ == '__main__':
    
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    
    print(train.head(5))
    
    train_rows=train.shape[0]
    test_rows=test.shape[0]
    total=train_rows+test_rows
    print("          | train  | test")
    print("Records   |",train_rows,"|",test_rows)
    print("Fraction  |",round(train_rows*100/total),"    |",round(test_rows*100/total))
    
    #Create slice that only has the category columns
    train_df_category_columns_only=train.iloc[:,2:]
    
    '''
    By sum(axis=0), we will count records where a non zero value appears
    for any category
    '''
    records_per_category=train_df_category_columns_only.sum(axis=0)
    
    '''
    If sum_of_all_category_values_per_record is 0 for any record, then 
    that record can be considered clean
    '''
    sum_of_all_category_values_per_record=train_df_category_columns_only.sum(axis=1)
    train['clean']=(sum_of_all_category_values_per_record==0)
    #mapping to 1 or 0 for ease in summation and keeping consistency with other categories
    train['clean'] = train['clean'].map({True:1,False:0})
    
    #count number of clean entries
    number_of_clean_records=train['clean'].sum()
    
    print("Number of comments = ",len(train))
    print("Number of clean comments = ",number_of_clean_records)
    print("Number of unclean comments =",records_per_category.sum())
    
    print("Check for missing values in Train dataset")
    null_check=train.isnull().sum()
    print(null_check)
    print("Check for missing values in Test dataset")
    null_check=test.isnull().sum()
    print(null_check)
    print("filling NA with \"UNK\"")
    train["comment_text"].fillna("UNK", inplace=True)
    test["comment_text"].fillna("UNK", inplace=True)
    
    train.to_csv('../../data/cleaned/train.csv')
    test.to_csv('../../data/cleaned/test.csv')