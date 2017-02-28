from pandas import to_datetime
import numpy as np
import re

def engineer(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    return df

def num_keyword(df):
    # n_num_keyword: check if a key word makes a difference in terms of interest_level:
    match_list=[map(lambda x: re.search('elevator|cats|dogs|doorman|dishwasher|no fee|laundry|fitness',x.lower()),
                         list(df['features'])[i]) for i in np.arange(0,len(df['features']),1)]
    nfeat_list =[] 
    for i in match_list:
        if i==None:
            nfeat_list.append(0)
        else:
            if any(i)== False: # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1=[]
                map(lambda x: lis1.append(1) if x!= None else lis1.append(0),i)            
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score 
    nfeat_score=[]
    for i in nfeat_list:
        if i<=5:
            nfeat_score.append(0)
        elif i==6:
            nfeat_score.append(1)
        elif i==7:
            nfeat_score.append(2)
        elif i==8:
            nfeat_score.append(3)
        elif i==9:
            nfeat_score.append(4)
        elif i==10:
            nfeat_score.append(5)
        else:
            nfeat_score.append(6)

    df['n_num_keyfeat_score']= nfeat_score
    return df

def no_photo(df):
    df['n_no_photo'] = [1 if i == 0 else 0 for i in map(len,df['photos']
    return df                                                    
