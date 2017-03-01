from pandas import to_datetime
import numpy as np
import re


def scrub(df):
    """
    function designed to perform data cleaning on the training data set
    """
    df['building_id'].replace(to_replace='0', value=np.nan, inplace=True)
    df["created"] = to_datetime(df["created"])
    return df

def engineer(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
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
    df['n_no_photo'] = [1 if i == 0 else 0 for i in map(len,df['photos'])]
    return df  

def count_caps(df):
    def get_caps(message):
        caps =sum(1 for c in message if c.isupper())
        total_characters =sum(1 for c in message if c.isalpha())
        if total_characters>0:
            caps = caps/(total_characters* 1.0)
        return caps
    df['amount_of_caps']=df['description'].apply(get_caps)
    return df

def has_phone(df):
    phone_regex = "(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})" # http://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
    has_phone = t_df['description'].str.extract(phone_regex)
    df['has_phone']=[type(item)==unicode for item in has_phone]
    return df

def n_price_sqrt(df):
    # n_price_sqrt improves original 'price' variable smoothing extreme right skew and fat tails. 
    # Use either 'price' or this new var to avoid multicolinearity.
    df['n_price_sqrt'] =  df['price']**(0.5)
    return df

def n_expensive(df):
    # 'Low' interest make 70% population. Statistical analysis shows price among 'Low' interest exhibits the highest kurtosis and skew. 
    # n_expensive is 1 when the price is above 75% percentile aggregate prices and 0 otherwise.
    # you can use it along with either price or n_price_sqrt.
    threshold_75p = df[['price']].describe().loc['75%','price']
    df['n_expensive']=[1 if i > threshold_75p else 0 for i in list(df['price'])]
    return df

