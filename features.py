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


def basic_numeric_features(df):
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df[
        "description"].apply(lambda x: len(x.split(" ")))
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    return df


def num_keyword(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('elevator|cats|dogs|doorman|dishwasher|no fee|laundry|fitness', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i <= 5:
            nfeat_score.append(0)
        elif i == 6:
            nfeat_score.append(1)
        elif i == 7:
            nfeat_score.append(2)
        elif i == 8:
            nfeat_score.append(3)
        elif i == 9:
            nfeat_score.append(4)
        elif i == 10:
            nfeat_score.append(5)
        else:
            nfeat_score.append(6)

    df['n_num_keyfeat_score'] = nfeat_score
    return df


def no_photo(df):
    df['n_no_photo'] = [1 if i == 0 else 0 for i in map(len, df['photos'])]
    return df


def count_caps(df):
    def get_caps(message):
        caps = sum(1 for c in message if c.isupper())
        total_characters = sum(1 for c in message if c.isalpha())
        if total_characters > 0:
            caps = caps / (total_characters * 1.0)
        return caps
    df['amount_of_caps'] = df['description'].apply(get_caps)
    return df


def has_phone(df):
    # http://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
    phone_regex = "(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
    has_phone = df['description'].str.extract(phone_regex)
    df['has_phone'] = [type(item) == unicode for item in has_phone]
    return df


def n_log_price(df):
    # n_price_sqrt improves original 'price' variable smoothing extreme
    # right skew and fat tails.
    # Use either 'price' or this new var to avoid multicolinearity.
    df['n_log_price'] = np.log(df['price'])
    return df


def n_expensive(df):
    # 'Low' interest make 70% population. Statistical analysis shows price
    # among 'Low' interest exhibits the highest kurtosis and skew.
    # n_expensive is 1 when the price is above 75% percentile aggregate
    # prices and 0 otherwise.
    # you can use it along with either price or n_price_sqrt.
    threshold_75p = df[['price']].describe().loc['75%', 'price']
    df['n_expensive'] = [
        1 if i > threshold_75p else 0 for i in list(df['price'])]
    return df


def dist_from_midtown(df):
    from geopy.distance import vincenty
    # pip install geopy
    # https://github.com/geopy/geopy
    # calculates vincenty dist
    # https://en.wikipedia.org/wiki/Vincenty's_formulae
    lat = df['latitude'].tolist()
    long_ = df['longitude'].tolist()
    midtown_lat = 40.7586
    midtown_long = -73.9838
    distance = []
    for i in range(len(lat)):
        distance.append(
            vincenty((lat[i], long_[i]), (midtown_lat, midtown_long)).meters)
    df['distance_from_midtown'] = distance
    return df


def nearest_neighbors(df, n):
    # Input: df and num of meighbors
    # Output: df with price_vs_median for each row
    df_sub = df[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms']]
    rows = range(df.shape[0])
    diffs = map(lambda row: compare_price_vs_median(df_sub, n, row), rows)
    df['price_vs_median_' + str(n)] = diffs
    return df


def compare_price_vs_median(df, n, i):
    from geopy.distance import vincenty
    # Help function For nearest_neighbors
    # Requires geopy.distance
    # for each lat long
    # calculate dist from all other lat longs with same beds and bathrooms
    # find n nearest neighbors
    # calculate median price of n nearest neighbors
    # compare price vs median
    row = df.iloc[i, :]
    lat = row['latitude']
    lon = row['longitude']
    bed = row['bedrooms']
    bath = row['bathrooms']
    price = row['price']
    df.index = range(df.shape[0])
    all_other_data = df.drop(df.index[[i]])
    with_same_bed_bath = all_other_data[all_other_data['bedrooms'] == bed]
    with_same_bed_bath = with_same_bed_bath[
        with_same_bed_bath['bathrooms'] == bath]
    longs = with_same_bed_bath['longitude'].tolist()
    lats = with_same_bed_bath['latitude'].tolist()
    distances = []
    for j in range(len(lats)):
        distance = vincenty((lats[j], longs[j]), (lat, lon)).meters
        distances.append(distance)
    # http://stackoverflow.com/questions/13070461/get-index-of-the-top-n-values-of-a
    dist_positions = sorted(range(len(distances)),
                            key=lambda k: distances[k], reverse=True)[-n:]
    top_dist_df = with_same_bed_bath.iloc[dist_positions, :]
    med_price = with_same_bed_bath['price'].median()
    diff = price / med_price
    return diff

def price_vs_mean_30(df):
    # userfriendly for def_nearest_neighbour created earlier.
    # Output: df with price_vs_median for each row
    # The code below solves NA issues and round some results to save execution errors
    temp = pd.read_json("price_vs_median30.json")['price_vs_median_30']
    mean = np.mean(temp) 
    import math
    df['price_vs_median_30'] = [mean if math.isnan(i)== True  else round(i,2) for i in temp]
    return df

def price_vs_mean_72(df):
    # userfriendly for def_nearest_neighbour created earlier.
    # Output: df with price_vs_median for each row
    # The code below solves NA issues and round some results to save execution errors
    temp = pd.read_json("price_vs_median72.json")['price_vs_median_72']
    mean = np.mean(temp) 
    import math
    df['price_vs_median_72'] = [mean if math.isnan(i)== True  else round(i,2) for i in temp]
    return df

def nearest_neighbors_with_date(df, n,days):
    from datetime import datetime
    # Input: df and num of meighbors
    # Output: df with price_vs_median for each row
    df_sub = df[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms','created']]
    df_sub['date']=df_sub['created'].apply(lambda d: datetime.strptime(d.split(" ")[0], "%Y-%m-%d"))
    rows = range(df.shape[0])
    diffs = map(lambda row: compare_price_vs_median_with_date(df_sub, n, row,days), rows)
    df['price_vs_median_' + str(n)] = diffs
    return df



def compare_price_vs_median_with_date(df, n, i,days):
    from geopy.distance import vincenty
    from datetime import datetime,timedelta
    # Help function For nearest_neighbors
    # Requires geopy.distance
    # for each lat long
    # filter for only places with same beds and bathroom
    # filter for places that were posted within z days
    # calculate dist from all other lat longs with same beds and bathrooms
    # find n nearest neighbors
    # calculate median price of n nearest neighbors
    # compare price vs median
    row = df.iloc[i, :]
    lat = row['latitude']
    lon = row['longitude']
    bed = row['bedrooms']
    bath = row['bathrooms']
    price = row['price']
    date = row['date']
    date_after_n_days = date + timedelta(days=days)
    date_before_n_days = date + timedelta(days=-days)
    df.index = range(df.shape[0])
    all_other_data = df.drop(df.index[[i]])
    with_same_bed_bath = all_other_data[all_other_data['bedrooms'] == bed]
    with_same_bed_bath = with_same_bed_bath[with_same_bed_bath['bathrooms'] == bath]
    with_same_bed_bath = with_same_bed_bath[(with_same_bed_bath['date'] > date_before_n_days) & (with_same_bed_bath['date'] < date_after_n_days)]
    longs = with_same_bed_bath['longitude'].tolist()
    lats = with_same_bed_bath['latitude'].tolist()
    distances = []
    for j in range(len(lats)):
        distance = vincenty((lats[j], longs[j]), (lat, lon)).meters
        distances.append(distance)
    # http://stackoverflow.com/questions/13070461/get-index-of-the-top-n-values-of-a
    dist_positions = sorted(range(len(distances)),
                            key=lambda k: distances[k], reverse=True)[-n:]
    top_dist_df = with_same_bed_bath.iloc[dist_positions, :]
    med_price = with_same_bed_bath['price'].median()
    diff = price / med_price
    return diff


def manager_skill(df):
    new_var = 'manager_id'#'manager_id_encoded'
    #response var
    resp_var = 'interest_level'
    temp = pd.concat([df[new_var], pd.get_dummies(df[resp_var])], axis = 1).groupby(new_var).mean()
    temp.columns = ['high_frac','low_frac', 'medium_frac']
    temp['count'] = df.groupby(new_var).count().iloc[:,1]
    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
    manager_skill=[]
    for i in df['manager_id']:
        for j in temp.index:
            if i==j:
                manager_skill.append(temp['manager_skill'][j])
    df['manager_skill']=manager_skill


#Indepth Features

def allow_pet(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('cats|dogs', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['allow_pets'] = nfeat_score
    return df


def laundry(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('laundry', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['laundry'] = nfeat_score
    return df

def preWar(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('prewar|pre-war', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['preWar'] = nfeat_score
    return df

def furnished(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('furnished', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['furnished'] = nfeat_score
    return df

def dishwash(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('dishwasher', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['dishwash'] = nfeat_score
    return df

def hardwood(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('hardwood', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['hardwood'] = nfeat_score
    return df

def fitness(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('fitness|swimming', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['fitness'] = nfeat_score
    return df

def doorman(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('doorman', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['doorman'] = nfeat_score
    return df

def no_fee(df):
    # n_num_keyword: check if a key word makes a difference in terms of
    # interest_level:
    match_list = [map(lambda x: re.search('no fee', x.lower()),
                      list(df['features'])[i]) for i in np.arange(0, len(df['features']), 1)]
    nfeat_list = []
    for i in match_list:
        if i is None:
            nfeat_list.append(0)
        else:
            if not any(i):  # check to filter out lists with no all None values
                nfeat_list.append(0)
            else:
                lis1 = []
                map(lambda x: lis1.append(1) if x is None else lis1.append(0), i)
                nfeat_list.append(sum(lis1))

    # new variable n_num_keyfeat_score
    nfeat_score = []
    for i in nfeat_list:
        if i >0:
            nfeat_score.append(1)
        else:
            nfeat_score.append(0)

    df['no_fee_'] = nfeat_score
    return df

def 'dist_to_nearest_college'(df):
    Baruch = (40.7402, -73.9834)
    Columbia = (40.8075, -73.9626)
    Cooper_Union = (40.7299, -73.9903)
    FIT = (40.7475, -73.9951)
    Hunter_College = (40.7685, -73.9657)
    John_Jay = (40.7704, -73.9885)
    Julliard = (40.7738, -73.9828)
    NYU = (40.7295, -73.9965)
    NYU_Tandon = (40.6942, -73.9866)
    Pace_University=(40.7111, -74.0049)
    Pratt_University = (40.6913, -73.9625)
    The_New_School = (40.7355199, -73.99715879999997)
    Weill_Cornell = (40.7650, -73.9548) 

    schools = [Baruch,Columbia,Cooper_Union,FIT,Hunter_College,John_Jay, Julliard, NYU, NYU_Tandon,
              Pace_University, Pratt_University, The_New_School, Weill_Cornell]

    from geopy.distance import vincenty
    import numpy as np
    distance = []
    for i in range(0,len(df['latitude']),1):
        lat_long = (list(df['latitude'])[i],list(df['longitude'])[i])
        temp=[]
        for j in schools:
            temp.append(
            vincenty(lat_long, j).meters)
        distance.append(min(temp))
    df['dist_to_nearest_college']= distance
    return df
