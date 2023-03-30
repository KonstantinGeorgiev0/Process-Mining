import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, datetime, psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import *
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from hypopt import GridSearch
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



### PREPROCESSING ### 


train = pd.read_csv('BPI_Challenge_2012-training.csv')
test = pd.read_csv('BPI_Challenge_2012-test.csv')

# ### **Functions concerning time**

def day(x):
    """Convert object to the day of year

    Args:
        x (str)

    Returns:
        DateTime object
    """
    return x.day

def day_week(x):
    """Convert object to the day of week

    Args:
        x (str)

    Returns:
        DateTime object
    """
    return x.weekday()

def time_of_day(x):
    """Convert object to the hour of the day

    Args:
        x (str)

    Returns:
        DateTime object
    """
    return x.hour

# ### **Function 'time_conversion**

def time_conversion(dataframe):
    """Transform 'event time:timestamp' and 'case REG_DATE' from str to DateTime in a given Dataframe
        Additionally, this function creates timestamps for the start and finish of a task in a seperate column. 
        The difference between these timestaps is the time to complete a task, which is also added to the dataframe.

        Commented out lines are still for discussion
        
    Args:
        dataframe (pd.DataFrame): A pd.DataFrame in the format from the BPI_challenge 2012

    Returns:
        dataframe_output: A pd.DataFrame with all the strings reformatted to DateTime in the 'event time:timestamp' and 'case REG_DATE' columns
    """
    
#     dataframe.drop(columns = ['eventID '], inplace=True) # Drop eventID
    dataframe.reset_index(inplace=True)
    
    #Transform 'event time:timestamp' and 'case REG_DATE' from str to DateTime  
    dataframe['case REG_DATE'] =  pd.to_datetime(dataframe['case REG_DATE'], format='%d-%m-%Y %H:%M:%S.%f')
    dataframe['event time:timestamp'] =  pd.to_datetime(dataframe['event time:timestamp'], format='%d-%m-%Y %H:%M:%S.%f')
    
    #Creates timestamps for the start and finish of a task in a seperate column + the time to complete the task.
    dataframe['timestamp_start'] = dataframe['case REG_DATE'].values.astype(np.int64) // 10 ** 9
    dataframe['timestamp_finish'] = dataframe['event time:timestamp'].values.astype(np.int64) // 10 ** 9 
#     dataframe['time_to_complete']= (dataframe["event time:timestamp"] - dataframe["case REG_DATE"])/10**6

    # Convert the timestamps of the event time to day of week, specific day and time of that day.
    
    dataframe["day_week"] = dataframe["event time:timestamp"].apply(day_week)
    dataframe['time_of_day'] = dataframe['event time:timestamp'].apply(time_of_day)
    
    return dataframe

# ### ** Function 'encoding'**

def calculate_time_difference(dataframe):
    return dataframe.apply(lambda row: (row['next_time'] - row['event time:timestamp']).total_seconds(), axis=1)

def calculate_time_prev(dataframe):
    return dataframe.apply(lambda row: (row['event time:timestamp'] - row['prev_time']).total_seconds(), axis=1)

def encoding(dataframe):
    """Encoding 

    What kind of encoding is this exactly?
    
    Args:
        dataframe (pd.DataFrame): A pd.DataFrame in the format from the BPI_challenge 2012

    Returns:
        dataframe: A pd.DataFrame with cases and events sorted wrt time, each event has a position within its case
    """
    # sort cases wrt time, for each case sort events 
    dataframe.sort_values(['timestamp_start',"timestamp_finish"], axis=0, ascending=True, inplace=True, ignore_index=True)
    
    # assign the position in the sequence to each event
    dataframe['position'] = None
    dataframe['position'] = dataframe.groupby('case concept:name').cumcount() + 1
  
    # create columns with previous and future (times of) events

    dataframe["prev_event"] = dataframe.groupby("case concept:name")["event concept:name"].shift(1)
    dataframe["2prev_event"] = dataframe.groupby("case concept:name")["event concept:name"].shift(2)
    dataframe["next_event"] = dataframe.groupby("case concept:name")["event concept:name"].shift(-1)

    dataframe["prev_time"] = dataframe.groupby("case concept:name")["event time:timestamp"].shift(1)
    dataframe["next_time"] = dataframe.groupby("case concept:name")["event time:timestamp"].shift(-1)
    
    
    dataframe["next_event"].fillna("LAST EVENT", inplace=True)
    dataframe["prev_event"].fillna("FIRST EVENT", inplace=True)
    dataframe["2prev_event"].fillna("FIRST EVENT", inplace=True)
    
    #     these values should be empty and filling them equals creating wrong data, but otherwise models dont work :( 
    dataframe["next_time"].fillna(method='ffill', inplace=True)
    dataframe["prev_time"].fillna(method='bfill', inplace=True)    
    
    dataframe['seconds_next'] = calculate_time_difference(dataframe)
    dataframe['seconds_prev'] = calculate_time_prev(dataframe)
    
    
        # how many events are left until the end of the case
    dataframe['position inverse'] = dataframe.groupby('case concept:name').cumcount(ascending=False)+1

    # how many cases start after the current event, inclusive
    dataframe['case start count'] = dataframe[dataframe['position'] == 1].groupby('position').cumcount(ascending=False) + 1

    # how many cases end before the current event, inclusive
    dataframe['case end count'] = dataframe[dataframe['position inverse'] == 1].groupby('position inverse').cumcount() + 1

    dataframe.at[-1, 'case start count'] = 0
    dataframe.at[0, 'case end count'] = 0
    dataframe['case start count'].fillna(method='bfill', inplace = True)
    dataframe['case end count'].fillna(method='ffill', inplace = True)
    
    
    total_cases = dataframe['case start count'].loc[0]
    dataframe['active cases'] = total_cases - dataframe['case start count'].shift(periods = -1, fill_value=0) - dataframe['case end count']
    dataframe.drop(dataframe.tail(1).index,inplace=True)
    
    return dataframe
# ### **Function: 'preprocessing'**

def preprocessing(dataframe):
    """Does all the processing needed for the naive estimator

    Args:
        dataframe (pd.DataFrame): A pd.DataFrame in the format from the BPI_challenge 2012
    """
    pp_df = encoding(time_conversion(dataframe))
    
    
    return pp_df

# ## **Preprocessing and splitting**
# merge the data sets and preprocess
train = pd.concat([train, test], axis="rows", ignore_index=True)
train = preprocessing(train)
# test = preprocessing(test)
# train.drop(['index'],axis='columns',inplace=True)
# a check for duplicate events, gives error if none repeat
# pd.concat(g for _, g in train.groupby("eventID ") if len(g) > 1).tail(50)

test = train.loc[220981:]
train = train.loc[:220981]

lst=['W_Wijzigen contractgegevens', 'W_Wijzigen contractgegevens', 'A_CANCELLED', 'W_Wijzigen contractgegevens']
train['event concept:name'].any() in lst
# Find the timestamp of the first event in the test set
first_test_event_timestamp = test['event time:timestamp'].min()

# Filter out cases that end after the first test event starts or start before the first train event
train_data_filtered = train.groupby('case concept:name').filter(lambda g: g['event time:timestamp'].min() < first_test_event_timestamp and g['event time:timestamp'].max() > first_test_event_timestamp)
train.shape, test.shape
train_data_filtered.shape, test.shape
first_test_event_timestamp
# ### **Export dataframe to .CSV**

train_data_filtered.to_csv('preprocessed_train.csv')
test.to_csv('preprocessed_test.csv')





### NAIVE EVENT ESTIMATION ###





df = pd.read_csv('preprocessed_train.csv')
test = pd.read_csv('preprocessed_test.csv')

# # Naive estimator

case_ids=df["case concept:name"].unique()
event_types=df["event concept:name"].unique()

def make_dict(event_types):
    fn={}
    cnt={}
    bk={}
    for j in event_types:
        cnt[j]=0
    for j in event_types:
        fn[j]=cnt
    return fn
info=make_dict(event_types)           

case_ids=df["case concept:name"].unique()
case_ids_test=test["case concept:name"].unique()

def can():  #helper fucntion of naive_est
    fn={} #key position , value most popular event for this position
    dx=df[["index",'position',"event concept:name"]].groupby(["position","event concept:name"]).count().reset_index()
    m=df["position"].unique()
    for pos in m:
        val=dx[dx["position"]==pos]["index"].idxmax()
        pop=dx[dx["position"]==pos]["event concept:name"][val]
        fn[pos]=pop
    return fn


def  naive_est(case_ids):
    db=can()
    def predict(x):
        try:
            return db[x+1]
        except KeyError:
                return db[x]
    
    df["next_event"]= df["position"].apply(predict)
    
    example_df= df[df["case concept:name"]==case_ids[0]]
    return  example_df[["event concept:name",'next_event']].head(5)

naive_est(case_ids)    

mp={ j:index+1 for index, j in enumerate(event_types)}

def label_event(x):
    try:
        return mp[x]
    except KeyError:
        return 1
df["label_y"]= df["event concept:name"].apply(label_event)
df["label_ypred"]= df["next_event"].shift(1).apply(label_event)
df["label_ypred"].head()

y_actu = df["label_y"] 
y_pred = df["label_ypred"]

new_df=pd.DataFrame()
new_df["resu"]=y_actu.eq(y_pred)
new_df["pred"]=y_pred

def rev(x):
    if x==False:
        return 0
    if x==True:
        return 1
new_df["pred"]= new_df["resu"].apply(rev)
new_df["resu"]=new_df["resu"].apply(rev)

#confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(new_df["resu"],new_df["pred"], labels=[0, 1]).ravel()
print(tn, fp, fn, tp) 

sm= { "TP":[tp,0], "TN":[0,tn]}
dm= pd.DataFrame(sm,index=["TP","TN"])

#plot acc per label
acc_label={}
for label in range(1,25):
    curr= df[df["label_y"]==label]
    y= curr["label_y"]
    yhat= curr["label_ypred"]
    p= y.eq(yhat).sum()
    psize= curr["label_y"].size
    acc_label[label]=  p/psize
    
pm= {mp[key]:key for key in mp}
fn_vis= {pm[key]: acc_label[key] for key in acc_label }

ax =pd.DataFrame(fn_vis,index=["acc"]).stack().plot(kind="barh",figsize=(8,8))
ax.set_xlabel("Accuracy")  

f1=f1_score(df["label_y"], df["label_ypred"], average='weighted')
glob_acc= (df["label_y"].eq(df["label_ypred"])).sum() /df["label_y"].size
pre =  precision_score(df["label_y"], df["label_ypred"], average='weighted')
recall=recall_score(df["label_y"], df["label_ypred"], average='weighted')
df_stats= pd.DataFrame( {"f1_score":f1, "Accurary":glob_acc,"Precision":pre,"Recall":recall},index=["baseline"])

df_stats


# ## CPU/RAM usage

# start measuring CPU and memory usage
process = psutil.Process()

# start point
start_time = time.time()

naive_est(case_ids)

# end point
end_time = time.time()

# calculate time taken
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")

# measure CPU and memory usage
cpu_usage = process.cpu_percent()
memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
print(f"CPU usage: {cpu_usage:.2f}%")
print(f"Memory usage: {memory_usage:.2f} MB")





### NAIVE TIME ESTIMATION ###





train = pd.read_csv('preprocessed_train.csv')
test = pd.read_csv('preprocessed_test.csv')

def naive_time_estimator(train, test):
    train['next position'] = train.groupby('case concept:name')['position'].shift(-1) #Creates new column with NaN values for the last even in a case
    test['next position'] = test.groupby('case concept:name')['position'].shift(-1)
    
    train = train.dropna(subset=['next position']) #Drop all last events per case
    
    train_position = train.groupby('position')['seconds_next'].mean()
    print(train_position)
    test['prediction_seconds'] = test.apply(lambda x: train_position[x['position']]
                                            if not pd.isnull(x['next position']) else np.nan, axis = 1)
    
    return test

# naive_time_estimator(train, test)

def MAE(test):
    test2 = test.dropna(subset=['prediction_seconds','seconds_next'])
    
    y_pred = test2['prediction_seconds']
    y_true = test2['seconds_next']
    
    return mean_absolute_error(y_true, y_pred)
# MAE(test)





### MODELS ###







# pip install hypopt 

train = pd.read_csv('preprocessed_train.csv')
test = pd.read_csv('preprocessed_test.csv')


# # remove last event for each case in order to not break the model when accessing the next event time + make predictions on seconds_next + do tuning (grid search, feature selection) and cv

# # Removing the last event from each case to reduce noise in the models

train.drop(train.groupby('case concept:name').tail(1).index, axis=0, inplace=True)
test.drop(test.groupby('case concept:name').tail(1).index, axis=0, inplace=True)

# # Evaluation
# 

def time_evaluation(y_test, y_pred, model: str):
 
    print(f"Error metrics (measured in hours) for the {model} when predicting the next event's Unix timestamp")
    print('\n')
    print('Mean Absolute Error:', round(mean_absolute_error(y_test, y_pred)/3600,3))
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y_test, y_pred)/3600),3))
    print('R2 score:', round(r2_score(y_test, y_pred),3))
    
    
def event_evaluation(y_test, y_pred, model: str, avg="weighted"):

    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
    F1_score = f1_score(y_test, y_pred, average=avg, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Error metrics for the {model} when predicting the next event')
    print('\n')
    print(f'Accuracy: {round(accuracy,3)}.')
    print(f'Precision: {round(precision,3)}')
    print(f'Recall: {round(recall,3)}')
    print(f'f1-score: {round(F1_score,3)}')
#     print(confusion_matrix(y_test, y_pred))


# # Data splitting and encoding

# def make_val_set(dataframe):
#     """make a validation set from the dataframe"""
    
#     #set seed for reproducibility
#     np.random.seed(69)
    
#     #extract all unique case IDs
#     unique_ids = dataframe['case concept:name'].unique()
    
#     #select 10% of the unique IDs and use them to create a validation set
#     samples = np.random.choice(unique_ids, size=int(len(unique_ids)*0.1), replace=False)
#     val_set = dataframe[dataframe['case concept:name'].isin(samples)]
    
#     train = dataframe[~dataframe['case concept:name'].isin(samples)]
    
#     return val_set, train

train_LE = train.copy()
train_LE = train_LE.replace({'event lifecycle:transition': {'SCHEDULE': 0, 'START': 1, 'COMPLETE': 2}})

test_LE = test.copy()
test_LE = test_LE.replace({'event lifecycle:transition': {'SCHEDULE': 0, 'START': 1, 'COMPLETE': 2}})


train_OHE = pd.get_dummies(train_LE, prefix = ['current', 'prev', '2prev'], 
                           columns = ['event concept:name', 'prev_event', '2prev_event'])
test_OHE = pd.get_dummies(test_LE, prefix = ['current', 'prev', '2prev'], 
                          columns = ['event concept:name', 'prev_event', '2prev_event'])

# val_OHE = train_OHE.loc[147054:]
# train_OHE = train_OHE.loc[:147054]

# first_val_event_timestamp = val_OHE['event time:timestamp'].min()
# train_OHE = train_OHE.groupby('case concept:name').filter(lambda g: g['event time:timestamp'].min() < first_val_event_timestamp and g['event time:timestamp'].max() < first_val_event_timestamp)

# first_test_event_timestamp = test_OHE['event time:timestamp'].min()
# val_OHE = val_OHE.groupby('case concept:name').filter(lambda g: g['event time:timestamp'].min() < first_test_event_timestamp and g['event time:timestamp'].max() < first_test_event_timestamp)

# train_OHE.shape, val_OHE.shape, test_OHE.shape, train_LE.shape, test_LE.shape

temp3 = []
for element in list(train_OHE.columns):
    if element not in list(test_OHE.columns):
        temp3.append(element)
print('events that happen in the train but not the test set: ' + str(temp3))

features_time = ['timestamp_finish', 'seconds_prev', 'active cases', 'day_week', 'time_of_day', 
                 'case AMOUNT_REQ', 'event lifecycle:transition', 
       'current_A_ACCEPTED', 'current_A_ACTIVATED', 'current_A_APPROVED',
       'current_A_CANCELLED', 'current_A_DECLINED', 'current_A_FINALIZED',
       'current_A_PARTLYSUBMITTED', 'current_A_PREACCEPTED',
       'current_A_REGISTERED', 'current_A_SUBMITTED', 'current_O_ACCEPTED',
       'current_O_CANCELLED', 'current_O_CREATED', 'current_O_DECLINED',
       'current_O_SELECTED', 'current_O_SENT', 'current_O_SENT_BACK',
       'current_W_Afhandelen leads', 'current_W_Beoordelen fraude',
       'current_W_Completeren aanvraag',
       'current_W_Nabellen incomplete dossiers', 'current_W_Nabellen offertes',
       'current_W_Valideren aanvraag', 'current_W_Wijzigen contractgegevens']

features_time_test = ['timestamp_finish', 'seconds_prev', 'active cases', 'day_week', 'time_of_day', 
                 'case AMOUNT_REQ', 'event lifecycle:transition', 
       'current_A_ACCEPTED', 'current_A_ACTIVATED', 'current_A_APPROVED',
       'current_A_CANCELLED', 'current_A_DECLINED', 'current_A_FINALIZED',
       'current_A_PARTLYSUBMITTED', 'current_A_PREACCEPTED',
       'current_A_REGISTERED', 'current_A_SUBMITTED', 'current_O_ACCEPTED',
       'current_O_CANCELLED', 'current_O_CREATED', 'current_O_DECLINED',
       'current_O_SELECTED', 'current_O_SENT', 'current_O_SENT_BACK',
       'current_W_Afhandelen leads', 'current_W_Beoordelen fraude',
       'current_W_Completeren aanvraag',
       'current_W_Nabellen incomplete dossiers', 'current_W_Nabellen offertes',
       'current_W_Valideren aanvraag']
# features_time_test = features_time.copy()

target_time = 'seconds_next'

features_event = ['active cases', 'day_week', 'time_of_day',  
                  'event lifecycle:transition', 'case AMOUNT_REQ', 
       'current_A_ACCEPTED', 'current_A_ACTIVATED', 'current_A_APPROVED',
       'current_A_CANCELLED', 'current_A_DECLINED', 'current_A_FINALIZED',
       'current_A_PARTLYSUBMITTED', 'current_A_PREACCEPTED',
       'current_A_REGISTERED', 'current_A_SUBMITTED', 'current_O_ACCEPTED',
       'current_O_CANCELLED', 'current_O_CREATED', 'current_O_DECLINED',
       'current_O_SELECTED', 'current_O_SENT', 'current_O_SENT_BACK',
       'current_W_Afhandelen leads', 'current_W_Beoordelen fraude',
       'current_W_Completeren aanvraag',
       'current_W_Nabellen incomplete dossiers', 'current_W_Nabellen offertes',
       'current_W_Valideren aanvraag', 'current_W_Wijzigen contractgegevens',
       'prev_A_ACCEPTED', 'prev_A_ACTIVATED', 'prev_A_APPROVED',
       'prev_A_CANCELLED', 'prev_A_DECLINED', 'prev_A_FINALIZED',
       'prev_A_PARTLYSUBMITTED', 'prev_A_PREACCEPTED', 'prev_A_REGISTERED',
       'prev_A_SUBMITTED', 'prev_FIRST EVENT', 'prev_O_ACCEPTED',
       'prev_O_CANCELLED', 'prev_O_CREATED', 'prev_O_DECLINED',
       'prev_O_SELECTED', 'prev_O_SENT', 'prev_O_SENT_BACK',
       'prev_W_Afhandelen leads', 'prev_W_Beoordelen fraude',
       'prev_W_Completeren aanvraag', 'prev_W_Nabellen incomplete dossiers',
       'prev_W_Nabellen offertes', 'prev_W_Valideren aanvraag',
       'prev_W_Wijzigen contractgegevens', '2prev_A_ACCEPTED',
       '2prev_A_ACTIVATED', '2prev_A_APPROVED', '2prev_A_CANCELLED',
       '2prev_A_FINALIZED', '2prev_A_PARTLYSUBMITTED', '2prev_A_PREACCEPTED',
       '2prev_A_REGISTERED', '2prev_A_SUBMITTED', '2prev_FIRST EVENT',
       '2prev_O_ACCEPTED', '2prev_O_CANCELLED', '2prev_O_CREATED',
       '2prev_O_SELECTED', '2prev_O_SENT', '2prev_O_SENT_BACK',
       '2prev_W_Afhandelen leads', '2prev_W_Beoordelen fraude',
       '2prev_W_Completeren aanvraag', '2prev_W_Nabellen incomplete dossiers',
       '2prev_W_Nabellen offertes', '2prev_W_Valideren aanvraag',
       '2prev_W_Wijzigen contractgegevens']

features_event_test = ['active cases', 'day_week', 'time_of_day',  
                  'event lifecycle:transition', 'case AMOUNT_REQ', 
       'current_A_ACCEPTED', 'current_A_ACTIVATED', 'current_A_APPROVED',
       'current_A_CANCELLED', 'current_A_DECLINED', 'current_A_FINALIZED',
       'current_A_PARTLYSUBMITTED', 'current_A_PREACCEPTED',
       'current_A_REGISTERED', 'current_A_SUBMITTED', 'current_O_ACCEPTED',
       'current_O_CANCELLED', 'current_O_CREATED', 'current_O_DECLINED',
       'current_O_SELECTED', 'current_O_SENT', 'current_O_SENT_BACK',
       'current_W_Afhandelen leads', 'current_W_Beoordelen fraude',
       'current_W_Completeren aanvraag',
       'current_W_Nabellen incomplete dossiers', 'current_W_Nabellen offertes',
       'current_W_Valideren aanvraag',
       'prev_A_ACCEPTED', 'prev_A_ACTIVATED', 'prev_A_APPROVED',
       'prev_A_CANCELLED', 'prev_A_DECLINED', 'prev_A_FINALIZED',
       'prev_A_PARTLYSUBMITTED', 'prev_A_PREACCEPTED', 'prev_A_REGISTERED',
       'prev_A_SUBMITTED', 'prev_FIRST EVENT', 'prev_O_ACCEPTED',
       'prev_O_CANCELLED', 'prev_O_CREATED', 'prev_O_DECLINED',
       'prev_O_SELECTED', 'prev_O_SENT', 'prev_O_SENT_BACK',
       'prev_W_Afhandelen leads', 'prev_W_Beoordelen fraude',
       'prev_W_Completeren aanvraag', 'prev_W_Nabellen incomplete dossiers',
       'prev_W_Nabellen offertes', 'prev_W_Valideren aanvraag',
       '2prev_A_ACCEPTED', '2prev_A_ACTIVATED', '2prev_A_APPROVED',
       '2prev_A_FINALIZED', '2prev_A_PARTLYSUBMITTED', '2prev_A_PREACCEPTED',
       '2prev_A_REGISTERED', '2prev_A_SUBMITTED', '2prev_FIRST EVENT',
       '2prev_O_ACCEPTED', '2prev_O_CANCELLED', '2prev_O_CREATED',
       '2prev_O_SELECTED', '2prev_O_SENT', '2prev_O_SENT_BACK',
       '2prev_W_Afhandelen leads', '2prev_W_Beoordelen fraude',
       '2prev_W_Completeren aanvraag', '2prev_W_Nabellen incomplete dossiers',
       '2prev_W_Nabellen offertes', '2prev_W_Valideren aanvraag']
# features_event_test = features_event.copy()

target_event = 'next_event'


# TIME
X_train_time = train_OHE[features_time]
y_train_time = train_OHE[target_time]

# X_val_time = val_OHE[features_time]
# y_val_time = val_OHE[target_time]

X_test_time = test_OHE[features_time_test].copy()
y_test_time = test_OHE[target_time]


# EVENT
X_train_event = train_OHE[features_event]
y_train_event = train_OHE[target_event]

# X_val_event = val_OHE[features_event]
# y_val_event = val_OHE[target_event]

X_test_event = test_OHE[features_event_test].copy()
y_test_event = test_OHE[target_event]



X_test_time['current_W_Wijzigen contractgegevens']=-1
X_test_event[['current_W_Wijzigen contractgegevens', 'prev_W_Wijzigen contractgegevens', '2prev_A_CANCELLED', '2prev_W_Wijzigen contractgegevens']]=-1
# X_test_event[['current_W_Wijzigen contractgegevens']]=0


X_train_event.shape, X_train_time.shape, X_test_event.shape, X_test_time.shape


# ohe = OneHotEncoder().fit(train['event concept:name'].to_numpy().reshape(-1, 1))

# transformed = ohe.transform(train['event concept:name'].to_numpy().reshape(-1, 1))
# train_OHE = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names())


# # Event prediction

def RandomForestEvents(X_train, y_train, X_val, y_val):
    
    
    params={'max_depth': [None],
     'n_estimators': [1000, 1200, 1400, 1600]}

    
    forest_clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='sqrt', random_state=42)    
    
    grid = GridSearch(model=forest_clf, param_grid=params, parallelize=False)

    grid.fit(X_train, y_train, X_val, y_val)
    
    return grid.best_estimator_


rf_event = RandomForestEvents(X_train_event, y_train_event, X_test_event, y_test_event)


rf_event_train = rf_event.predict(X_train_event)
# rf_event_val = rf_event.predict(X_val_event)
rf_event_test = rf_event.predict(X_test_event)


print(rf_event.get_params())

event_evaluation(y_train_event, rf_event_train, 'RF EVENT TRAIN')
# event_evaluation(y_val_event, rf_event_val, 'RF EVENT VAL')
event_evaluation(y_test_event, rf_event_test, 'RF EVENT TEST')


# # Time prediction


def RandomForestTime(X_train, y_train, X_val, y_val):
    
    
    params={'max_depth': [None],
     'n_estimators': [1000,1200,1400,1600]}

    forest_reg = RandomForestRegressor(bootstrap = True, max_features='sqrt', random_state=42)
    
    grid = GridSearch(model=forest_reg, param_grid=params, parallelize=False)

    grid.fit(X_train, y_train, X_val, y_val)
    
    return grid.best_estimator_


rf_time = RandomForestTime(X_train_time, y_train_time, X_test_time, y_test_time)

rf_time_train = rf_time.predict(X_train_time)
# rf_time_val = rf_time.predict(X_val_time)
rf_time_test = rf_time.predict(X_test_time)

print(rf_time.get_params())

time_evaluation(y_train_time, rf_time_train, 'RF TIME TRAIN')
# time_evaluation(y_val_time, rf_time_val, 'RF TIME VAL')
time_evaluation(y_test_time, rf_time_test, 'RF TIME TEST')


# # Decision Tree time prediction

from sklearn.tree import DecisionTreeRegressor

def DecisionTreeTime(X_train, X_test, y_train, y_test):
    '''no grid search, no cv, or feature selection, do for final model
    '''
    
    X_train_time = X_train[['event concept:name','prev_event', '2prev_event', 'seconds_prev', 'timestamp_finish']]
    y_train_time = y_train['seconds_next']

    X_test_time = X_test[['event concept:name','prev_event', '2prev_event', 'seconds_prev', 'timestamp_finish']]
    y_test_time = y_test['seconds_next']
    
    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(X_train_time, y_train_time)
    
    y_pred = dt_reg.predict(X_test_time)

    return dt_reg, y_pred

dt_time, y_pred_dt_time = DecisionTreeTime(X_train, X_test, y_train, y_test)
time_evaluation(y_test['seconds_next'], y_pred_dt_time, 'Decision tree regressor')

# get feature importances and their names
feature_importances_dt_time = dt_time.feature_importances_
feature_names_dt_time = ['seconds_prev', 'timestamp_finish', 'prev_event', 'event concept:name', '2prev_event']

# sort the features by importance in descending order
inx = np.argsort(feature_importances_dt_time)[::-1]
sorted_imp_dt_time = feature_importances_dt_time[inx]

# ### Naive and Random Forest estimators accuracy and precision

# plot the naive and random forest f1, acc, pre, recall
metrics_df = pd.DataFrame({
    ' ': ['F1 Score', 'Accuracy', 'Precision', 'Recall'],
    'Naive': ['0.266', '0.346', '0.297', '0.346'],
    'Random Forest': ['0.748', '0.785', '0.772', '0.785']    
})

metrics_df.to_string(index=False)





### DECISION TREE ### 






def importance(model, X_val, y_val):
    r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X_val.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

def tune_rf_activity(train_data, columns):
    X = train_data[columns]
    y = train_data[constants.NEXT_EVENT]

    
    ######## Parameters #######
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 15, 50]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False] # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = random_grid, 
                                   n_iter = 10, 
                                   cv = 3, 
                                   verbose=0, 
                                   random_state=42, 
                                   n_jobs = -1)
    rf_random.fit(X, y)

    preds = rf_random.best_estimator_.predict(X)
    
    
    train_data[constants.NEXT_EVENT_PREDICTION] = preds

    print(rf_random.best_estimator_)
    print(accuracy_score(preds,y))

    return rf_random.best_estimator_

def train_activity_model(train_data, clf, columns, normal=True):
    X = train_data[columns]

    if not normal:
        y = train_data['next_activity_id']
    else:
        y = train_data[constants.NEXT_EVENT]

    clf.fit(X,y)
    preds = clf.predict(X)
    
    train_data[constants.NEXT_EVENT_PREDICTION] = preds
    

    classification_performance(train_data, "Confusion_Matrices/conf_matrix_random_forest_train.png")
    
    print(accuracy_score(preds,y))
    return clf

def train_time_model(train_data, clf, columns):
    X = train_data[columns]

    y = train_data[constants.TIME_DIFFERENCE]

    clf.fit(X,y)
    preds = clf.predict(X)

    train_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    regression_performance(train_data)

    print(mean_absolute_error(preds,y))
    return clf

def test_activity_model(test_data, clf, columns, normal=True):
    X = test_data[columns]

    if not normal:
        y = test_data['next_activity_id']
    else:
        y = test_data[constants.NEXT_EVENT]
    
    preds = clf.predict(X)

    test_data[constants.NEXT_EVENT_PREDICTION] = preds
    
    classification_performance(test_data, "Confusion_Matrices/conf_matrix_random_forest_test.png")

    acc = accuracy_score(preds, y)
    print(acc)
    return acc

def test_time_model(test_data, clf, columns):
    X = test_data[columns]

    y = test_data[constants.TIME_DIFFERENCE]
    preds = clf.predict(X)

    test_data[constants.TIME_DIFFERENCE_PREDICTION] = preds

    regression_performance(test_data)
    
    return 1

def compare_all_models(train_data, test_data, timer):
    cols = ['activity number in case', 'case end count',
            'days_until_next_holiday', 'is_weekend', 
            'seconds_since_week_start', 'is_work_time', 
            'seconds_to_work_hours', 'is_holiday',
            'workrate', 'active cases']

    # ----------------- TRAIN DATASET ---------------------------------
    # copy so we don't modify the original training set
    train_data = copy.deepcopy(train_data).rename(columns={constants.CASE_POSITION_COLUMN: 'name'})
    names_ohe = pd.get_dummies(train_data['name'])
    first_lag_ohe = pd.get_dummies(train_data['first_lag_event']).add_prefix('first_lag_')
    second_lag_ohe = pd.get_dummies(train_data['second_lag_event']).add_prefix('second_lag_')

    cols_train = list(names_ohe.columns) + list(first_lag_ohe.columns) + list(second_lag_ohe.columns)

    train_data = train_data.drop('name', axis=1).join(names_ohe).join(first_lag_ohe).join(second_lag_ohe).dropna()
    train_data['next_activity_id'] = pd.factorize(train_data[constants.NEXT_EVENT])[0]
   
    # --------------------- TEST DATASET -------------------------------
    # copy test dataset
    test_data = copy.deepcopy(test_data).rename(columns={constants.CASE_POSITION_COLUMN: 'name'})
    names_ohe = pd.get_dummies(test_data['name'])
    first_lag_ohe = pd.get_dummies(test_data['first_lag_event']).add_prefix('first_lag_')
    second_lag_ohe = pd.get_dummies(test_data['second_lag_event']).add_prefix('second_lag_')
    
    cols_test = list(names_ohe.columns) + list(first_lag_ohe.columns) + list(second_lag_ohe.columns)

    test_data = test_data.drop('name', axis=1).join(names_ohe).dropna().join(first_lag_ohe).join(second_lag_ohe).dropna()
    test_data['next_activity_id'] = pd.factorize(test_data[constants.NEXT_EVENT])[0]
    
    cols.extend([el for el in cols_train if el in cols_test])

    timer.send("Time to prepare columns (in seconds): ")
    
    print("Decision Tree:")
    print("-----------------------------")
    print("Next activity:")
    clf = DecisionTreeClassifier()
    dec_tree_clas = train_activity_model(train_data, clf, cols)

    timer.send("Time to train decision tree classifier (in seconds): ")

    test_activity_model(test_data, dec_tree_clas, cols)
    
    timer.send("Time to evaluate decision tree classifier (in seconds): ")

    print("Random Forest:")
    print("-----------------------------")
    print("Next activity:")
    clf = RandomForestClassifier()
    rand_forest_class = train_activity_model(train_data, clf, cols)

    timer.send("Time to train random forest classifier (in seconds): ")

    test_activity_model(test_data, rand_forest_class, cols)
    
    timer.send("Time to evaluation random forest classifier (in seconds): ")

    print("Linear Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = LinearRegression()
    lin_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train linear regression (in seconds): ")

    test_time_model(test_data, lin_regr, cols)

    timer.send("Time to evaluate linear regression (in seconds): ")

    print("Random Forest Regression:")
    print("-----------------------------")
    print("Time to next activity:")
    reg = RandomForestRegressor()
    rand_forest_regr = train_time_model(train_data, reg, cols)

    timer.send("Time to train random forest regression (in seconds): ")

    test_time_model(test_data, rand_forest_regr, cols)

    timer.send("Time to evaluate random forest regression (in seconds): ")







### ELASTICNET ###






train = pd.read_csv('preprocessed_train.csv')
test = pd.read_csv('preprocessed_test.csv')

train['next position'] = train.groupby('case concept:name')['position'].shift(-1) #Creates new column with NaN values for the last even in a case
test['next position'] = test.groupby('case concept:name')['position'].shift(-1)

train = train.dropna(subset=['next position']) #Drop all last events per case
test = test.dropna(subset=['next position'])

train_LE = train.copy()
train_LE = train_LE.replace({'event lifecycle:transition': {'SCHEDULE': 0, 'START': 1, 'COMPLETE': 2}})

train_OHE = pd.get_dummies(train_LE, prefix=['type'], columns = ['event concept:name'])
test_OHE = pd.get_dummies(train_LE, prefix=['type'], columns = ['event concept:name'])

def make_val_set(dataframe):
    """make a validation set from the dataframe"""
    
    #set seed for reproducibility
    np.random.seed(69)
    
    #extract all unique case IDs
    unique_ids = dataframe['case concept:name'].unique()
    
    #select 10% of the unique IDs and use them to create a validation set
    samples = np.random.choice(unique_ids, size=int(len(unique_ids)*0.1), replace=False)
    val_set = dataframe[dataframe['case concept:name'].isin(samples)]
    
    train = dataframe[~dataframe['case concept:name'].isin(samples)]
    
    return val_set, train

val_OHE, train_OHE = make_val_set(train_OHE)


x_train_time = train_OHE[['case AMOUNT_REQ','timestamp_finish', 'day_week', 'time_of_day','seconds_prev', 'type_A_ACCEPTED', 'type_A_ACTIVATED', 'type_A_APPROVED','type_A_CANCELLED', 'type_A_DECLINED', 'type_A_FINALIZED','type_A_PARTLYSUBMITTED', 'type_A_PREACCEPTED', 'type_A_REGISTERED','type_A_SUBMITTED', 'type_O_ACCEPTED', 'type_O_CANCELLED','type_O_CREATED', 'type_O_DECLINED', 'type_O_SELECTED', 'type_O_SENT','type_O_SENT_BACK', 'type_W_Afhandelen leads','type_W_Beoordelen fraude', 'type_W_Completeren aanvraag','type_W_Nabellen incomplete dossiers', 'type_W_Nabellen offertes','type_W_Valideren aanvraag', 'type_W_Wijzigen contractgegevens']]

y_train_time = train_OHE['seconds_next']

x_val_time = val_OHE[['case AMOUNT_REQ','timestamp_finish', 'day_week',
 'time_of_day','seconds_prev', 'type_A_ACCEPTED', 'type_A_ACTIVATED', 'type_A_APPROVED',
 'type_A_CANCELLED', 'type_A_DECLINED', 'type_A_FINALIZED','type_A_PARTLYSUBMITTED', 'type_A_PREACCEPTED', 
 'type_A_REGISTERED','type_A_SUBMITTED', 'type_O_ACCEPTED', 'type_O_CANCELLED','type_O_CREATED', 'type_O_DECLINED', 
 'type_O_SELECTED', 'type_O_SENT','type_O_SENT_BACK', 'type_W_Afhandelen leads','type_W_Beoordelen fraude', 
 'type_W_Completeren aanvraag','type_W_Nabellen incomplete dossiers', 'type_W_Nabellen offertes','type_W_Valideren aanvraag',
 'type_W_Wijzigen contractgegevens']]

y_val_time = val_OHE['seconds_next']

x_test_time = test_OHE[['case AMOUNT_REQ','timestamp_finish', 'day_week',
 'time_of_day','seconds_prev', 'type_A_ACCEPTED', 'type_A_ACTIVATED', 'type_A_APPROVED',
 'type_A_CANCELLED', 'type_A_DECLINED', 'type_A_FINALIZED','type_A_PARTLYSUBMITTED', 'type_A_PREACCEPTED', 
 'type_A_REGISTERED','type_A_SUBMITTED', 'type_O_ACCEPTED', 'type_O_CANCELLED','type_O_CREATED', 'type_O_DECLINED', 
 'type_O_SELECTED', 'type_O_SENT','type_O_SENT_BACK', 'type_W_Afhandelen leads','type_W_Beoordelen fraude', 
 'type_W_Completeren aanvraag','type_W_Nabellen incomplete dossiers', 'type_W_Nabellen offertes','type_W_Valideren aanvraag',
 'type_W_Wijzigen contractgegevens']]
 
y_test_time = test_OHE['seconds_next']

y_test_time



# Alpha in sklearn is lambda
# 

parameters = {"l1_ratio": [.1, .3 ,.5,.85,.95, .99, 1],
              'alpha':[0.1,0.3,0.5,0.7,0.9,1],
              'max_iter': [4000,5000]}


regr = ElasticNetCV(cv=5,random_state=2,l1_ratio = l1,alphas = alphas,max_iter=5000 ) 
regr.fit(x_train_time, y_train_time)


y_pred_time = regr.predict(x_test_time)

def time_evaluation(y_test, y_pred, model: str):
 
    print(f"Error metrics (measured in hours) for the {model} when predicting the time until next event")
    print('\n')
    print('Mean Absolute Error:', round(mean_absolute_error(y_test, y_pred)/3600,3))
    print('Root Mean Squared Error:', round(np.sqrt(mean_squared_error(y_test, y_pred)/3600),3))
    print('R2 score:', round(r2_score(y_test, y_pred),3))

time_evaluation(y_test_time, y_pred_time, 'Elastic net')

eNet = ElasticNet()
grid = GridSearchCV(eNet, param_grid = parameters,scoring='r2',cv=5,verbose=1)
grid.fit(x_train_time,y_train_time)
y_pred_grid = grid.predict(x_test_time)

time_evaluation(y_test_time,y_pred_grid,'Elastic net grid search')






### SVR ###





df_train = pd.read_csv('BPI_Challenge_2012-training.csv')
df_test = pd.read_csv('BPI_Challenge_2012-test.csv')

df_train.drop('eventID ', axis=1, inplace=True)
df_test.drop('eventID ', axis=1, inplace=True)


# # Data preprocessing

#Apply label encoding to event lifecycle:transistion column in train data. 0: schedule, 1: start, 2: complete
df_train_LE = df_train.copy()
df_train_LE = df_train_LE.replace({'event lifecycle:transition': {'SCHEDULE': 0, 'START': 1, 'COMPLETE': 2}})
#Note: df_train_LE stands for 'dataframe train label encoded'


#One hot encoding for event concept:name column in train data
df_train_OHE = pd.get_dummies(df_train_LE, prefix=['type'], columns = ['event concept:name'])
#uncomment to show the dataframe
# df_train_OHE 
#Note: df_train_OHE stands for 'dataframe train one hot encoded'


#To show NaN values are all for the same cases. Therefore these rows can dropped from the dataframe
df_train_OHE[df_train_OHE['event org:resource'].isna()].head()


#Normalize case AMOUNT_REQ and event org:resource columns in train data

#drop NaN values
df_train_OHE = df_train_OHE.dropna()

#to_norm are the to-normalize columns, norm are the normalized columns
to_norm = df_train_OHE[['case AMOUNT_REQ', 'event org:resource']].copy()
norm = pd.DataFrame(Normalizer().fit_transform(to_norm), columns=to_norm.columns)
#norm is converted to a dataframe and the columns are renamed to the original column names

#append/replace the normalized columns to the dataframe
df_train_OHE_norm = df_train_OHE.copy()
df_train_OHE_norm['case AMOUNT_REQ'] = norm['case AMOUNT_REQ'].values
df_train_OHE_norm['event org:resource'] = norm['event org:resource'].values
#Note: df_train_OHE_norm stands for 'dataframe train one hot encoded normalized'

df_train_OHE_norm