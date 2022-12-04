import tweepy
import pandas as pd
import json
from datetime import datetime
import s3fs
import re
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np


def run_twitter_etl():

    access_key = "you're access_key"
    access_secret = "you're access_secret"
    consumer_key = "you're consumer_key"
    consumer_sceret = "you're consumer_sceret"


    # Twitter authentication
    auth = tweepy.OAuthHandler(access_key, access_secret)
    auth.set_access_token(consumer_key, consumer_sceret)

    # # # Creating an API object
    api = tweepy.API(auth)
    tweets = api.user_timeline(screen_name='@GeorgeMasonU',
                            # 200 is the maximum allowed count
                            count=500,
                            include_rts = False,
                            # Necessary to keep full_text
                            # otherwise only the first 140 words are extracted
                            tweet_mode = 'extended'
                            )


    list = []
    for tweet in tweets:
        text = tweet._json["full_text"]

        refined_tweet = {"user": tweet.user.screen_name,
                        'text' : text,
                        'favorite_count' : tweet.favorite_count,
                        'retweet_count' : tweet.retweet_count,
                        'created_at' : tweet.created_at}

        list.append(refined_tweet)


    df = pd.DataFrame(list)
    #return df

    df.to_csv('newfile.csv', index = False)

run_twitter_etl()

new_df = pd.read_csv(r'newfile.csv')
new_df

new = new_df.loc[:,"text"]
df_new = pd.DataFrame(new)
df_new

corpus = df_new['text'].str.lower()
df_l = pd.DataFrame(corpus)
df_l

df_space = df_l['text'].str.strip()
df_em = pd.DataFrame(df_space)
df_em

regex=[]
for x in df_em['text']:
    corpus = re.sub(r'https?://\S+|www\.\S+', '', x)
    corpus = re.sub(r'^\s+|\s+$', '', corpus)
    corpus = re.sub('\[.*?\]', '', corpus)
    corpus = re.sub('<.*?>+', '', corpus)
    corpus = re.sub('\n', '', corpus)
    corpus = re.sub('\w*\d\w*', '', corpus)
    corpus=re.sub(r'@[A-Za-z0-9]+','',corpus)
    #corpus=re.sub(r'#','', corpus)
    corpus=re.sub(r'RT[\s]+','',corpus)
    corpus=re.sub(r'[^\w]', ' ', corpus)
    corpus=re.sub(r"georgemasonu", ' ',corpus)

    regex.append(corpus)

df_em["new"]=regex
df_em

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model(input)

event_description = []
for name in df_em["new"]:
    event_description.append(name)

event_description_embedding=embed(event_description)
event_description_embedding

event_description_arr= event_description_embedding.numpy()
event_description_arr

event_desc=[]
for vector in  event_description_arr:
    event_desc.append(vector)
event_desc

df_em['embedded_event_desc']=event_desc
df_em

import numpy as np
from numpy.linalg import norm

arr=[]
for i in range(len(df_em)):
    c=0
    a=[]
    for j in range(len(df_em)):
      cosine = np.dot(df_em['embedded_event_desc'][i],df_em['embedded_event_desc'][j])/(norm(df_em['embedded_event_desc'][i])*norm(df_em['embedded_event_desc'][j]))
      a.append(cosine)
    arr.append(a)

df_em['similarity']=arr

df_em['similarity']

import numpy
km = []
#newArr = np.reshape(df_em['similarity'], (-1, 1))
for i in df_em['similarity']:
  a = numpy.array(i)
  #a.reshape(1, -1)
  km.append(a)
km

# standardizing the data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
data_scaled = scaler.fit_transform(km)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()
Kmean = KMeans(n_clusters=5)
Kmean.fit(pd.DataFrame(data_scaled))

#predict the labels of clusters.
label = Kmean.fit_predict(data_scaled)
print(label)

#Importing required modules
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def plots():
  u_labels = np.unique(label)
  # print(u_labels)
  #plotting the results:
  for i in u_labels:
    plt.scatter(data_scaled[label == i , 0] , data_scaled[label == i , 1] , label = i)
  plt.legend()
  plt.show()
plots()

cluster_map = pd.DataFrame()
def cluster_disp():
  cluster_map['index'] = df_em['similarity'].index.values
  cluster_map['cluster'] = Kmean.labels_
  cluster_map['cluster']
cluster_disp()

df=df_em

df_em['index']=df_em.index

df_em

cluster_map['dat'] = pd.Series(dtype='str')

def display_cluster():
  for index, row in cluster_map.iterrows():
    cluster_map.at[index,'dat']=df_em.loc[df_em['index'] == index]['text']
display_cluster()

cluster_map[cluster_map.cluster==0]

def cluster_dat():
  cluster_map.to_csv("GMU_data.csv", index=False)
  return cluster_map
cluster_dat()
