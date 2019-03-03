
# coding: utf-8

# In[2]:


import pandas as pd
import re
import numpy as np 
import matplotlib.pyplot as plt 
#!pip install seaborn
import seaborn as sns
import string
#!pip install nltk
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('train_tweets.csv',encoding = 'latin',names=["target","id","date","Query","user","tweet"])


# In[117]:


#Partir el dataset
df.head()
df1 = df.loc[:200000,:]
df2 = df.loc[400000:799999,:]
df3 = df.loc[800000:1199999,:]
df4 = df.loc[1200000:,:]
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)


# In[9]:


grouped = df.groupby("target")


# In[10]:


grouped.count()["id"]


# In[118]:


#Metodo que remueve un patrón usado para quitar las menciones @user
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)    
    return input_txt 


# In[119]:


#Se elminan las menciones del dataset
df1.loc[:,'preprocess'] = np.vectorize(remove_pattern)(df1['tweet'], "@[\w]*")


# In[42]:


df2.loc[:,'preprocess'] = np.vectorize(remove_pattern)(df2['tweet'], "@[\w]*")
df3.loc[:,'preprocess'] = np.vectorize(remove_pattern)(df3['tweet'], "@[\w]*")
df4.loc[:,'preprocess'] = np.vectorize(remove_pattern)(df4['tweet'], "@[\w]*")


# In[104]:


#SE USA PARA VISUALIZAR EL DATASET ANTES Y DESPUES DEL PREPROCESS
df1.head().loc[:,"tweet":"preprocess"]


# In[111]:


df2.head()


# In[112]:


df3.head()


# In[113]:


df4.head()


# In[105]:


#Se remueven las url
df1.loc[:,'preprocess'] = df1.loc[:,'preprocess'].str.replace("https\S+|http\S+|www.\S+","",case = False)
#Se remueven las apostrofes
df1.loc[:,'preprocess'] = df1.loc[:,'preprocess'].str.replace("'","")
#Se remueven los caracteres especiales
df1.loc[:,'preprocess'] = df1.loc[:,'preprocess'].str.replace("[^a-zA-Z#]"," ")


# In[106]:


#Se elminan todas las palabras de menos de 3 letras
df1.loc[:,'preprocess'] = df1.loc[:,'preprocess'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[107]:


#Se convierten las palabras el tweet a elementos de un arreglo
tokenized_tweet = df1.loc[:,'preprocess'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[108]:


from nltk.stem.porter import *
#stemmer = PorterStemmer()
#Librerìa para manejar los verbos en sus distintas formas (falta validar si es buena opción)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


# In[109]:


#Despues de transformar los verbos, los ponemos en el dataset
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df1.loc[:,'preprocess'] = tokenized_tweet


# In[110]:


df1.loc[:,"tweet":'preprocess']


# In[112]:


#!pip install wordcloud
#Graficar las palabras en una nube de palabras, actualmente no me deja instalar la librería por memoria
all_words = ' '.join([text for text in df1.loc[:,'preprocess']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[116]:


get_ipython().system('pip install wordcloud')


# In[114]:


get_ipython().system('sudo swapon -s')

