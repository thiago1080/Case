import numpy as np
import plotly.express as px
import gensim
from pprint import pprint
import os
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import gensim.corpora as corpora
from wordcloud import WordCloud
import re
import pandas as pd
from functions import adjust, todf, sent_to_words, remove_stopwords, search_topic, tonum
import os


df = pd.read_excel('inputs.xlsm')
### Vamos aplicar algumas funções para ajustar os dados
df = df.iloc[:,0].apply(adjust)
df = todf(df)
### Teste de consistência
ii = []
for n,i in enumerate(df[5].values):
    try:
        float(i)
    except:
        ii.append(n)

df = df.reset_index().rename(columns = dict(zip(['index'] + list(range(df.shape[1])), ['name' ,'description','employees','total_funding','city','subcountry','lat','lng'])))

df['total_funding'] = df['total_funding'].apply(lambda x: str(x).lower()).map({'nan':np.nan})
df['lat'] = df['lat'].apply(tonum)
df['lng'] = df['lng'].apply(tonum)
df['total_funding'] = df['total_funding'].apply(tonum)
df['employees'] = df['employees'].apply(tonum)

print('% de Nan')
(df.isna().sum() / df.shape[0]*100).apply(lambda x: f'{x:.2f}%')
df.drop('total_funding', axis=1, inplace = True)
df = df.dropna()
df

### Data Cleaning
# Remove punctuation
df['description'] = \
df['description'].map(lambda x: re.sub('[,\.!?]', '', x))
### Wordcloud
# Import the wordcloud library
# Join the different processed titles together.
long_string = ','.join(list(df['description'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()
### Pre processamento
data = df.description.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])
### Criar Corpus
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]