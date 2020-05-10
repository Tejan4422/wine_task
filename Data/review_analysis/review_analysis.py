import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

traindf = pd.read_csv('train.csv')

print("Total number of examples: ", traindf.shape[0])
print("Number of examples with the same title and description: ", traindf[traindf.duplicated(['review_description','review_title'])].shape[0])

dropped_duplicates=traindf.drop_duplicates(['review_description','review_title'])
dropped_duplicates=dropped_duplicates.reset_index(drop=True)

dropped_duplicates.info()
dropped_duplicates.isna().sum()
dropped_duplicates.nunique()


plt.figure(figsize=(14,16))

winery = dropped_duplicates.winery.value_counts()[:20]

g = sns.countplot(x='winery', 
                  data=dropped_duplicates.loc[(dropped_duplicates.winery.isin(winery.index.values))], 
                  color='darkgreen')
g.set_title("TOP 20 most frequent Winery's", fontsize=20)
g.set_xlabel(" ", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.savefig('Top 20 wineries.png', dpi = 100)
plt.show()

g1 = sns.boxplot(y='price', x='winery',
                  data=dropped_duplicates.loc[(dropped_duplicates.winery.isin(winery.index.values))],
                 color='darkgreen')
g1.set_title("Price by Winery's", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Price", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
plt.savefig('price vs winery.png', dpi = 100)
plt.show()

g2 = sns.boxplot(y='points', x='winery',
                  data=dropped_duplicates.loc[(dropped_duplicates.winery.isin(winery.index.values))],
                 color='darkgreen')
g2.set_title("Points by Winery's", fontsize=20)
g2.set_xlabel("Winery's", fontsize=15)
g2.set_ylabel("Points", fontsize=15)
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
plt.savefig('winery vs points.png', dpi = 100)
plt.show()



dropped_duplicates = dropped_duplicates.assign(desc_length = dropped_duplicates['review_description'].apply(len))

plt.figure(figsize=(14,6))
g = sns.boxplot(x='points', y='desc_length', data=dropped_duplicates,
                color='darkgreen')
g.set_title('Description Length by Points', fontsize=20)
g.set_ylabel('Description Length', fontsize = 16) # Y label
g.set_xlabel('Points', fontsize = 16) # X label
plt.savefig('description length vs points.png', dpi = 100)
plt.show()


plt.figure(figsize=(14,6))
g = sns.regplot(x='desc_length', y='price',
                data=dropped_duplicates, fit_reg=True, color='cyan', )
g.set_title('Price by Description Length', fontsize=20)
g.set_ylabel('Price', fontsize = 16) 
g.set_xlabel('Description Length', fontsize = 16)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.savefig('descriptoinLength.png', dpi = 100)
plt.show()

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

newStopWords = ['fruit', "Drink", "black", 'wine', 'drink']

stopwords.update(newStopWords)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=300,
    max_font_size=200, 
    width=1000, height=800,
    random_state=42,
).generate(" ".join(dropped_duplicates['review_description'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)
plt.axis('off')
plt.savefig('wordcloud_description.png', dpi = 100)
plt.show()



wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=300,
    max_font_size=200, 
    width=1000, height=800,
    random_state=42,
).generate(" ".join(dropped_duplicates['review_title'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - Review Title",fontsize=25)
plt.axis('off')
plt.savefig('wordcloud_title.png', dpi = 100)
plt.show()


from nltk.sentiment.vader import SentimentIntensityAnalyzer

SIA = SentimentIntensityAnalyzer()

# Applying Model, Variable Creation
sentiment = dropped_duplicates
sentiment['polarity_score']=sentiment.review_description.apply(lambda x:SIA.polarity_scores(x)['compound'])
sentiment['neutral_score']=sentiment.review_description.apply(lambda x:SIA.polarity_scores(x)['neu'])
sentiment['negative_score']=sentiment.review_description.apply(lambda x:SIA.polarity_scores(x)['neg'])
sentiment['positive_score']=sentiment.review_description.apply(lambda x:SIA.polarity_scores(x)['pos'])

sentiment['sentiment']= np.nan
sentiment.loc[sentiment.polarity_score>0,'sentiment']='POSITIVE'
sentiment.loc[sentiment.polarity_score==0,'sentiment']='NEUTRAL'
sentiment.loc[sentiment.polarity_score<0,'sentiment']='NEGATIVE'


plt.figure(figsize=(14,5))
ax = sns.boxplot(x='sentiment', y='points', data=sentiment)
ax.set_title("Sentiment by Points Distribution", fontsize=19)
ax.set_ylabel("Points ", fontsize=17)
ax.set_xlabel("Sentiment Label", fontsize=17)
plt.savefig('sentiment vs points.png', dpi = 100)
plt.show()

from sklearn.neighbors import NearestNeighbors # KNN Clustering 
from scipy.sparse import csr_matrix # Compressed Sparse Row matrix
from sklearn.decomposition import TruncatedSVD # Dimensional Reduction

# Lets choice rating of wine is points, title as user_id, and variety,
col = ['province','variety','points']

wine1 = dropped_duplicates[col]
wine1 = wine1.dropna(axis=0)
wine1 = wine1.drop_duplicates(['province','variety'])
wine1 = wine1[wine1['points'] > 85]

wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)
wine_pivot_matrix = csr_matrix(wine_pivot)

knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric= 'cosine')
model_knn = knn.fit(wine_pivot_matrix)

for n in range(5):
    query_index = np.random.choice(wine_pivot.shape[0])
    #print(n, query_index)
    distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors=6)
    for i in range(0, len(distance.flatten())):
        if  i == 0:
            print('Recommendation for ## {0} ##:'.format(wine_pivot.index[query_index]))
        else:
            print('{0}: {1} with distance: {2}'.format(i,wine_pivot.index[indice.flatten()[i]],distance.flatten()[i]))
    print('\n')