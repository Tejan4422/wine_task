from textblob import TextBlob
import pandas as pd

df = pd.read_csv('train.csv')

dataset = df.iloc[0:1000, :]



def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None

df['sentiment'] = df['review_description'].apply(sentiment_calc)
df.to_csv('review_sentiment_analysis.csv')