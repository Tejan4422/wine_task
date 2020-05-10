# wine_task
## Overview
* Tasks achieved
    * Exploratory data analysis on training dataset 
    * NLP applied on customer reviews
    * Prediction of variety of wine
## Data visualization
Remeber that in this stage our goal is not only to explore our data in order to get better predictions. 
We also want to get better understanding what is in data and explore data in 'normal' way. 
This kind of approch can be useful if we have to do some feature engineering, where good data understanding can really 
help to produce better features.

![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/price_distribution_barplot.png "Price Distribution")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/price_distribution_barplot_province.png "Price Distribution in Province")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/points_by_price.png "Points by countires")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/points_distribution_barplot.png "Points by countires")
Top Countries in terms of prices of wine in Descending order
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/top_countrieswith%20price.png "Price by countires")
Following box plot shows varieties against its points which helps us to derive which varieties secure top points
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/EDA/pointsvsvariety.png "Points by variety")

## Customer review analysis along with Sentiment analysis using NLP
  * Before heading towards NLP part lets try to discover some insights from dataset
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/Top%2020%20wineries.png "Top wineries")
Lengths of customer reviews
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/description%20length%20vs%20points.png "Description length")
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/sentiment%20vs%20points.png "Sentiments")
Wine recommender
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wine_recommended.png "Recommendations")
WordCloud of customer reviews
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wordcloud_description.png "wordcloud")
wordcloud of titles
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/wordcloud_title.png "wordcloud")
  
  * TextBlod is used to get the customer sentiments from their reviews
  In this file customer sentiments can be seen in the last column ** https://github.com/Tejan4422/wine_task/blob/master/Data/review_analysis/review_sentiment_analysis.csv

## Variety Prediction
  * Algorithms used
    * Random Forest Classifier
    * MLP Classifier
 * Following data shows accuracy achieved in training of these algorithms
 Random Forest Classifier withour Hyperparamete tuning was able to record accuracy of close to 50%
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/variety_predictoin/rf_not_tuned.png "rf")
While after hyperparametr tuning Random Forest classifier achived accuracy of close to 53%
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/variety_predictoin/rf_tuned.png "rf")
Neural Network withour hyper parameter tuning was able to record accuracy of 52%. Unfortunately I could not train this model 
with hyperparameter tuning because of long waitings in training but with training accuracy close to 60% can be achieved with this
neural network
![alt text](https://github.com/Tejan4422/wine_task/blob/master/Data/variety_predictoin/nn_not_tuned.png "nn")

## Platform
  * spyder 
  * anaconda 
  * python version 3.6
## Packages used
  * Pandas
  * matplotlib
  * seaborn
  * sklearn
  * textblob
  * regular expression






