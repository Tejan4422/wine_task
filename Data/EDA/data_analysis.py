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

def pastel_plot(dropped_duplicates, x, y):
    plt.figure(figsize = (15,6))
    plt.title('Points histogram - whole dataset')
    sns.set_color_codes("pastel")
    sns.barplot(x = x, y=y, data=dropped_duplicates)
    locs, labels = plt.xticks()
    plt.savefig('points_details.png', dpi = 100)
    plt.show()
    
temp = dropped_duplicates["points"].value_counts()
df = pd.DataFrame({'points': temp.index,
                   'number_of_wines': temp.values
                  })

pastel_plot(df,'points', 'number_of_wines')

plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.distplot(dropped_duplicates[dropped_duplicates["price"]<200]['price'])
plt.savefig('price_distribution.png', dpi = 100)

z=dropped_duplicates.groupby(['country'])['price','points'].mean().reset_index().sort_values('price',ascending=False)
z[['country','price']].head(n=10)

plt.figure(figsize = (15,6))
plt.title('Countries with highest prices of wine')
sns.set_color_codes("pastel")
sns.barplot(x = 'country', y= 'price', data=dropped_duplicates)
plt.xticks(rotation = 90)
plt.savefig('price_distribution_barplot.png', dpi = 100)
plt.show()


plt.figure(figsize = (15,6))
plt.title('Provinces with highest prices of wine')
sns.set_color_codes("pastel")
sns.barplot(x = 'province', y= 'price', data=dropped_duplicates.head(10))
plt.xticks(rotation = 90)
plt.savefig('price_distribution_barplot_province.png', dpi = 100)
plt.show()


plt.figure(figsize = (15,6))
plt.title('Countries with highest points of wine')
sns.set_color_codes("pastel")
sns.barplot(x = 'country', y= 'points', data=dropped_duplicates.head(10))
plt.xticks(rotation = 90)
plt.savefig('points_distribution_barplot.png', dpi = 100)
plt.show()


country = dropped_duplicates.country.value_counts()
country.head(10).plot.bar()

z['quality/price']=z['points']/z['price']
z.sort_values('quality/price', ascending=False)[['country','quality/price']]

plt.figure(figsize = (14,6))
sns.boxplot(
        x = 'variety',
        y = 'points',
        data = dropped_duplicates
        )
plt.xticks(rotation = 90)
plt.savefig('pointsvsvariety.png', dpi = 100)
plt.show()