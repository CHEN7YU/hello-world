import json
import pandas as pd

file_name = r'C:\Users\Gebruiker\Desktop\Data\Books_small.json'

reviews = []

with open(file_name) as f:
    for line in f: 
        review = json.loads(line)
        reviews.append((review['reviewText'],review['overall']))

print(type(reviews))
print(reviews[5])

def flag_df(df):
    if(df['overall']>4):
        return 'Excellent'
    elif (df['overall']>3):
        return 'Very Good'
    elif (df['overall']==3):
        return 'soso'
    else :
        return 'bad'

reviews = pd.DataFrame(reviews)
reviews.columns = ['reviewtext','overall']  # add column name

reviews['Sentiment'] = reviews.apply(flag_df, axis = 1)

print(reviews.head(15))        