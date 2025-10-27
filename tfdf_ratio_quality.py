import datetime

from gensim import corpora
from gensim import models
from pprint import pprint  # pretty-printer
from gensim import similarities

import re

from nltk.corpus import stopwords
from nltk import PorterStemmer

import pandas as pd

# initialize timer for measuring execution time
init_t = datetime.datetime.now()

# load the dataset
df = pd.read_csv('news2.csv')

# initialize preprocessing tools
porter = PorterStemmer()
stoplist = stopwords.words('english')

# remove common words, tokenize and take just description
texts = [
    [porter.stem(word) for word in str(doc).lower().split() if word not in stoplist]
    for doc in df['description']
]

print("Tokens of each document:")
print(texts)

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]

# create tfidf model
tfidf = models.TfidfModel(model_bow)
tfidf_vectors = tfidf[model_bow]

id2token = dict(dictionary.items())


def convert(match):
    return dictionary.id2token[int(match.group(0)[0:-1])]

print()
print("Vectors for documents (the positions with zeros are not shown):")
for doc in tfidf_vectors:
    print(re.sub("[0-9]+,", convert, str(doc)))

matrix_tfidf = similarities.MatrixSimilarity(tfidf_vectors)

end_creation_model_t: datetime = datetime.datetime.now()

print()
print("Matrix similarities")
print(matrix_tfidf)

# implementation of pseudocode

# filter articles by topic "Food and Drink"
food_drink_indices = df[df['article_section'] == 'Food & Drink'].index.tolist()
num_articles_food_and_drink = len(food_drink_indices)

print(f"Total articles in 'Food & Drink': {num_articles_food_and_drink}")

total_goods = 0

# for every article in "Food and Drink" topic
for idx in food_drink_indices:
    # get TF-IDF vector for current article
    vec_tfidf = tfidf_vectors[idx]

    # calculate similarities with all documents
    similarities = matrix_tfidf[vec_tfidf]

    # sort similarities in descending order and get top-10
    similarities_sorted_descending = sorted(enumerate(similarities), key=lambda item: -item[1])

    # get top-10 most similar articles (excluding the article itself)
    top_10 = [doc_idx for doc_idx, score in similarities_sorted_descending[1:11]]

    # count how many of top-10 are also "Food and Drink"
    goods = sum(1 for doc_idx in top_10 if df.iloc[doc_idx]['article_section'] == 'Food & Drink')

    total_goods += goods

# calculate ratio quality
ratio_quality = total_goods / (num_articles_food_and_drink * 10)

print(f"Total goods: {total_goods}")
print(f"Ratio quality: {ratio_quality}")

end_t: datetime = datetime.datetime.now()

# get execution time
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_comparison: datetime = end_t - end_creation_model_t

print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_comparison, 'seconds')
