import pandas as pd
import numpy as np
from collections import Counter

# parse tags from string format to list.
def parse_tags(tag_string):
    if pd.isna(tag_string):
        return []
    # slit by comma because tags are comma-separated
    return [tag.strip().lower() for tag in str(tag_string).split(',')]

# calculate Jaccard similarity between two tag sets or intersection over union
def jaccard_similarity(tags1, tags2):
    set1 = set(tags1)
    set2 = set(tags2)

    if len(set1.union(set2)) == 0:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union

#  calculate weighted similarity considering tag importance and tags that are rarer
#  in the corpus get higher weights
def weighted_tag_similarity(tags1, tags2, tag_weights=None):
    set1 = set(tags1)
    set2 = set(tags2)

    common_tags = set1.intersection(set2)
    all_tags = set1.union(set2)

    if len(all_tags) == 0:
        return 0.0

    if tag_weights is None:
        # simple Jaccard without weights
        return len(common_tags) / len(all_tags)

    # weighted similarity
    weighted_intersection = sum(tag_weights.get(tag, 1.0) for tag in common_tags)
    weighted_union = sum(tag_weights.get(tag, 1.0) for tag in all_tags)

    return weighted_intersection / weighted_union

# calculate cosine similarity using tag vectors.
def cosine_tag_similarity(tags1, tags2, all_unique_tags):

    # create binary vectors
    vec1 = np.array([1 if tag in tags1 else 0 for tag in all_unique_tags])
    vec2 = np.array([1 if tag in tags2 else 0 for tag in all_unique_tags])

    # calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ============= Complete Implementation =============

# load dataset
df = pd.read_csv('news2.csv')

# parse all tags
df['parsed_tags'] = df['tags'].apply(parse_tags)

# calculate tag weights (inverse document frequency for tags)
all_tags = []
for tags in df['parsed_tags']:
    all_tags.extend(tags)

tag_counter = Counter(all_tags)
total_docs = len(df)

# calculate IDF-like weights for tags (rarer tags get higher weight)
tag_weights = {}
for tag, count in tag_counter.items():
    tag_weights[tag] = np.log(total_docs / (count + 1))

# get all unique tags for cosine similarity
all_unique_tags = list(tag_counter.keys())

# ============= Apply to Pseudocode with Tags =============

print("=" * 50)
print("Tag-Based Similarity for Food & Drink")
print("=" * 50)

food_drink_indices = df[df['article_section'] == 'Food & Drink'].index.tolist()
num_articles_food_and_drink = len(food_drink_indices)

print(f"Total articles in 'Food & Drink': {num_articles_food_and_drink}")

# method 1: Jaccard Similarity
total_goods_jaccard = 0

for idx in food_drink_indices:
    tags_article = df.iloc[idx]['parsed_tags']

    # calculate similarities with all documents
    similarities_list = []
    for other_idx in range(len(df)):
        if other_idx == idx:
            similarities_list.append((other_idx, -1))  # skip self
        else:
            tags_other = df.iloc[other_idx]['parsed_tags']
            sim = jaccard_similarity(tags_article, tags_other)
            similarities_list.append((other_idx, sim))

    # sort and get top-10
    similarities_list.sort(key=lambda x: -x[1])
    top_10 = [doc_idx for doc_idx, score in similarities_list[:10]]

    # count matches
    goods = sum(1 for doc_idx in top_10 if df.iloc[doc_idx]['article_section'] == 'Food & Drink')
    total_goods_jaccard += goods

ratio_quality_jaccard = total_goods_jaccard / (num_articles_food_and_drink * 10)
print(f"Ratio quality (Jaccard Tag Similarity): {ratio_quality_jaccard}")

# method 2: Weighted Tag Similarity
total_goods_weighted = 0

for idx in food_drink_indices:
    tags_article = df.iloc[idx]['parsed_tags']

    similarities_list = []
    for other_idx in range(len(df)):
        if other_idx == idx:
            similarities_list.append((other_idx, -1))
        else:
            tags_other = df.iloc[other_idx]['parsed_tags']
            sim = weighted_tag_similarity(tags_article, tags_other, tag_weights)
            similarities_list.append((other_idx, sim))

    similarities_list.sort(key=lambda x: -x[1])
    top_10 = [doc_idx for doc_idx, score in similarities_list[:10]]

    goods = sum(1 for doc_idx in top_10 if df.iloc[doc_idx]['article_section'] == 'Food & Drink')
    total_goods_weighted += goods

ratio_quality_weighted = total_goods_weighted / (num_articles_food_and_drink * 10)
print(f"Ratio quality (Weighted Tag Similarity): {ratio_quality_weighted}")
