# python-recommender-systems
Benchmarking of recommender systems implemented with Python

## Collaborative Filtering

### knncf - Similarity-based Collaborative Filtering

- Similarity based collaborative filtering with cosine similarity.

### LightFM

- LightFM with CF data set.
- Original project: https://github.com/lyst/lightfm

### Evaluation Scores

```
knncf
{'precision': 0.3026511134676564, 'recall': 0.1973564695756637, 'coverage': 199}
0:00:02.954148

LightFM
{'precision': 0.32958642629904567, 'recall': 0.21420817457437155, 'coverage': 379}
0:00:01.323856

```

## Content Based Filtering

### knncbf - Similarity-based Content-based Filtering

- Similarity based content-based filtering with cosine similarity.

### Evaluation Scores

 
```

knncbf
{'precision': 0.02617021276595745, 'recall': 0.013221324710552727, 'coverage': 782}
0:00:02.073633
```

## Hybrid Methods 

### knncfknncbf

- Combining `knncf`, and `knncbf` with bagging. 

### Evaluation Scores


```
knncfknncbf
{'precision': 0.19830148619957538, 'recall': 0.11838370123290384, 'coverage': 365}
duration: 0:00:04.430598

```

## Baseline

### Random Model

- Random item recommendation

### Popularity Model

- Recommends most popular items in data set.

### Average Rating Model

- Recommends items with highest average rating.

### Evaluation Scores


```
random_model
{'precision': 0.013723404255319151, 'recall': 0.005686367271034278, 'coverage': 1671}
0:00:00.242194

popularity_model
{'precision': 0.1945744680851064, 'recall': 0.11579070561413977, 'coverage': 47}
0:00:00.435621

average_rating_model
{'precision': 0.08021276595744681, 'recall': 0.0371801445252744, 'coverage': 35}
0:00:00.216066
```

## Data sets

### Movielens 100K

- https://grouplens.org/datasets/movielens/100k/