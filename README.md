# python-recommender-systems
Benchmarking of recommender systems implemented with Python

## Collaborative Filtering

### knncf - Similarity-based Collaborative Filtering

- Similarity based collaborative filtering with cosine similarity.

### LightFM

- LightFM with CF data set.
- Original project: https://github.com/lyst/lightfm

## Content Based Filtering

### knncbf - Similarity-based Content-based Filtering

- Similarity based content-based filtering with cosine similarity.

## Hybrid Methods 

### knncfknncbf

- Combining `knncf`, and `knncbf` with bagging. 

## Baselines

### Random Model

- Random item recommendation

### Popularity Model

- Recommends most popular items in data set.

### Average Rating Model

- Recommends items with highest average rating.

## Evaluation Scores

### Movielens 100K

- https://grouplens.org/datasets/movielens/100k/
- ratings: 100,000 
- users: 1,000 
- items: 1,700 

```
knncf
{'precision': 0.3026511134676564, 'recall': 0.1973564695756637, 'coverage': 199}
0:00:02.954148

LightFM
{'precision': 0.32958642629904567, 'recall': 0.21420817457437155, 'coverage': 379}
0:00:01.323856

knncbf
{'precision': 0.02617021276595745, 'recall': 0.013221324710552727, 'coverage': 782}
0:00:02.073633

knncfknncbf
{'precision': 0.19830148619957538, 'recall': 0.11838370123290384, 'coverage': 365}
duration: 0:00:04.430598

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

### The Movies Sample
- https://www.kaggle.com/rounakbanik/the-movies-dataset
- ratings: 100,004 
- users: 671 
- items: 9,066 

- Test Parameters:
    - `parallel=False` 
    - `test_percentage=1.0`
    - `lightFm epochs=50`

```

random_model
fit duration: 0:00:00.000005
evaluate duration: 0:00:09.505676
{'precision': 0.0, 'recall': 0.0, 'coverage': 6560}

popularity_model
fit duration: 0:00:00.000418
evaluate duration: 0:00:02.757903
{'precision': 0.1444776119402985, 'recall': 0.0657327267424566, 'coverage': 50}

average_rating_model
fit duration: 0:00:00.001064
evaluate duration: 0:00:02.060017
{'precision': 0.058805970149253727, 'recall': 0.026759741755783988, 'coverage': 41}

knncf
fit duration: 0:00:00.311265
evaluate duration: 0:00:07.023510
{'precision': 0.14328358208955225, 'recall': 0.10561806997291268, 'coverage': 1029}

LightFM
fit duration: 0:00:03.881544
evaluate duration: 0:00:15.556031
{'precision': 0.17686567164179104, 'recall': 0.0854668048826362, 'coverage': 287}



```

### The Movies 
- https://www.kaggle.com/rounakbanik/the-movies-dataset
- ratings: 26M 
- users: 270,000 
- items: 45,000 

- Test Parameters:
    - `test_percentage=0.01`
    - `lightFm epochs=50`

```

random_model
fit duration: 0:00:00.000006
evaluate duration: 0:00:46.573553
{'precision': 8.025682182985554e-05, 'recall': 4.0514261019878995e-05, 'coverage': 23288}

popularity_model
fit duration: 0:00:00.057120
evaluate duration: 0:00:22.914063
{'precision': 0.10383830134748878, 'recall': 0.07657024326573882, 'coverage': 53}

average_rating_model
fit duration: 0:00:00.105975
evaluate duration: 0:00:22.557454
{'precision': 0.0334805299076676, 'recall': 0.024321722041955977, 'coverage': 28}

knncf
fit duration: 0:00:39.932602
evaluate duration: 0:01:12.308072
{'precision': 0.17564205457463886, 'recall': 0.14418837569962858, 'coverage': 980}


LightFM
fit duration: 0:24:21.522301
evaluate duration: 0:01:06.589433
{'precision': 0.1636644762272903, 'recall': 0.10063789514006921, 'coverage': 618}

```