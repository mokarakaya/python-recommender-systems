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

random_model
fit duration: 0:00:00.000005
evaluate duration: 0:00:00.039267
{'precision': 0.016666666666666666, 'recall': 0.023809523809523808, 'coverage': 59}

popularity_model
fit duration: 0:00:00.000135
evaluate duration: 0:00:00.063535
{'precision': 0.15555555555555556, 'recall': 0.17614379084967322, 'coverage': 17}

average_rating_model
fit duration: 0:00:00.000504
evaluate duration: 0:00:00.044474
{'precision': 0.05714285714285715, 'recall': 0.018059129810281883, 'coverage': 15}

knncf
fit duration: 0:00:00.059505
evaluate duration: 0:00:01.796871
{'precision': 0.3032908704883227, 'recall': 0.19422919775972713, 'coverage': 234}

svd
fit duration: 0:00:00.040252
evaluate duration: 0:00:00.323015
{'precision': 0.2873673036093418, 'recall': 0.1972163950171332, 'coverage': 498}

LightFM
fit duration: 0:00:02.520957
evaluate duration: 0:00:00.515876
{'precision': 0.333864118895966, 'recall': 0.21542940620606985, 'coverage': 431}

knncbf
{'precision': 0.011111111111111112, 'recall': 0.012345679012345678, 'coverage': 54}
duration: 0:00:00.075490

knncfknncbf
{'precision': 0.13, 'recall': 0.14866946778711482, 'coverage': 70}
duration: 0:00:00.165391

```

### The Movies Sample
- https://www.kaggle.com/rounakbanik/the-movies-dataset
- ratings: 100,004 
- users: 671 
- items: 9,066 

- Test Parameters:
    - `parallel=False` 
    - `test_percentage=1.0`
    - `lightFm epochs=150`

```

random_model
fit duration: 0:00:00.000007
evaluate duration: 0:00:00.115494
{'precision': 0.0, 'recall': 0.0, 'coverage': 50}

popularity_model
fit duration: 0:00:00.000231
evaluate duration: 0:00:00.042300
{'precision': 0.13999999999999999, 'recall': 0.030736714975845413, 'coverage': 20}

average_rating_model
fit duration: 0:00:00.000599
evaluate duration: 0:00:00.075318
{'precision': 0.030000000000000006, 'recall': 0.009854700854700854, 'coverage': 16}

knncf
fit duration: 0:00:00.375383
evaluate duration: 0:00:07.853654
{'precision': 0.13805970149253732, 'recall': 0.09739439943770821, 'coverage': 1049}

svd
fit duration: 0:00:00.136456
evaluate duration: 0:00:04.259677
{'precision': 0.2580597014925373, 'recall': 0.14664276460700815, 'coverage': 575}

LightFM
fit duration: 0:00:12.044977
evaluate duration: 0:00:16.021321
{'precision': 0.18895522388059705, 'recall': 0.09311237907594277, 'coverage': 435}

tfidfcbf
{'precision': 0.028571428571428574, 'recall': 0.015873015873015872, 'coverage': 54}
duration: 0:00:01.002759


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

svd
fit duration: 0:00:12.370896
evaluate duration: 0:01:00.792134
{'precision': 0.20692068429237948, 'recall': 0.17086245159235441, 'coverage': 677}

LightFM
fit duration: 0:24:21.522301
evaluate duration: 0:01:06.589433
{'precision': 0.1636644762272903, 'recall': 0.10063789514006921, 'coverage': 618}

```