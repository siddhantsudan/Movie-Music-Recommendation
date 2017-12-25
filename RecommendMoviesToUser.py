import numpy as np
import pandas as pd
import MatrixFactorization


MovieRatingsDf = pd.read_csv('MovieRatings.csv')


MoviesDf = pd.read_csv('movies.csv', index_col='movie_id')


RatingsDf = pd.pivot_table(MovieRatingsDf, index='user_id',
                            columns='movie_id',
                            aggfunc=np.max)


U, M = MatrixFactorization.low_rank_matrix_factorization(RatingsDf.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)


PredictedRatings = np.matmul(U, M)

print("Enter a UserID between 1 and 100 to get recommendations:")
UserId = int(input())

print("Movies Seen by user_id {}:".format(UserId))

MoviesSeen = MovieRatingsDf[MovieRatingsDf['user_id'] == UserId]

MoviesSeen = MoviesSeen.join(MoviesDf, on='movie_id')



print(MoviesSeen[['title', 'genre', 'value']])

input("Press enter to continue.")

print("Recommended Movies:")



MoviesDf['rating'] = PredictedRatings[UserId - 1]


MoviesWatched = MoviesSeen['movie_id']
MoviesToWatch = MoviesDf[MoviesDf.index.isin(MoviesWatched) == False]
MoviesToWatch = MoviesToWatch.sort_values(by=['rating'], ascending=False)

print(MoviesToWatch[['title', 'genre', 'rating']].head(5))



