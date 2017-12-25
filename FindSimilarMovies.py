import numpy as np
import pandas as pd
import MatrixFactorization


MovieRatingsDf = pd.read_csv('MovieRatings.csv')


MoviesDf = pd.read_csv('movies.csv', index_col='movie_id')




MovieRatingsPivotDf = pd.pivot_table(MovieRatingsDf, index='user_id', columns='movie_id', aggfunc=np.max)




U, M = MatrixFactorization.low_rank_matrix_factorization(MovieRatingsPivotDf.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=1.0)





M = np.transpose(M)


print("Enter MovieID between 1 and 34")
movieId= int(input())

MovieInformation = MoviesDf.loc[movieId]

print("Finding movies similar to the following movie:")
print("Movie title: {}".format(MovieInformation.title))
print("Genre: {}".format(MovieInformation.genre))


MovieFeatures = M[movieId - 1]

print("Movie attributes are:")
print(MovieFeatures)


difference = M - MovieFeatures

AbsoluteDifference = np.abs(difference)


TotalDifference = np.sum(AbsoluteDifference, axis=1)


MoviesDf['diff'] = TotalDifference




sorted_movie_list = MoviesDf.sort_values('diff')


print("Similar movies are:")
print(sorted_movie_list[['title', 'diff']][0:5])

