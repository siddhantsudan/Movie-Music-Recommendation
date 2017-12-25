import numpy as np
import pandas as pd
import MatrixFactorization


MovieRatingsTrainingDf = pd.read_csv('MovieRatingsTraining.csv')
MovieRatingsTestingDf = pd.read_csv('MovieRatingsTesting.csv')


MovieRatingsTrainingPivotDf = pd.pivot_table(MovieRatingsTrainingDf, index='user_id', columns='movie_id', aggfunc=np.max)
MovieRatingsTestingPivotDf = pd.pivot_table(MovieRatingsTestingDf, index='user_id', columns='movie_id', aggfunc=np.max)




U, M = MatrixFactorization.low_rank_matrix_factorization(MovieRatingsTrainingPivotDf.as_matrix(),
                                                                    num_features=11,
                                                                    regularization_amount=1.1)


PredictedRatings = np.matmul(U, M)


TrainingRMSE = MatrixFactorization.RMSE(MovieRatingsTrainingPivotDf.as_matrix(),
                                                    PredictedRatings)
TestingRMSE = MatrixFactorization.RMSE(MovieRatingsTestingPivotDf.as_matrix(),
                                                   PredictedRatings)

print("Training RMSE: {}".format(TrainingRMSE))
print("Testing RMSE: {}".format(TestingRMSE))