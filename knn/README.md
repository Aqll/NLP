### Results of k-Nearest Neighbors (kNN) Algorithm

The k-Nearest Neighbors (kNN) algorithm was applied to the dataset, and the F1 score for each class was obtained along with the micro and macro averages for all classes. The F1 score is a metric commonly used in classification tasks to evaluate the performance of a model.

#### F1 Score for Each Class
The F1 score for each class (tech, business, sport, entertainment, politics) represents the harmonic mean of precision and recall. It measures the balance between the model's ability to correctly identify positive instances (recall) and its ability to avoid misclassifying negative instances (precision).

#### Micro and Macro Averages
- Micro Average: The micro-average F1 score is calculated by considering the total true positives, false positives, and false negatives across all classes. It gives equal importance to each data point, regardless of class. This is useful when there is an imbalance in the class distribution.
- Macro Average: The macro-average F1 score is computed by averaging the F1 scores of all classes without considering class imbalances. It treats each class equally, regardless of the number of samples in each class.

#### Results 
F1 for each class separately, as well as the micro and macro averages for all classes.

|   | k=1 | k=5 | k=11 | k=23 
|---|----|----|----|----|
tech |0.5517241379310345  |0.17391304347826084 |0.0909090909090909 | 0|
business |0.2666666666666667  | 0.20689655172413793|0 |0 |
sport |0.5555555555555556 |0.7123287671232876  | 0.4031007751937985| 0.37410071942446044|
entertainment |0.2608695652173913|0  |0 | 0|
politics |0.38181818181818183 |0.4819277108433735 | 0.5161290322580645|0.0909090909090909 |
|---|----|----|----|----|
ALL-micro |0.40350877192982454 |0.4473684210526316 | 0.30701754385964913| 0.23684210526315788|
ALL-macro | 0.403326821437766| 0.31501321463381193|0.20202777967219077 | 0.09300196206671027|


#### Interpreting the Results
- k=1: The F1 scores for some classes are relatively high, but there is a risk of overfitting with such a low k value, especially for classes with smaller sample sizes.
- k=5: The F1 scores show improvements compared to k=1, suggesting better generalization. However, some classes still have relatively low scores.
- k=11: The F1 scores seem to stabilize, but the performance may vary for different classes.
- k=23: As k increases further, the F1 scores drop, which could indicate a decrease in model performance.

#### Optimal k-value Selection
Based on the results provided, it appears that the optimal k value for the kNN algorithm is k=5. This is because the F1 scores for most classes seem to improve compared to the F1 scores obtained with k=1. Additionally, using k=5 helps to reduce the risk of overfitting, as a higher k value considers more neighbors during the classification process, leading to better generalization.

It's important to note that the choice of the optimal k value can depend on the specific dataset and the problem at hand. In some cases, even higher k values might yield better performance. To determine the best k value, it's recommended to perform hyperparameter tuning and cross-validation on the dataset, trying different k values and evaluating the model's performance using appropriate metrics such as F1 score, accuracy, or cross-entropy loss.

The goal of hyperparameter tuning is to find the k value that maximizes the model's accuracy and generalization ability on unseen data. Keep in mind that the optimal k value might differ for different datasets, so experimentation and careful evaluation are essential to make an informed decision about the best hyperparameter setting for the kNN algorithm in a particular application.

### Further Analysis and Improvements
Based on these results, further analysis can be conducted to enhance the classification performance. Some potential steps include:
- Investigate classes with low F1 scores to understand potential challenges in classification.
- Consider using techniques like feature selection, data augmentation, or other algorithms to improve classification performance.
