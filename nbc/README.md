F1 obtained on the test corpus based on training using only the top-k features per class, for the following values of k. The selection based on Mutual Information.

Reporting the F1 for each class separately, as well as the micro and macro averages for all classes.

|   | top-10 | top-20 | top-40 | top-80 | top-160|
|---|--------|--------|--------|--------|--------|
tech |0.7906976744186046| 0.8837209302325582| 0.9268292682926829| 0.975609756097561|0.9523809523809523 |
business |0.9056603773584906 |0.8979591836734695 | 0.9199999999999999| 0.9600000000000001| 0.9019607843137256|
sport |0.8771929824561403 | 0.8474576271186441| 0.9122807017543859| 0.9285714285714286|0.9615384615384616 |
entertainment |0.9473684210526316 | 0.9189189189189189| 0.9743589743589743| 0.9500000000000001| 0.9|
politics |0.8947368421052632 |0.9 |0.9268292682926829 |0.9268292682926829 |0.9302325581395349 |
|---|--------|--------|--------|--------|--------|
ALL-micro | 0.8771929824561403|0.8859649122807017 | 0.9298245614035088| 0.9473684210526315| 0.9298245614035088|
ALL-macro | 0.8789222609296047| 0.8896113319887181| 0.9320596425397453|0.9482020905923345 | 0.9292225512745349|

---

### Results of Naive Bayes Classifier (NBC) with Top-k Features

The Naive Bayes Classifier (NBC) was trained using only the top-k features per class, selected based on Mutual Information. The F1 score obtained on the test corpus for different values of k is reported below, along with the F1 score for each class separately, as well as the micro and macro averages for all classes.

#### F1 Score for Each Class:
The F1 score for each class (tech, business, sport, entertainment, politics) represents the harmonic mean of precision and recall. It measures the balance between the model's ability to correctly identify positive instances (recall) and its ability to avoid misclassifying negative instances (precision).

#### Micro and Macro Averages:
- Micro Average: The micro-average F1 score is calculated by considering the total true positives, false positives, and false negatives across all classes. It gives equal importance to each data point, regardless of class. This is useful when there is an imbalance in the class distribution.
- Macro Average: The macro-average F1 score is computed by averaging the F1 scores of all classes without considering class imbalances. It treats each class equally, regardless of the number of samples in each class.

#### Interpreting the Results:
- The F1 scores generally improve as the number of top-k features increases for most classes, indicating that adding more informative features can enhance classification performance.
- Some classes may benefit from a higher number of top-k features (e.g., tech and entertainment), while others achieve peak performance with a lower number (e.g., business and sport).
- The micro and macro average F1 scores provide an overall view of the classifier's performance. Both metrics improve with more top-k features, indicating better generalization.

#### Optimal Top-k Selection:
The choice of the optimal top-k value can vary based on the dataset and the specific problem. To determine the best top-k value, it's recommended to perform hyperparameter tuning and cross-validation, trying different top-k values and evaluating the model's performance using appropriate metrics such as F1 score, accuracy, or cross-entropy loss.

It's essential to find the top-k value that maximizes the classifier's accuracy and generalization ability on unseen data. Keep in mind that the optimal top-k value might differ for different datasets, so experimentation and careful evaluation are essential to make an informed decision about the best hyperparameter setting for the NBC in a particular application.

### Further Analysis and Improvements
Based on these results, further analysis can be conducted to fine-tune the classification performance. Some potential steps include:
- Experimenting with different feature selection techniques to identify the most relevant features for each class.
- Trying alternative methods of handling class imbalances, if present in the dataset.
- Performing error analysis to understand misclassifications and identify patterns for potential model improvement.

