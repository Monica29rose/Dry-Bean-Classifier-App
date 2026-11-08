## a. Problem Statement
The problem is about differentiation among various types of dry beans. In this problem, there are seven different varieties of dry beans (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) 
which look quite similar. Sorting them into different categories takes time, lacks consistency and is prone to errors. 

So this project is about prediction of a variety of bean from a set of 16 measurements of shape and size extracted from a single grain image by a computer-vision system.
Hence, it is a **7-Class** Classification problem  with some imbalance in the classes. The largest class is 6.8 times the size of the smallest.

There is no single variety which is "positive", and all misclassification errors are equally costly. So the primary metric is **MCC** (multi-class Matthews correlation),
which gives a reliable score under class imbalance; macro-F1 is the secondary metric. Five classifiers are trained, tuned and compared, and exposed in a Streamlit app.


## b. Dataset description
| Item | Value |
|---|---|
| Source | Dry Bean Dataset, UCI Machine Learning Repository (Koklu and Ozkan, 2020, *Computers and Electronics in Agriculture* 174:105507, doi:10.1016/j.compag.2020.105507) |
| Instances | 13,611 raw; **13,543** after removing 68 exact duplicate rows (all 68 were Horoz beans) (minimum required: 500) |
| Features | **16** numeric features, no categorical (minimum required: 12): Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1-4 (column names kept exactly as in the source file) |
| Target | `Class`, 7 classes. Distribution after cleaning: Dermason 3,546 (26.2%), Sira 2,636 (19.5%), Seker 2,027 (15.0%), Horoz 1,860 (13.7%), Cali 1,630 (12.0%), Barbunya 1,322 (9.8%), Bombay 522 (3.9%) |
| Positive class | not applicable (multi-class: metrics are macro-averaged) |
| Missing values | none |
| Other properties | strong collinearity (15 feature pairs with abs(r) > 0.95, e.g. Area/ConvexArea r = 0.9999); right-skewed size features (Area skew 2.95) |

![class distribution](figures/class_distribution.png) 


**Pre-processing and protocol.** Duplicates removed; all features standardised (z-score) inside each
scikit-learn pipeline, fitted on training data only (no leakage); stratified 80/20 split with
`random_state=42` -> 10,834 training rows and 2,709 test rows. Environment of the graded run (BITS Virtual Lab): Python 3.12, scikit-learn 1.9.1, numpy 2.5.4, pandas 2.2.3. The test rows are saved as `test_data.csv`
and were never used for fitting or tuning. Hyper-parameters were tuned by 5-fold stratified
cross-validation on the training rows, scoring = MCC (grids and every CV score are in
`model/cv_results.csv`). **Metric definitions:** Accuracy and MCC as usual; Precision, Recall and F1 are
**macro-averaged** over the 7 classes; AUC is one-vs-rest, macro-averaged, from `predict_proba`
(SVM uses Platt scaling, `probability=True`).

## c. GitHub Repository Link
https://github.com/Monica29rose/Dry-Bean-Classifier-App

Live Streamlit app: https://dry-bean-classifier-app-rose.streamlit.app/
```
project-folder/
|-- app.py                 Streamlit app (loads saved models; never retrains)
|-- eval_utils.py          metric code shared by training script and app
|-- train_models.py        tunes + trains the 5 models, writes model/ and test_data.csv
|-- prepare_data.py        one-off xlsx -> csv conversion
|-- requirements.txt
|-- README.md
|-- test_data.csv          hold-out test set (2,709 rows: 16 features + Class)
|-- data/Dry_Bean_Dataset.csv
|-- model/                 5 x .joblib pipelines, metadata.json, results_test.csv, cv_results.csv
|-- analysis/              eda.py, analyze_results.py (evidence used for the observations)
|-- figures/
```
Run: `pip install -r requirements.txt`; `python train_models.py --data data/Dry_Bean_Dataset.csv --target Class`;
`streamlit run app.py`.

## d. Models used
Evaluation on the hold-out test set (2,709 beans):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9948 | 0.9309 | 0.9304 | 0.9305 | 0.9023 |
| Decision Tree | 0.9085 | 0.9774 | 0.9203 | 0.9163 | 0.9181 | 0.8893 |
| KNN | 0.9177 | 0.9918 | 0.9320 | 0.9277 | 0.9294 | 0.9005 |
| Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | 0.9236 | 0.9944 | 0.9356 | 0.9340 | 0.9347 | 0.9076 |

Tuned hyper-parameters (5-fold CV, MCC): Logistic Regression C = 1000; Decision Tree entropy, max_depth = 8,
min_samples_leaf = 5; KNN k = 25, distance-weighted; Naive Bayes var_smoothing = 1e-14 (all ten values tie);
SVM RBF, C = 10, gamma = "scale". A majority-class predictor would score accuracy 0.262 and MCC 0.

**Known limitation.** scikit-learn 1.9 deprecates `SVC(probability=True)` (removal planned for 1.11). The project therefore pins scikit-learn to the exact training version. 
A future-proof alternative is `CalibratedClassifierCV(SVC(), ensemble=False)`.


### Observations on the performance of each model
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The model showed an accuracy of 0.9192, an MCC of 0.9023 and the highest AUC (0.9948) across all five classifiers. This shows that the size and shape features give sufficient information to differentiate most bean varieties by using a linear decision boundary. The training-test accuracy gap is small, 0.008, which suggests that generalisation is good.<br><br>Cross-validation demonstrated that MCC values increased from 0.9001 at C = 0.01 to 0.9104 at C = 1000, but remained almost stable beyond C = 1 (0.9085–0.9104). As this variation is small compared with the fold standard deviation of approximately 0.005, the result at C = 1000 shows a performance plateau instead of a substantial improvement. Although 15 pairs of correlated features may influence coefficient interpretation, predictive performance remains strong. With a storage requirement of approximately 3 KB, Logistic Regression is also the smallest model. |
| Decision Tree | The Decision Tree model shows an accuracy of 0.9085 and an MCC of 0.8893, which makes it the second-weakest of the five classifiers (only Naive Bayes scored lower). It also achieved the lowest AUC (0.9774). The depth-8 tree, with 145 leaves, uses axis-parallel splits, which may not capture the diagonal class boundaries associated with correlated shape features effectively. Its leaf-based probability estimates are also relatively coarse, which influences the AUC. The training–test accuracy gap of 0.030 shows some overfitting despite the depth restriction. The model also showed the lowest Sira F1-score of all five models (0.851) and a relatively weak score for Barbunya (0.865). Its performance was significantly poorer than SVM according to the McNemar test (p = 0.0004). However, its main advantage is interpretability, as the decision rules can be examined and explained. |
| KNN | The performance of the KNN model was close to Logistic Regression, with an accuracy of 0.9177 and an MCC of 0.9005. Standardisation was important because the features have different scales; without it, larger values such as Area (approximately 5 × 10⁴) would dominate the distance calculations. The selected value of k = 25 with distance weighting provided the best result, while higher values of k (41 and 61) did not enhance performance. Although training accuracy was 1.0, this is an artefact of distance weighting, as each training observation is its own nearest neighbour. Therefore, the CV MCC of 0.9099 gives a more reliable estimate and is consistent with the test MCC. However, KNN requires storing the complete training dataset (approximately 1.3 MB) and calculating distances during prediction, making it more computationally expensive as the dataset increases. |
| Naive Bayes | The Naive Bayes model provided the lowest accuracy (0.897) and MCC (0.8762) of all five classifiers. Its performance is likely influenced by the conditional-independence assumption, as 15 feature pairs have absolute correlations above 0.95, including Area and ConvexArea (r = 0.9999). This may result in correlated information being counted repeatedly. The model also made the highest number of Dermason-to-Sira errors (78) and recorded the lowest Barbunya F1-score (0.832). Despite all of this, its AUC of 0.9911 shows good class ranking, although its probability estimates may be poorly calibrated. The negligible training–test gap suggests high bias rather than overfitting. Further, var_smoothing had no influence on CV MCC (0.8756 across all ten values), and Naive Bayes was the fastest model to train. |
| Support Vector Machine | The Support Vector Machine (SVM) recorded the best overall performance, with the highest accuracy (0.9236), precision, recall, F1-score (0.9347) and MCC (0.9076). It also achieved the highest cross-validation MCC (0.9186). Its AUC of 0.9944 was marginally below Logistic Regression (0.9948). The RBF kernel helps capture the non-linear overlap between neighbouring bean varieties, with C = 10 and gamma = "scale" showing the best parameter combination. However, SVM required the longest tuning and fitting time (approximately 50 seconds, including probability calibration using Platt scaling) and produced a larger model file (264 KB). |
| Overall, Winner for your dataset? | For the Dry Bean dataset, **Support Vector Machine** is the preferred model, with the highest MCC (0.9076), accuracy (0.9236), macro-F1 (0.9347) and CV MCC (0.9186). Its advantage over Logistic Regression (MCC 0.9023) and KNN (0.9005) is relatively small. The McNemar test confirms that SVM performs significantly better than Decision Tree (p = 0.0004) and Naive Bayes (p = 2e-7), but not significantly better than Logistic Regression (p = 0.207) or KNN (p = 0.101). This is also reflected in the overlapping 95% bootstrap MCC intervals for the top three models. Therefore, SVM is the overall winner, but Logistic Regression remains a practical alternative when simplicity, speed and smaller model size are priorities.<br><br>Across the models, the most frequent confusion is between Dermason and Sira, followed by Barbunya and Cali. Sira has the lowest F1-score in four models (0.851–0.872), while Barbunya has the lowest score for Naive Bayes (0.832). In contrast, all five models classify Bombay perfectly (F1 = 1.000), possibly because its mean area (173,485 px) is substantially higher than that of the next-largest variety, Cali (75,538 px). |

