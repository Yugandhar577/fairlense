# Demo datasets

These two small CSV files are **synthetic teaching examples**, not observations of real people or property. Their results must not be used as evidence about real-world fairness or model performance. They contain only 40 rows, so subgroup measurements are deliberately low-confidence.

## Classification

Upload `classification_demo.csv`, choose **Classification**, target `outcome`, and positive class `yes`. Select `region` as sensitive and `experience` and `qualification` as features. Optionally select `age_band` as another sensitive view and enable their intersection. The constructed age bands are strongly associated with the synthetic outcome; this is a teaching device, not a causal explanation.

Choose two models, such as Logistic Regression and Decision Tree Classifier. Optionally enable reweighing or threshold adjustment targeting **region**. Do not target `age_band` with threshold adjustment: this toy dataset does not have both outcomes in every age band.

## Regression

Upload `regression_demo.csv`, choose **Regression**, target `price`, and sensitive column `region`. Use `area` and `rooms` as features. Choose Linear Regression and Ridge Regression. Inspect MAE, RMSE, R², and the group-error table. Values are invented, unitless teaching data—not actual house prices.
