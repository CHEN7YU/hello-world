import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\breast_cancer_qc.csv", sep="," )

df_x = df
df_x = df_x.drop(columns = ['Col31'])
df_y = df['Col31']

model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
model.fit(x_train,y_train)

pred_results = model.predict(x_test)
pred_prob = model.predict_proba(x_test)
pred_prob1 = pred_prob[:,1]


# print some predicted results & prob%
print("Print Probability")
print(pred_results[0])
print(pred_prob[0])
print(pred_prob1[0])
print(pred_results[1])
print(pred_prob[1])
print(pred_prob1[1])
print(pred_results[2])
print(pred_prob[2])
print(pred_prob1[2])


# print(pred_results)
# print(pred_prob)

# calculate accuracy
print("model score")
print(model.score(x_test,y_test))  # score of the model, 1 is perfect

# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, pred_results))

# confusion matrix first argument is true values, second argument is predicted values
print("Confusion Matrix")
print("    0 F", " 1 True predicted")
print(metrics.confusion_matrix(y_test, pred_results))

# True Positives (TP): we correctly predicted that they do have diabetes
# True Negatives (TN): we correctly predicted that they don't have diabetes
# False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
# False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, pred_results)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


print((TP + TN) / float(TP + TN + FP + FN))
print("Classification Accuracy: how often is the classifier correct? ",metrics.accuracy_score(y_test, pred_results))
# other rates / scores 
# https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb
print("Sensitivity: When the actual value is positive, how often is the prediction correct?"
      , metrics.recall_score(y_test, pred_results) )
print("Specificity: When the actual value is negative, how often is the prediction correct?"
     , TN / float(TN + FP))

# ROC curve
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob1)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# print the rate based on Threshold input
print(evaluate_threshold(0.5))
print(evaluate_threshold(0.3))

# AUC score
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, pred_prob1))