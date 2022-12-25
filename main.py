import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC  # "Support vector classifier"
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dftrain = pd.read_csv('train.csv')
df_2 = pd.read_csv('test.csv')
dfDataSet = pd.read_csv("Dataset.csv")

dfDataSet.drop_duplicates(inplace=True)

plt.hist(dfDataSet.age)
plt.show()
dfDataSet.head()
dfDataSet.describe()


def plot_boxplot(dtf, ft):
    dtf.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

plot_boxplot(dfDataSet, "capital-loss")
plot_boxplot(df_2, "capital-loss")

def outliers(dtf, ft):

    Q1 = dtf[ft].quantile(0.25)
    Q3 = dtf[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    ls = dtf[ft].index[(dtf[ft] < lower_bound) | (dtf[ft] > upper_bound)]
    return ls


index_list = []
index_list2 = []
for feature in ['age', 'work-fnl', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
    index_list.extend(outliers(dfDataSet, feature))
for feature in ['age', 'work-fnl', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
    index_list2.extend(outliers(df_2, feature))

def remove(dtf, ls):
    ls = sorted(set(ls))
    dtf = dtf.drop(ls)
    return dtf
dfDataSet = remove(dfDataSet, index_list)

dfDataSet['work-class'] = dfDataSet['work-class'].str.replace('?', dfDataSet['work-class'].mode()[0])

dfDataSet['native-country'] = dfDataSet['native-country'].str.replace(' ?', dfDataSet['native-country'].mode()[0])

dfDataSet['marital-status'] = dfDataSet['marital-status'].str.replace('?', dfDataSet['marital-status'].mode()[0])

dfDataSet['relationship'] = dfDataSet['relationship'].str.replace('?', dfDataSet['relationship'].mode()[0])

dfDataSet['race'] = dfDataSet['race'].str.replace('?', dfDataSet['race'].mode()[0])

dfDataSet['sex'] = dfDataSet['sex'].str.replace('?',dfDataSet['sex'].mode()[0])

df_2['work-class'] = df_2['work-class'].str.replace('?', dfDataSet['work-class'].mode()[0])

df_2['native-country'] = df_2['native-country'].str.replace(' ?', dfDataSet['native-country'].mode()[0])

df_2['marital-status'] = df_2['marital-status'].str.replace('?', dfDataSet['marital-status'].mode()[0])

df_2['relationship'] = df_2['relationship'].str.replace('?', dfDataSet['relationship'].mode()[0])

df_2['race'] = df_2['race'].str.replace('?', dfDataSet['race'].mode()[0])

df_2['sex'] = df_2['sex'].str.replace('?', dfDataSet['sex'].mode()[0])


le = LabelEncoder()
le.fit(dfDataSet['salary'])
le.transform(dfDataSet['salary'])
le_2 = dfDataSet['salary']
dfDataSet['salary'] = le.transform(dfDataSet['salary'])
print(dfDataSet[['salary']])

def encode(df, col):
    le.fit(df[col])
    df[col] = le.transform(df[col])
def encode_test(df, col):
    le.fit(df[col])
    df[col] = le.transform(df[col])

for col in ['work-class', 'education', 'marital-status',  'relationship', 'race', 'sex', 'native-country']:
    encode(dfDataSet, col)

for col in ['work-class', 'education', 'marital-status', 'relationship', 'race', 'sex','native-country']:
    encode_test(df_2, col)

# obtain the correlations of each features in dataset
corrmat = dfDataSet.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(12, 12))
# plot heat map
g = sns.heatmap(dfDataSet[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
print(dfDataSet)


from sklearn.model_selection import train_test_split
columns = dfDataSet.iloc[:, 0:13]
target = dfDataSet['salary']
X_train, X_test, y_train, y_test = train_test_split(columns, target, test_size=0.30, random_state=40, shuffle=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print('X train')
print(X_train.shape[1])
X_test = scaler.transform(X_test)
print('X test')
print(X_test.shape[1])
print("logistic regression model")
classifier = LogisticRegression(solver='lbfgs', multi_class="auto", random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#standardization(feature scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print('X train')
print(X_train.shape[1])
X_test = scaler.transform(X_test)
print('X test')
print(X_test.shape[1])

df_2=scaler.transform(df_2)
test_predictions=classifier.predict(df_2)
print(test_predictions)
pred_list=[]
for i,pred in enumerate(test_predictions):
     if pred==0:
         pred_list.append('<=50K')
     elif (pred==1):
         pred_list.append('>50K')

data={'index':range(0,len(pred_list)),'salary':pred_list}
test_df=pd.DataFrame(data)
test_df.to_csv("test_file4.csv",index=False)
print(len(pred_list))

com = confusion_matrix(y_test, y_pred)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                com.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(com, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall positive : {metrics.recall_score(y_test, y_pred, pos_label=1) * 100:.2f}%")
print(f"Recall negative : {metrics.recall_score(y_test, y_pred, pos_label=0) * 100:.2f}%")
print(f"precision postive : {metrics.precision_score(y_test, y_pred, pos_label=1) * 100:.2f}%")
print(f"precision negative : {metrics.precision_score(y_test, y_pred, pos_label=0) * 100:.2f}%")
print(f"The Mean Squared Error : {metrics.mean_squared_error(y_test, y_pred) * 100:.2f}%")
plt.show()
print("desicion tree model")
DecisionTree = DecisionTreeClassifier(max_depth=4, min_samples_split=2)
DecisionTree = DecisionTree.fit(X_train, y_train)
y_pred_dt: object = DecisionTree.predict(X_test)
print("-----------------------------------------------")


com = confusion_matrix(y_test, y_pred_dt)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                com.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(com, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred_dt) * 100:.2f}%")
print(f"Recall positive : {metrics.recall_score(y_test, y_pred_dt, pos_label=1) * 100:.2f}%")
print(f"Recall negative : {metrics.recall_score(y_test, y_pred_dt, pos_label=0) * 100:.2f}%")
print(f"precision postive : {metrics.precision_score(y_test, y_pred_dt, pos_label=1) * 100:.2f}%")
print(f"precision negative : {metrics.precision_score(y_test, y_pred_dt, pos_label=0) * 100:.2f}%")
print(f"The Mean Squared Error : {metrics.mean_squared_error(y_test, y_pred_dt) * 100:.2f}%")
plt.show()
print("random forest model")
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=12, random_state=42)
randomforest.fit(X_train, y_train)
y_pred_rf = randomforest.predict(X_test)

com = confusion_matrix(y_test, y_pred_rf)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in com.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(com, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(f"Recall positive : {metrics.recall_score(y_test, y_pred_rf, pos_label=1) * 100:.2f}%")
print(f"Recall negative : {metrics.recall_score(y_test, y_pred_rf, pos_label=0) * 100:.2f}%")
print(f"precision postive : {metrics.precision_score(y_test, y_pred_rf, pos_label=1) * 100:.2f}%")
print(f"precision negative : {metrics.precision_score(y_test, y_pred_rf, pos_label=0) * 100:.2f}%")
print(f"The Mean Squared Error : {metrics.mean_squared_error(y_test, y_pred_rf) * 100:.2f}%")
plt.show()

print("SVM model")
svm = SVC(C=1, degree=3, kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

com = confusion_matrix(y_test, y_pred_svm)
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                com.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(com, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print(f"Recall positive : {metrics.recall_score(y_test, y_pred_svm, pos_label=1) * 100:.2f}%")
print(f"Recall negative : {metrics.recall_score(y_test, y_pred_svm, pos_label=0) * 100:.2f}%")
print(f"precision postive : {metrics.precision_score(y_test, y_pred_svm, pos_label=1) * 100:.2f}%")
print(f"precision negative : {metrics.precision_score(y_test, y_pred_svm, pos_label=0) * 100:.2f}%")
print(f"The Mean Squared Error : {metrics.mean_squared_error(y_test, y_pred_svm) * 100:.2f}%")
plt.show()