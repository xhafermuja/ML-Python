# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Importing the basic libraries
import pandas as pd #to read the matrix csv file
import matplotlib.pyplot as plt #to visualize the data
from sklearn.metrics import classification_report, confusion_matrix #to train the model and test the model
from sklearn.model_selection import train_test_split


# Show the confusion matrix
import seaborn as sns
import matplotlib.ticker as ticker


#Using NB classifier
from sklearn.naive_bayes import GaussianNB


#i kemi rregullu te dhenat
df = pd.read_csv('car_evaluation.csv') 

#converting the data , to ints
d = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
df['Buying'] = df['Buying'].map(d)

df['Maint'] = df['Maint'].map(d)

d = {'2': 2, '3': 3, '4': 4, '5more': 5}
df['Doors'] = df['Doors'].map(d)

d = {'2': 2, '4': 4, 'more': 5}
df['Persons'] = df['Persons'].map(d)

d = {'big': 1, 'med': 2, 'small': 3}
df['LugBoot'] = df['LugBoot'].map(d)

d = {'high': 1, 'med': 2, 'low': 3}
df['Safety'] = df['Safety'].map(d)

d = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
df['Evaluation'] = df['Evaluation'].map(d)

x = df.iloc[:, :-1]
y = df.iloc[:, 6]

features = ['Buying', 'Maint', 'Doors', 'Persons', 'LugBoot', 'Safety']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


clf = GaussianNB()
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
print("Training Accuracy: ",clf.score(x_train, y_train))
print("Testing Accuracy: ", clf.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test,y_pred))




values = [1, 2, 3, 4]
tags = ['unacc', 'acc', 'good', 'vgood']

ax = sns.heatmap(cm, linewidth=0.5, annot=True, fmt="d", cmap="Blues")

plt.title('Confusion Matrix')
plt.xlabel('Real')
plt.ylabel('Predicted')

plt.xticks(values,tags)
plt.yticks(values,tags, rotation = 0)

# Hide major tick labels
ax.xaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_major_formatter(ticker.NullFormatter())

# Customize minor tick labels
ax.xaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5]))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(tags))

ax.yaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5]))
ax.yaxis.set_minor_formatter(ticker.FixedFormatter(tags))

plt.savefig('confusion_matrix_MNB.png')
plt.show()