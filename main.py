from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(iris.feature_names)
print(iris.target)
X=iris.data[:,:4]
Y=iris.target
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.10, random_state=42) 

obj=KNeighborsClassifier(n_neighbors=5)
obj.fit(xtrain,ytrain)

#testset=[[1.4,3.6,3.4,1.2], [1.8,3.6,3.9,1.8]]
pred = obj.predict(xtest)
print(pred)
print(obj.kneighbors(xtest,return_distance = False))

from sklearn.metrics import confusion_matrix, classification_report 
cm = confusion_matrix(ytest, pred)
print(cm)
print(classification_report(ytest,pred))

acc_rate =[]
for i in range(1,40):
    Obj = KNeighborsClassifier(n_neighbors = i)
#    score = cross_val_score(Obj,dataset_updated, dataset['TARGET CLASS'], cv=10)
    Obj.fit(xtrain,ytrain)
    pred_i = Obj.predict(xtest)
    acc_rate.append(1-np.mean(pred_i !=ytest))
    
print(acc_rate)

error_rate = []
for i in range(1,40):
    Obj = KNeighborsClassifier(n_neighbors = i)
    Obj.fit(xtrain,ytrain)
    pred_i = Obj.predict(xtest)
    error_rate.append(np.mean(pred_i !=ytest))

plt.figure(figsize=(10,6))
#plt.plot(range(1,40),error_rate,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.plot(range(1,40),error_rate,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
