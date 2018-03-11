# Import python modules
import numpy as np
from sklearn.metrics import accuracy_score
import scipy
import matplotlib.pyplot as plt
import time
import kaggle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold


# Read in train and test tumor data
def read_tumor_data():
    print('Reading tumor data ...')
    train_x = np.loadtxt('../../Data/Tumor/data_train.txt', delimiter=',', dtype=float)
    train_y = np.loadtxt('../../Data/Tumor/label_train.txt', delimiter=',', dtype=float)
    test_x = np.loadtxt('../../Data/Tumor/data_test.txt', delimiter=',', dtype=float)

    return (train_x, train_y, test_x)


############################################################################
# Compute MSE
def compute_MSE(y, y_hat):
    # mean squared error
    return np.mean(np.power(y - y_hat, 2))


train_x, train_y, test_x = read_tumor_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# # Output file location
# file_name = '../Predictions/Tumor/best.csv'
# # Writing output in Kaggle format
# print('Writing output to ', file_name)
# kaggle.kaggleize(predicted_y, file_name, False)


##############################  SVM   #################################

train_x, train_y, test_x = read_tumor_data()

C_array = [1, 0.01, 0.0001]
Gamma_array = [1, 0.01, 0.001]
accuracy_array = []

######################## SVM WITH RBF Kernel #########################

for c in C_array:
    for gamma in Gamma_array:
        svc_tumor = SVC(C=c, kernel='rbf', gamma=gamma, random_state=1)
        svc_tumor.fit(train_x, train_y)
        scores = cross_val_score(svc_tumor,train_x,train_y,cv = 15)
        accuracy_array.append(np.mean(scores))
        print("RBF  c = ", c, " Gamma = ", gamma," Accuracy = ",np.mean(scores))
#print("Accuracy array  = ", accuracy_array)


########################  SVM WITH Poly. degree 3 ########################

for c in C_array:
    for gamma in Gamma_array:
        svc_tumor = SVC(C=c, kernel='poly',gamma=gamma,degree=3, random_state=1)
        svc_tumor.fit(train_x,train_y)
        scores = cross_val_score(svc_tumor, train_x, train_y, cv=15)
        accuracy_array.append(np.mean(scores))
        print("Poly 3, c = ", c, " Gamma = ", gamma," Accuracy = ",np.mean(scores))
#print("Accuracy array  = ", accuracy_array)

########################  SVM WITH Poly. degree 5 ########################

for c in C_array:
    for gamma in Gamma_array:
        svc_tumor = SVC(C=c, kernel='poly',gamma=gamma,degree=5, random_state=1)
        svc_tumor.fit(train_x,train_y)
        scores = cross_val_score(svc_tumor, train_x, train_y, cv=15)
        accuracy_array.append(np.mean(scores))
        print("Poly 5, c = ", c, " Gamma = ", gamma," Accuracy = ",np.mean(scores))
#print("Accuracy array  = ", accuracy_array)


######################## SVM WITH Linear ################################

for c in C_array:
    for gamma in Gamma_array:
        svc_tumor = SVC(C=c, kernel='linear',gamma=gamma, random_state=1)
        svc_tumor.fit(train_x,train_y)
        scores = cross_val_score(svc_tumor, train_x, train_y, cv=15)
        accuracy_array.append(np.mean(scores))
        print("Linear, c = ", c, " Gamma = ", gamma," Accuracy = ",np.mean(scores))


print("shape of accuracy array = ",np.shape(np.array(accuracy_array)))

index_max = np.argmax(accuracy_array)


np.set_printoptions(suppress=True)
C_NP_array = np.repeat(np.array(C_array),3)
print("C_NP_array = ",C_NP_array)

Gamma_NP_array = np.array(Gamma_array*3)
print("Gamma_NP_array = ",Gamma_NP_array)


index = index_max%9

print("Max Accuracy = ",accuracy_array[index_max]," at c = ",C_NP_array[index]," and gamma = ",Gamma_NP_array[index])


if index_max <9:
    print("RBF Kernel")
    print("Max Accuracy = ",accuracy_array[index_max]," at c = ",C_NP_array[index]," and gamma = ",
          Gamma_NP_array[index])
    svc_tumor_new = SVC(C=C_NP_array[index], kernel='rbf', gamma=Gamma_NP_array[index], random_state=1)
    svc_tumor_new.fit(train_x, train_y)
    pred_new = svc_tumor_new.predict(test_x)
elif index_max>=9 and index_max<18:
    print("Poly Degree 3")
    print("Max Accuracy = ", accuracy_array[index_max], " at c = ", C_NP_array[index], " and gamma = ",
          Gamma_NP_array[index])
    svc_tumor_new = SVC(C=C_NP_array[index], kernel='poly',gamma=Gamma_NP_array[index],degree=3, random_state=1)
    svc_tumor_new.fit(train_x, train_y)
    pred_new = svc_tumor_new.predict(test_x)

elif index_max>=18 and index_max<27:
    print("Poly Degree 5")
    print("Max Accuracy = ", accuracy_array[index_max], " at c = ", C_NP_array[index], " and gamma = ",
          Gamma_NP_array[index])
    svc_tumor_new = SVC(C=C_NP_array[index], kernel='poly',gamma=Gamma_NP_array[index],degree=5, random_state=1)
    svc_tumor_new.fit(train_x, train_y)
    pred_new = svc_tumor_new.predict(test_x)

else:
    print("Linear")
    print("Max Accuracy = ", accuracy_array[index_max], " at c = ", C_NP_array[index], " and gamma = ",
          Gamma_NP_array[index])

    svc_tumor = SVC(C=C_NP_array[index], kernel='linear', gamma=Gamma_NP_array[index], random_state=1)
    svc_tumor_new.fit(train_x, train_y)
    pred_new = svc_tumor_new.predict(test_x)


file_name = '../Predictions/Tumor/best_Tumor.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(pred_new, file_name, False)
