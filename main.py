# Step 1: Import packages, functions, and classes
import seaborn as sns
from enum import auto
from tkinter import ttk, Frame
from tkinter import *
import tkinter as tk
from sklearn import tree
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from featurewiz import featurewiz
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# read data from file csv
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015 - diabetes_binary_health_indicators_BRFSS2015.csv')
# print(data["BMI"].max)

# cleaning
# 1 ) Datatype Null , Count
# data.info()

# 2) empty cell
# data.drop(inplace = True)

# 3) duplicate
# data.drop_duplicate()

# 4) if null
# data.isnull().sum()


#
# automatic feature selection by using featurewiz package
target = 'Diabetes_binary'

# heat map
# Using Pearson Correlation
plt.figure(figsize=(12, 12))
cor = data.drop(columns="Diabetes_binary", axis=1).corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

features, train = featurewiz(data, target, corr_limit=0.45, verbose=2)
# print(features)

# features dataset
x = train.drop(columns="Diabetes_binary", axis=1)

# target dataset
y = train["Diabetes_binary"]

# preprocessing the features
# X_scaled = StandardScaler().fit_transform(x)


# slove unbalanced data
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)


""" #print('Original dataset shape', y.value_counts())
#print('Resample dataset shape', y_smote.value_counts())
#dupli =data[data.duplicated()]
# print(y_smote.duplicated().value_counts())
 """

test_scores = []

train_scores = []

# split data 20%
x_train, x_test, y_train, y_test = train_test_split(
    x_smote, y_smote, test_size=0.25, random_state=0)



choose1 = input("choose one model to test: ")

if choose1 == "1":
    print("svm\n")
  #  clf = svm.SVC(kernel="linear", max_iter=20)
    for i in range(1, 5):
        clf = svm.LinearSVC(max_iter=20)
        clf.fit(x_train, y_train)

        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))
    plt.figure(figsize=(12, 5))
    p = sns.lineplot(range(1, 5), train_scores,
                     marker='*', label='Train Score')
    p = sns.lineplot(range(1, 5), test_scores, marker='o', label='Test Score')
    plt.show()

    """     # Initialise the Scaler
    scaler = StandardScaler()
    # To scale data
    scaler.fit(x_train) """

elif (choose1 == "2"):
    print("tree\n")
    for i in range(1, 15):
#criterion='gini', max_depth=10, random_state=2
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))
    plt.figure(figsize=(12, 5))
    p = sns.lineplot(range(1, 15), train_scores,
                     marker='*', label='Train Score')
    p = sns.lineplot(range(1, 15), test_scores, marker='o', label='Test Score')
    plt.show()

elif choose1 == "3":
    print("logistic\n")
    clf = LogisticRegression()
    for i in range(1, 15):

        clf = LogisticRegression()
        clf.fit(x_train, y_train)

        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))
    plt.figure(figsize=(12, 5))
    p = sns.lineplot(range(1, 15), train_scores,
                     marker='*', label='Train Score')
    p = sns.lineplot(range(1, 15), test_scores, marker='o', label='Test Score')
    plt.show()




elif(choose1 == 4):
    for i in range(1, 15):

        clf = RandomForestClassifier(i)
        clf.fit(x_train, y_train)

        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))
    plt.figure(figsize=(12, 5))
    p = sns.lineplot(range(1, 15), train_scores,
                     marker='*', label='Train Score')
    p = sns.lineplot(range(1, 15), test_scores, marker='o', label='Test Score')
    plt.show()

else:
    # Objects to be received by the voting classifier object
    tree_clf = tree.DecisionTreeClassifier()
    logistic_clf = LogisticRegression()
    svc_clf = svm.LinearSVC(max_iter=20)

    # making a list to put in a loop instead of calling the fun 3 time
    # the list has tuples with the names of the classifiers and its objects name
    estimators = [
        ('tree', tree_clf),
        ('logistic', logistic_clf),
        ('svc', svc_clf),
    ]

    # the voting classifier object
    voting_clf = VotingClassifier(estimators)

    # adding the voting classifier object to the list
    all_estimators = estimators + [('voting', voting_clf)]

    # for loop to display the accuracy of training,validation,testing
    # for (name, clf) in all_estimators:
    #     acc_train, acc_val, acc_test = evaluate_model(clf, x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled,
    #                                                   y_test)

print("===============")
# print(arr.shape)
y_pred2 = clf.predict(x_test)

def test():
    clf = svm.LinearSVC(max_iter=20)
    clf.fit(x_train, y_train)
    y_pred1 = clf.predict(x_test)
    svm_accuracy = metrics.accuracy_score(y_test, y_pred1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred3 = clf.predict(x_test)
    tree_accuracy = metrics.accuracy_score(y_test, y_pred3)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred4 = clf.predict(x_test)
    logistic_accuracy = metrics.accuracy_score(y_test, y_pred4)

    if (svm_accuracy > logistic_accuracy and svm_accuracy > tree_accuracy):
        print("Accuracy Svm is best Accuracy")

    elif (svm_accuracy < logistic_accuracy and logistic_accuracy > tree_accuracy):
        print("logistic accuracy is the best accuracy")

    elif (tree_accuracy > logistic_accuracy and svm_accuracy < tree_accuracy):
        print("tree accuracy is the best accuracy")


print('confusion_matrix:', confusion_matrix(y_test, y_pred2))

accuracy4 = metrics.accuracy_score(y_test, y_pred2)
print('accuracy:', accuracy4)

precision_score4 = metrics.precision_score(y_test, y_pred2)
print('precision_score:', precision_score4)

f1_score4 = metrics.f1_score(y_test, y_pred2)
print('f1_score:', f1_score4)

report = classification_report(y_test, y_pred2)
print('report:', report, sep='\n')

testData = pd.read_csv('diabetesTest.csv')


XnewTest = testData[features]
YnewTest = testData["Diabetes_binary"]

y_predNew = clf.predict(XnewTest)

print('confusion_matrix:', confusion_matrix(YnewTest, y_predNew))

accuracy4 = metrics.accuracy_score(YnewTest, y_predNew)
print('accuracy:', accuracy4)

precision_score4 = metrics.precision_score(YnewTest, y_predNew)
print('precision_score:', precision_score4)

f1_score4 = metrics.f1_score(YnewTest, y_predNew)
print('f1_score:', f1_score4)

"""from sklearn import tree
from sklearn import svm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

arr = {
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 0,
    "BMI": 0,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits": 0,
    "Veggies": 0,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 0,
    "GenHlth": 0,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 0,
    "Age": 0,
    "Education": 0,
    "Income": 0,
}


def get_d():
    arr["HighBP"] = var.get()
    arr["HighChol"] = var2.get()
    arr["CholCheck"] = var3.get()
    arr["BMI"] = int(bmi_input.get())
    arr["Smoker"] = var4.get()
    arr["Stroke"] = var5.get()
    arr["HeartDiseaseorAttack"] = var6.get()
    arr["PhysActivity"] = var7.get()
    arr["Fruits"] = var8.get()
    arr["Veggies"] = var9.get()
    arr["HvyAlcoholConsump"] = var10.get()
    arr["AnyHealthcare"] = var11.get()
    arr["NoDocbcCost"] = var12.get()
    arr["GenHlth"] = int(gh_input.get())
    arr["MentHlth"] = int(mh_input.get())
    arr["PhysHlth"] = int(ph_input.get())
    arr["DiffWalk"] = var13.get()
    arr["Sex"] = var14.get()
    arr["Age"] = int(age_input.get())
    arr["Education"] = int(ed_input.get())
    arr["Income"] = int(in_input.get())
    print(arr)
    convertDict()


test = []


def convertDict():
    for data in arr:
        test.append(arr[data])
    print(test)


# read data from file csv
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015 - diabetes_binary_health_indicators_BRFSS2015.csv')
# print(data["BMI"].max)

x = data.drop(columns="Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

# solve unbalanced data
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)

# print(x_smote.shape)
print('Original dataset shape', y.value_counts())
print('Resample dataset shape', y_smote.value_counts())

# dupli =data[data.duplicated()]
dupli =data[data.duplicated()]
print(dupli['Diabetes_binary'].value_counts())
# print(y_smote.duplicated().value_counts())
# split data 20%

x_train, x_test, y_train, y_test = train_test_split(
    x_smote, y_smote, test_size=0.2, random_state=0)


# training function

def get_choice(x):
    if x == 1:
        print("svm\n")
        clf = svm.SVC(kernel="linear", max_iter=20)

    elif x == 2:
        print("tree\n")
        clf = tree.DecisionTreeClassifier()

    elif x == 3:
        print("logistic\n")
        clf = LogisticRegression()

    clf.fit(x_train, y_train)
    y_pred2 = clf.predict(x_test)

    print('confusion_matrix:', confusion_matrix(y_test, y_pred2))
    cm_result.configure(text=confusion_matrix(y_test, y_pred2))

    accuracy4 = metrics.accuracy_score(y_test, y_pred2)
    print('accuracy:', accuracy4)
    acc_result.configure(text=accuracy4)

    precision_score4 = metrics.precision_score(y_test, y_pred2)
    print('precision_score:', precision_score4)
    pre_result.configure(text=precision_score4)

    f1_score4 = metrics.f1_score(y_test, y_pred2)
    print('f1_score:', f1_score4)
    f1_result.configure(text=f1_score4)


def test_f(c, test2):
    x = test2
    x = np.expand_dims(x, axis=1)
    y = data.drop(columns="Diabetes_binary", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    if c == 1:
        print("svm\n")
        clf = svm.SVC(kernel="linear", max_iter=20)

    elif c == 2:
        print("tree\n")
        clf = tree.DecisionTreeClassifier()

    elif c == 3:
        print("logistic\n")
        clf = LogisticRegression()

    clf.fit(x_train, y_train)
    y_pred2 = clf.predict(x_test)

    print('confusion_matrix:', confusion_matrix(y_test, y_pred2))
    cm_result.configure(text=confusion_matrix(y_test, y_pred2))

    accuracy4 = metrics.accuracy_score(y_test, y_pred2)
    print('accuracy:', accuracy4)
    acc_result.configure(text=accuracy4)

    precision_score4 = metrics.precision_score(y_test, y_pred2)
    print('precision_score:', precision_score4)
    pre_result.configure(text=precision_score4)

    f1_score4 = metrics.f1_score(y_test, y_pred2)
    print('f1_score:', f1_score4)
    f1_result.configure(text=f1_score4)


#  prdc_result.configure(text=)


########################################################
import tkinter as tk
from tkinter import *
from tkinter import ttk, Frame

# Creating the window
window = tk.Tk()
window.geometry("750x730")
window.title("Diabetes Health Indicator")

# Models
frame1 = Frame(window)
frame1.grid(column=0, row=0)

md_f = Label(frame1, text="Methodology")
md_f.grid(column=0, row=0)

choice = IntVar()

ckb_svm = Checkbutton(frame1, text="SVM", variable=choice, onvalue=1)
ckb_svm.grid(column=0, row=3)

ckb_lr = Checkbutton(frame1, text="Logistic Regression", onvalue=3, variable=choice)
ckb_lr.grid(column=2, row=3)

ckb_dt = Checkbutton(frame1, text="Decision Tree", onvalue=2, variable=choice)
ckb_dt.grid(column=3, row=3)

btn_train = Button(frame1, text="Train", width=10, command=lambda: get_choice(choice.get()))
btn_train.grid(column=0, row=5)

btn_tst = Button(frame1, text="Test", width=10, command=get_d)
btn_tst.grid(column=3, row=5)

lb = Label(frame1, text="Methodology")
lb.grid(column=0, row=0, ipadx=10, ipady=10)

####################################################################

q = Frame(window)  # frame contains questions and answers
q.grid(column=0, row=1)

qf = Frame(q)  # questions frame
qf.grid(column=0, row=0)

af = Frame(q)  # answer frame
af.grid(column=1, row=0)

# High Blood Pressure ##############################################
hbp_f = Frame(q)
hbp_f.grid(column=0, row=2)

hbp_lbl = Label(qf, text="->Do you have High Blood Pressure?")
hbp_lbl.grid(column=0, row=0, ipadx=10, ipady=10)

spc3 = Label(af, text=" ")
spc3.grid(column=0, row=0)

var = IntVar()

hbp_an = Frame(af)
hbp_an.grid(column=0, row=0, ipadx=10, ipady=10)

hbp_yes = Radiobutton(hbp_an, text="Yes", variable=var, value=1)
hbp_yes.grid(column=0, row=0)

hbp_no = Radiobutton(hbp_an, text="No", variable=var, value=0)
hbp_no.grid(column=1, row=0)

# High Alcohol #######################################################
hc_f = Frame(q)
hc_f.grid(column=0, row=1)

hc_lbl = Label(qf, text="->Are you high Alcoholec?")
hc_lbl.grid(column=0, row=1, ipadx=10, ipady=10)

var2 = IntVar()

hc_an = Frame(af)
hc_an.grid(column=0, row=1, ipadx=10, ipady=10)

hc_yes = Radiobutton(hc_an, text="Yes", variable=var2, value=1)
hc_yes.grid(column=0, row=0)

hc_no = Radiobutton(hc_an, text="No", variable=var2, value=0)
hc_no.grid(column=1, row=0)

# Drinks Alcohol Check ########################################
cc_f = Frame(q)
cc_f.grid(column=0, row=3)

cc_lbl = Label(qf, text="->Do you drink any kind of Alcohols?")
cc_lbl.grid(column=0, row=2, ipadx=10, ipady=10)

var3 = IntVar()

cc_an = Frame(af)
cc_an.grid(column=0, row=2, ipadx=10, ipady=10)

cc_yes = Radiobutton(cc_an, text="Yes", variable=var3, value=1)
cc_yes.grid(column=0, row=0)

cc_no = Radiobutton(cc_an, text="No", variable=var3, value=0)
cc_no.grid(column=1, row=0)

# Smoker #############################################
smk_f = Frame(q)
smk_f.grid(column=0, row=4)

smk_lbl = Label(qf, text="->Do you Smoke?")
smk_lbl.grid(column=0, row=3, ipadx=10, ipady=10)

var4 = IntVar()

smk_an = Frame(af)
smk_an.grid(column=0, row=3, ipadx=10, ipady=10)

smk_yes = Radiobutton(smk_an, text="Yes", variable=var4, value=1)
smk_yes.grid(column=0, row=0)

smk_no = Radiobutton(smk_an, text="No", variable=var4, value=0)
smk_no.grid(column=1, row=0)

# Stroke ################################################
str_f = Frame(q)
str_f.grid(column=0, row=5)

str_lbl = Label(qf, text="->Do you Stroke?")
str_lbl.grid(column=0, row=4, ipadx=10, ipady=10)

var5 = IntVar()

str_an = Frame(af)
str_an.grid(column=0, row=4, ipadx=10, ipady=10)

str_yes = Radiobutton(str_an, text="Yes", variable=var5, value=1)
str_yes.grid(column=0, row=0)

str_no = Radiobutton(str_an, text="No", variable=var5, value=0)
str_no.grid(column=1, row=0)

# Heart Disease-or Attack ##############################################
hda_f = Frame(q)
hda_f.grid(column=0, row=6)

hda_lbl = Label(qf, text="->Do you have Heart Disease-or Attack?")
hda_lbl.grid(column=0, row=5, ipadx=10, ipady=10)

var6 = IntVar()

hda_an = Frame(af)
hda_an.grid(column=0, row=5, ipadx=10, ipady=10)

hda_yes = Radiobutton(hda_an, text="Yes", variable=var6, value=1)
hda_yes.grid(column=0, row=0)

hda_no = Radiobutton(hda_an, text="No", variable=var6, value=0)
hda_no.grid(column=1, row=0)

# Physical Activities ##############################################
pa_f = Frame(q)
pa_f.grid(column=0, row=7)

pa_lbl = Label(qf, text="->Do you have do any Physical Activities?")
pa_lbl.grid(column=0, row=6, ipadx=10, ipady=10)

var7 = IntVar()

pa_an = Frame(af)
pa_an.grid(column=0, row=6, ipadx=10, ipady=10)

pa_yes = Radiobutton(pa_an, text="Yes", variable=var7, value=1)
pa_yes.grid(column=0, row=0)

pa_no = Radiobutton(pa_an, text="No", variable=var7, value=0)
pa_no.grid(column=1, row=0)

# Fruits ########################################
f_f = Frame(q)
f_f.grid(column=0, row=8)

f_lbl = Label(qf, text="->Do you eat Fruits?")
f_lbl.grid(column=0, row=7, ipadx=10, ipady=10)

var8 = IntVar()

f_an = Frame(af)
f_an.grid(column=0, row=7, ipadx=10, ipady=10)

f_yes = Radiobutton(f_an, text="Yes", variable=var8, value=1)
f_yes.grid(column=0, row=0)

f_no = Radiobutton(f_an, text="No", variable=var8, value=0)
f_no.grid(column=1, row=0)

# Veggies ########################################
v_f = Frame(q)
v_f.grid(column=0, row=9)

v_lbl = Label(qf, text="->Do you eat Veggies?")
v_lbl.grid(column=0, row=8, ipadx=10, ipady=10)

var9 = IntVar()

v_an = Frame(af)
v_an.grid(column=0, row=8, ipadx=10, ipady=10)

v_yes = Radiobutton(v_an, text="Yes", variable=var9, value=1)
v_yes.grid(column=0, row=0)

v_no = Radiobutton(v_an, text="No", variable=var9, value=0)
v_no.grid(column=1, row=0)

# Heavy Alcohol Consumption ########################################
hac_f = Frame(q)
hac_f.grid(column=0, row=10)

hac_lbl = Label(qf, text="->Do you have Heavy Alcohol Consumption?")
hac_lbl.grid(column=0, row=9, ipadx=10, ipady=10)

var10 = IntVar()

hac_an = Frame(af)
hac_an.grid(column=0, row=9, ipadx=10, ipady=10)

hac_yes = Radiobutton(hac_an, text="Yes", variable=var10, value=1)
hac_yes.grid(column=0, row=0)

hac_no = Radiobutton(hac_an, text="No", variable=var10, value=0)
hac_no.grid(column=1, row=0)

# Any Health Care ########################################
ahc_f = Frame(q)
ahc_f.grid(column=0, row=11)

ahc_lbl = Label(qf, text="->Do you have Any Health Care?")
ahc_lbl.grid(column=0, row=10, ipadx=10, ipady=10)

var11 = IntVar()

ahc_an = Frame(af)
ahc_an.grid(column=0, row=10, ipadx=10, ipady=10)

ahc_yes = Radiobutton(ahc_an, text="Yes", variable=var11, value=1)
ahc_yes.grid(column=0, row=0)

ahc_no = Radiobutton(ahc_an, text="No", variable=var11, value=0)
ahc_no.grid(column=1, row=0)

# No Docbc Cost ########################################
ndc_f = Frame(q)
ndc_f.grid(column=0, row=12)

ndc_lbl = Label(qf, text="->Do you have No Docbc Cost?")
ndc_lbl.grid(column=0, row=11, ipadx=10, ipady=10)

var12 = IntVar()

ndc_an = Frame(af)
ndc_an.grid(column=0, row=11, ipadx=10, ipady=10)

ndc_yes = Radiobutton(ndc_an, text="Yes", variable=var12, value=1)
ndc_yes.grid(column=0, row=0)

ndc_no = Radiobutton(ndc_an, text="No", variable=var12, value=0)
ndc_no.grid(column=1, row=0)
################################################################
q2 = Frame(window)
q2.grid(column=1, row=1)

lbls = Frame(q2)
lbls.grid(column=0, row=0)

entrys = Frame(q2)
entrys.grid(column=1, row=0)

########################################################################################################################

q3 = Frame(window)
q3.grid(column=1, row=0)

acc_lbl = Label(q3, text="Accuracy")
acc_lbl.grid(column=0, row=0)

pr_lbl = Label(q3, text="Precision")
pr_lbl.grid(column=0, row=1)

f1_lbl = Label(q3, text="F1 score")
f1_lbl.grid(column=0, row=2)

cm_lbl = Label(q3, text="Confusion matrix")
cm_lbl.grid(column=0, row=3)

acc_result = Label(q3, text="result")
acc_result.grid(column=1, row=0)

f1_result = Label(q3, text="result")
f1_result.grid(column=1, row=2)

cm_result = Label(q3, text="result")
cm_result.grid(column=1, row=3)

pre_result = Label(q3, text="result")
pre_result.grid(column=1, row=1)

# BMI ########################################
bmi_lbl = Label(lbls, text="BMI")
bmi_lbl.grid(column=0, row=0)

bmi_input = Entry(entrys)
bmi_input.grid(column=1, row=0)

######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=1)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=1)
######

# General Health ########################################
gh_lbl = Label(lbls, text="General Health")
gh_lbl.grid(column=0, row=2)

gh_input = Entry(entrys)
gh_input.grid(column=1, row=2)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=3)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=3)
######

# Mental Health ##########################################
mh_lbl = Label(lbls, text="Mental Health")
mh_lbl.grid(column=0, row=4)

mh_input = Entry(entrys)
mh_input.grid(column=1, row=4)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=5)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=5)
######


# Physical Health ##########################################
ph_lbl = Label(lbls, text="Physical Health")
ph_lbl.grid(column=0, row=6)

ph_input = Entry(entrys)
ph_input.grid(column=1, row=6)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=7)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=7)
######


# Age ##########################################
age_lbl = Label(lbls, text="Age")
age_lbl.grid(column=0, row=8)

age_input = Entry(entrys)
age_input.grid(column=1, row=8)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=9)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=9)
######


# Eduction ##########################################
ed_lbl = Label(lbls, text="Education")
ed_lbl.grid(column=0, row=10)

ed_input = Entry(entrys)
ed_input.grid(column=1, row=10)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=11)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=11)
######

# Income ##########################################
in_lbl = Label(lbls, text="Income")
in_lbl.grid(column=0, row=12)

in_input = Entry(entrys)
in_input.grid(column=1, row=12)
######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=13)

spc2 = Label(entrys, text='''
''')
spc2.grid(column=0, row=13)
######


# Sex ##########################################
sx_lbl = Label(lbls, text="Sex")
sx_lbl.grid(column=0, row=14)

var14 = IntVar()

sx_an = Frame(entrys)
sx_an.grid(column=1, row=14)

sx_fem = Radiobutton(sx_an, text="Female", variable=var14, value=1)
sx_fem.grid(column=0, row=0)

sx_m = Radiobutton(sx_an, text="Male", variable=var14, value=0)
sx_m.grid(column=1, row=0)

######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=15)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=15)
######


# Different Walk ##########################################
dfw_lbl = Label(lbls, text="Diff Walk")
dfw_lbl.grid(column=0, row=16)

var13 = IntVar()

dfw_an = Frame(entrys)
dfw_an.grid(column=1, row=16)

dfw_yes = Radiobutton(dfw_an, text="Yes", variable=var13, value=1)
dfw_yes.grid(column=0, row=0)

dfw_no = Radiobutton(dfw_an, text="No", variable=var13, value=0)
dfw_no.grid(column=1, row=0)

######
spc2 = Label(lbls, text=" ")
spc2.grid(column=0, row=17)

spc2 = Label(entrys, text=" ")
spc2.grid(column=0, row=17)

#############################################

prd = Button(q2, text="Predict", width=10, command=lambda: test_f(choice.get(), test))
prd.grid(column=1, row=10)

xx = Label(q2, text=" ")
xx.grid(column=1, row=11)

prdc_result = Label(q2, text="result")
prdc_result.grid(column=1, row=12)

window.mainloop()
"""