from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
# Create your views here.
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from . models import *


def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')


def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']
        d=Register.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        print(Name,email,password,conpassword)
        if password==conpassword:
            rdata=Register(email=email,password=password)
            rdata.save()
            return render(request,'login.html')
        else:
            msg='Register failed!!'
            return render(request,'registration.html')
    return render(request,'registration.html')
    # return render(request,'registration.html')


def userhome(request):
    return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_csv(file)
        messages.info(request,"Data Uploaded Successfully")
   return render(request,'load.html')

def view(request):
    col=df.to_html
    dummy=df.head(100)
    col=dummy.columns
    rows=dummy.values.tolist()
    return render(request, 'view.html',{'col':col,'rows':rows})
    # return render(request,'viewdata.html', {'columns':df.columns.values, 'rows':df.values.tolist()})
    
    
def preprocessing(request):
    global x_train,x_test,y_train,y_test,x,y
    if request.method == "POST":
        # size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        x=df.drop('Monkeypox',axis=1)
        y=df['Monkeypox']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)
        messages.info(request,"Data Preprocessed and It Splits Succesfully")
    return render(request,'preprocessing.html')
 

def model(request):
    if request.method == "POST":
        model = request.POST['algo']
        if model == "0":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train, y_train)
            y_pred = lda.predict(x_test)
            acc_lda = accuracy_score(y_test, y_pred)
            # Calculate precision, recall, and F1-score
            precision_lda = precision_score(y_test, y_pred)
            recall_lda = recall_score(y_test, y_pred)
            f1_score_lda = f1_score(y_test, y_pred)
            msg = 'Accuracy of LinearDiscriminantAnalysis : ' + str(acc_lda)
            msg1 = 'Accuracy of LinearDiscriminantAnalysis : ' + str(precision_lda)
            msg2 = 'Accuracy of LinearDiscriminantAnalysis : ' + str(recall_lda)
            msg3 = 'Accuracy of LinearDiscriminantAnalysis : ' + str(f1_score_lda)
            return render(request,'model.html',{'msg':msg,'msg1':msg1,'msg2':msg2,'msg3':msg3})
        elif model == "1":
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb.fit(x_train, y_train)
            y_pred = adb.predict(x_test)
            acc_adb = accuracy_score(y_test, y_pred)
            # Calculate precision, recall, and F1-score
            precision_adb = precision_score(y_test, y_pred)
            recall_adb = recall_score(y_test, y_pred)
            f1_score_adb = f1_score(y_test, y_pred)
            msg = 'Accuracy of AdaBoostClassifier : ' + str(acc_adb)
            msg1 = 'Accuracy of AdaBoostClassifier : ' + str(precision_adb)
            msg2 = 'Accuracy of AdaBoostClassifier : ' + str(recall_adb)
            msg3 = 'Accuracy of AdaBoostClassifier : ' + str(f1_score_adb)
            return render(request,'model.html',{'msg':msg,'msg1':msg1,'msg2':msg2,'msg3':msg3})
        elif model == "2":
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier()
            mlp.fit(x_train, y_train)
            y_pred = mlp.predict(x_test)
            acc_mlp = accuracy_score(y_test, y_pred)
            # Calculate precision, recall, and F1-score
            precision_mlp = precision_score(y_test, y_pred)
            recall_mlp = recall_score(y_test, y_pred)
            f1_score_mlp = f1_score(y_test, y_pred)
            msg = 'Accuracy of MLPClassifier : ' + str(acc_mlp)
            msg1 = 'Accuracy of MLPClassifier : ' + str(precision_mlp)
            msg2 = 'Accuracy of MLPClassifier : ' + str(recall_mlp)
            msg3 = 'Accuracy of MLPClassifier : ' + str(f1_score_mlp)
            return render(request,'model.html',{'msg':msg,'msg1':msg1,'msg2':msg2,'msg3':msg3})    
    return render(request,'model.html')

def prediction(request):
    global x_train, x_test, y_train, y_test, x, y
    if request.method == 'POST':
        f1 = float(request.POST.get('COVID_Symptoms', 0))
        f2 = float(request.POST.get('COVID_Tests', 0))
        f3 = float(request.POST.get('COVID_Vaccination', 0))
        f4 = float(request.POST.get('Age', 0))
        f5 = float(request.POST.get('Contact_with_Monkeypox', 0))
        f6 = float(request.POST.get('Travel_History', 0))
        f7 = float(request.POST.get('Healthcare_Worker', 0))
        f8 = float(request.POST.get('Body_Temperature', 0))
        f9 = float(request.POST.get('Hand_Hygiene', 0))
        f10 = float(request.POST.get('Mask_Usage', 0))
        f11 = float(request.POST.get('Closse_Contact', 0))

        PRED = [[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]]

        
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier()
        model.fit(x_train, y_train)
        xgp = np.array(model.predict(PRED))

        if xgp == 0:
            msg = 'The Person is not affected with MONKEY_POX'
        else:
            msg = 'The Person is affected with MONKEY_POX'

        return render(request, 'prediction.html', {'msg': msg})

    return render(request, 'prediction.html')