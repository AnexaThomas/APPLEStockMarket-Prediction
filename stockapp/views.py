from django.shortcuts import render,redirect
from .models import user, company
from django.contrib import messages
import pandas_datareader as web
from .models import contacts,feed
import pandas as pd
from django.http import HttpResponse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import datetime
from sklearn.metrics import accuracy_score
# Create your views here.

def home(request):
    companies = {'c': company.objects.all()}
    return render(request, 'index.html', companies)

def about(request):
    return render(request, 'about.html')

def contact(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        message=request.POST['message']

        contacts(name=name,email=email,message=message).save()
        messages.success(request,"Feedback sended Successfully...!")
        return render(request,'contact.html')
    
    else:
        return render(request,'contact.html')
def blog(request):
    return render(request, 'blog.html')

def comp(request):
    companies = {'c': company.objects.all()}
    return render(request, 'company.html', companies)

def service(request):
    return render(request, 'service.html')

def single(request):
    return render(request, 'single.html')

def help(request):
    return render(request, 'help.html')

def register(request):
    if request.method=='POST':
        name=request.POST['name']   
        email=request.POST['email']
        password=request.POST['password']
        rpwd=request.POST['repeatpassword']
        emailExist = user.objects.filter(email = email)

        if(emailExist):
            messages.success(request,"E-mail Id already exist...!")
            return render(request,'register.html')
        else:
            user(name=name, email=email, password=password, rpwd=rpwd).save()
            messages.success(request, 'The New User ' + request.POST['name'] + " is saved Successfully...!")
            return redirect('/login')
    
    else:
        return render(request,'register.html') 

def feedback(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        message=request.POST['message']

        feed(name=name,email=email,message=message).save()
        messages.success(request,"Feedback sended Successfully...!")
        return render(request,'feedback.html')
    
    else:
        return render(request,'feedback.html')

def viewfeedback(request):
    context={
    'feeds':feed.objects.all()
    }
    return render(request, 'viewfeedback.html',context)


def login(request):
    if request.method == "POST":

        try:
            Userdetails = user.objects.get(email=request.POST['email'], password=request.POST['password'])
            print("Username=", Userdetails)
            request.session['id'] = Userdetails.id
            print(request.session['id'])
            return render(request, 'prediction.html')
        except user.DoesNotExist as e:
            messages.success(request, 'Username/Password Invalid...!')
    return render(request, 'login.html')
  

def logout(request):
    try:
        del request.session['email'] 
    except: 
        return render(request,'index.html')
    return render(request,'index.html')

def prediction(request): 
  
    return render(request, 'prediction.html')

def forgotPassword(request):
    return render (request,'forgot-password.html')

def updatePassword(request):
    email = request.POST['email']
    password = request.POST['password']
    
    checkAccountExistOrNot = user.objects.filter(email = email)
    if checkAccountExistOrNot:
        user.objects.filter(email = email).update(password = password)
        messages.success(request,"Password updated Successfully...!")
        return render(request,'forgot-password.html')
    else:
        messages.success(request,"No account found on this E-mail Id...!")
        return render (request,'forgot-password.html')
    
    

def predi(request):
    end_date = request.POST.get('end_date')
    
    #importing libraries

    # print('8888',end_date)


    #Get the stock quote
    df=web.DataReader('AAPL',data_source='yahoo',start='2015-01-01',end=end_date)
   
   

  
    #creating a new dataframe with only 'close columns
    data=df.filter(["Close"])
    #convert dataframe into a numpy array
    dataset=data.values
    #get the number of rows to train the model on
    training_data_len =math.ceil(len(dataset)*.8)
   
    #scale the data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
   
   
    #creating training dataset
    #creating a scaled dataset
    train_data=scaled_data[0:training_data_len,:]
    #split the data into x_train and y_train dataset
    x_train=[]
    y_train=[]
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
        if i<=61:
            print(x_train)
            print(y_train)
            print()
        
    #converting x_train and y_tran into numpy arrays
    x_train,y_train=np.array(x_train),np.array(y_train)

    #reshape the data
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape
   

   #build LSTM model
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #compile the model
    model.compile(optimizer="adam",loss="mean_squared_error")
    
    #train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1) 
   
    #creating testing dataset
    #creating new array containing scaled values from index 1543 to 2003
    test_data=scaled_data[training_data_len-60:,:]
    #creat the data set x_test and y_test
    x_test=[]
    y_test=dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    
    #converting data into numpy arrray
    x_test=np.array(x_test)
    
    #Reshape the data set
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    #get the models predicted price values
    predictions=model.predict(x_test)
    predictions=scaler.inverse_transform(predictions)
    
    #geting the root mean squared root(RMSE)
    rmse=np.sqrt(np.mean(predictions-y_test)**2)

    #get the quote
    apple_quote=web.DataReader('AAPL',data_source="yahoo",start='2015-01-01',end=end_date)
    #create a new dataframe
    new_df=apple_quote.filter(["Close"])
    #getting the last 60 day closing price values and convert the dataframe to an array
    last_60_days=new_df[-60:].values
    #scale the data to be values between 0 and 1
    last_60_days_scaled=scaler.transform(last_60_days)
    #create an empty list
    x_test=[]
    #append the past 60 days
    x_test.append(last_60_days_scaled)
    #convert the x_test data set to a numpy array
    x_test=np.array(x_test)
    #reshape the data
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    #get the predicted scaled price
    pred_price=model.predict(x_test)
    #undo the scaling
    pred_price=scaler.inverse_transform(pred_price)
    print('llll',pred_price)
 
    #get the quote
    apple_quote2=web.DataReader("AAPL",data_source="yahoo",start='2019-09-24',end=end_date)
    print('*****',apple_quote2["Close"])
    
    #finalPrediction = apple_quote2["Close"][-1]
    finalPrediction = pred_price
    print('99999999',finalPrediction )
   
    # print(type(apple_quote2["Close"]))

    # custom = apple_quote2["Close"]
    # for er in custom:
    #     print('5555',er)
      #visualize the closing price history
    
    # plt.figure(figsize=(16,8))
    # sns.set(style="darkgrid")
    # plt.title("Close Price History")
    # plt.plot(df["Close"])
    # plt.xlabel("Data",fontsize=18)
    # plt.ylabel("Close Price USD ($),",fontsize=18)
    # plt.show()

    
    return render(request, 'predi.html', {'finalPrediction':finalPrediction})