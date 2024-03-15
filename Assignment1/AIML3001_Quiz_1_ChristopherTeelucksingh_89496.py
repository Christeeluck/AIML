#Christopher Teelucksingh 
#89496
#AIML3001 Assignment 1



import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import os
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


#1 scatter plot
x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
y=[90,80,87,88,90,86,89,87,94,78,77,85,86]
plt.scatter(x,y)
plt.xlabel('Age of car')
plt.ylabel('Speed of car')
plt.show()


#2 linear regression
slope,intercept,r,p,std_err = stats.linregress(x,y)
Reg_LineY = np.array(x)*slope + intercept
plt.scatter(x,y)
plt.plot(x,Reg_LineY)
plt.xlabel('Age of car')
plt.ylabel('Speed of car')
plt.grid()
plt.show()

#3 estimate coefficient of correlation r
print(r)

#4 predict spd of 13 yr old car using lin reg model
Car_Age=13
Speed_Predict= Car_Age *slope + intercept
print(Speed_Predict)

#5 3rd degree polynomial plot of data
n=3
mymodel=np.poly1d(np.polyfit(x,y,n))
myline=np.linspace(1,17,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.xlabel('Age of car')
plt.ylabel('Speed of car')
plt.grid()
plt.show()



#6 predict speed of 13 year old car using 3rd degree polynomial
speed=mymodel(13)
print(speed)


#7 estimate the r-squared value 
print(r2_score(y,mymodel(x)))

#8 difference between lin and poly prediction


#----------excel sheet section-------------


#9 read in excel file and store the data in a dataframe
dir_path = r"C:\Users\Admin\Desktop\compEng\Year 3\Semester2\AIML\AIML\Assignment1"
file_name = "data1.csv"
file_path = os.path.join(dir_path, file_name)

# reading in the file as an object
df = pd.read_csv(file_path)

#10 extract weight and volume fields and strore in X
X = df[["Weight","Volume"]]

#11 extract co2 field and store in variable y
y = df[["CO2"]]


#12 fit a lin reg model to variables X and y
linReg = linear_model.LinearRegression() 
linReg.fit(X, y)


#13 predict value of co2 for volvo Xc70 with weight of 1746kg and vol of 2000cm
predictedCO2 = linReg.predict([[1746, 2000]])



#14 find the ref coefficient between X and y and explain its meaning
print(linReg.coef_)


#15 find the R2 score and explain its meaning
print(linReg.score(X,y))


#16 scale the variables in X using the standardization method
scale = StandardScaler()
scaledX = scale.fit_transform(X) 


#17 repeat step 13 using scaled X var
linRegScaled = linear_model.LinearRegression()
linRegScaled.fit(scaledX, y)
scaledValue = scale.transform([[1746, 2000]])
scaledPredictedCO2 = linRegScaled.predict([scaledValue[0]])



#20 applying train/test method

#--------------------training section------
#extract first 30 rows from dataframe
trainx= df[["Weight","Volume"]].head(30)
trainy=df["CO2"].head(30)

#fit training data to linear model
TrainlinReg = linear_model.LinearRegression() 
TrainlinReg.fit(trainx, trainy)


#predict value of co2 for volvo Xc70 with weight of 1746kg and vol of 2000cm
Train_predictedCO2 = TrainlinReg.predict([[1746, 2000]])



#find the ref coefficient between X and y and explain its meaning
print(TrainlinReg.coef_)


#find the R2 score and explain its meaning
print(TrainlinReg.score(trainx,trainy))

#------------Testting Section-------------
#extract last 6 rows from datafram since the first 30 
#rows were used for training.There are 36 rows in total
testx= df[["Weight","Volume"]].tail(6)
testy=df["CO2"].tail(6)

testResults=pd.Series(dtype=float,name="PredCO2")
#predict CO2 for each value of the testing set
for index, row in testx.iterrows():
    #predictCo2
    testResults.loc[index]=TrainlinReg.predict([[row["Weight"], row["Volume"]]])
    
#calc percentage diff between actual values and predicted values
percDiff=abs(((testResults-testy)/testy)*100)
    
    
    
    
    
    
    
    
