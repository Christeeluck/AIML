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


#--------------------training section------
#9 read in excel file and store the data in a dataframe

#10 extract weight and volume fields and strore in X

#11 extract co2 field and store in variable y

#12 fit a lin reg model to variables X and y

#13 predict value of co2 for volvo Xc70 with weight of 1746kg and vol of 2000cm
#explain why value is difference from file values

#14 find the ref coefficient between X and y and explain its meaning

#15 find the R2 score and explain its meaning

#16 scale the variables in X using the standardization method

#17 repeat step 13 using scaled X var

#18 explain any diff betwwen 13 and 16 

#19 explain what is the train /test model and why its important

#20 applying train/test method