
#Import Data
import pandas as pd
data=pd.read_csv("C:/Users/Saniya and Family/Downloads/Iris.csv")
data = data.drop(columns=["Id"])

#Split train and test
x=data.drop(columns=["Species"])
y=data["Species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.7) 

#Training model
from sklearn.ensemble import RandomForestClassifier
modelrf = RandomForestClassifier(n_estimators=200,max_depth=3)
modelrf = modelrf.fit(x_train, y_train) 
y_pred=modelrf.predict(x_test)
print("Score for model 0", end="  ")
print(modelrf.score(x_test,y_test))


#Using pool-based sampling with a batch size of 10 for 4 loops
i=1
for i in range(1,5):
    trainx = x_train.append(x_test[0:i*10])
    trainy = y_train.append(y_test[0:i*10])
    
    modelrf = modelrf.fit(trainx, trainy) 
    y_pred=modelrf.predict(x_test)
    print("Score for model",i, end="  ")
    print(modelrf.score(x_test,y_test)) 

result = pd.concat([trainx, trainy], axis=1, join="inner")
result.to_csv("ai_data.csv")
