import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

playerList=pd.read_csv('testData.csv')["Player"][0:30].values






scaler = StandardScaler()
shots2016 = pd.read_csv("shots_2016.csv")
shots2017 = pd.read_csv("shots_2017.csv")
shots2018 = pd.read_csv("shots_2018.csv")
shots2019 = pd.read_csv("shots_2019.csv")
shots2020 = pd.read_csv("shots_2020.csv")
shots2021 = pd.read_csv("shots_2021.csv")

predictions_countList=list()
actual_countList=list()
moneyPuckxGList = list()
stattrickxGList = list()


for playerName in playerList:
    trainingData = pd.concat([shots2016,shots2017,shots2018,shots2019,shots2020,shots2021])
    testData = pd.read_csv("shots_2022.csv")
    moneypuck = pd.read_csv("moneypuck2022.csv")
    stattrick = pd.read_csv("naturalstattrick2022.csv")

    moneypuck = moneypuck[(moneypuck["name"] == playerName) & (moneypuck["situation"]=="all")]
    moneyPuckxG = moneypuck["I_F_xGoals"].values[0]

    stattrick = stattrick[stattrick["Player"] == playerName]
    stattrickxG = stattrick["ixG"].values[0]


    trainingData = trainingData[trainingData["shooterName"] == playerName]
    trainingData=trainingData.drop(['teamCode','awayTeamCode','shotID','shooterName','gameOver','game_id','goalieNameForShot','homeTeamCode'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['homeWinProbability','homeTeamWon','id','isPlayoffGame','playerNumThatDidEvent','playerNumThatDidLastEvent'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['playoffGame','roadTeamCode','season','shooterPlayerId','wentToOT','wentToShootout','xFroze','xGoal','xPlayContinuedInZone','xPlayContinuedOutsideZone','homeTeamScore'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['xPlayStopped','xRebound','xShotWasOnGoal','event','roadTeamScore','shotGoalProbability','shotPlayContinued','penaltyLength','timeBetweenEvents'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['shotPlayContinuedOutsideZone','shotPlayContinuedInZone','shotGeneratedRebound','shotPlayStopped'],axis=1,errors='ignore')
    trainingData.fillna(0,inplace=True)
    trainingData=pd.get_dummies(trainingData,columns=['shotType','team','location','shooterLeftRight','playerPositionThatDidEvent','lastEventCategory','lastEventTeam'])
    trainingData=trainingData.drop(['shotType_0','timeLeft'],axis=1,errors='ignore')

    testData=testData[(testData["shooterName"]==playerName) & (testData["isPlayoffGame"]==0)]
    testData=testData.drop(['teamCode','awayTeamCode','shotID','shooterName','gameOver','game_id','goalieNameForShot','homeTeamCode'],axis=1,errors='ignore')
    testData=testData.drop(['homeWinProbability','homeTeamWon','id','isPlayoffGame','playerNumThatDidEvent','playerNumThatDidLastEvent'],axis=1,errors='ignore')
    testData=testData.drop(['playoffGame','roadTeamCode','season','shooterPlayerId','wentToOT','wentToShootout','xFroze','xGoal','xPlayContinuedInZone','xPlayContinuedOutsideZone'],axis=1,errors='ignore')
    testData=testData.drop(['xPlayStopped','xRebound','xShotWasOnGoal','event','homeTeamScore'],axis=1,errors='ignore')
    testData=testData.drop(['shotPlayContinuedOutsideZone','shotPlayContinuedInZone','shotGeneratedRebound','shotPlayStopped'],axis=1,errors='ignore')
    testData.fillna(0,inplace=True)
    testData=pd.get_dummies(testData,columns=['shotType','team','location','shooterLeftRight','playerPositionThatDidEvent','lastEventCategory','lastEventTeam'])
    testData=testData.drop(['shotType_0'],axis=1,errors='ignore')
    testData = testData.reindex(columns=trainingData.columns)

    testData.fillna(0,inplace=True)
    trainingData.fillna(0,inplace=True)
    model = LogisticRegression(max_iter=50000)
    X1 = trainingData.loc[:, ~trainingData.columns.str.contains('goal')]
    Y1 = trainingData['goal']
    model.fit(X1,Y1)
    predictions = np.array(model.predict(testData.loc[:, ~testData.columns.str.contains('goal')]))
    actual = np.array(testData['goal'])


    predictions_count = np.count_nonzero(predictions == 1)
    actual_count = np.count_nonzero(actual == 1)

    predictions_countList.append(predictions_count)
    actual_countList.append(actual_count)
    moneyPuckxGList.append(moneyPuckxG)
    stattrickxGList.append(stattrickxG)

plt.figure('Expected Goals Scatter Plot')
plt.scatter(playerList, predictions_countList, color='red', label='My Model xGoal Count')
plt.scatter(playerList, actual_countList, color='blue', label='Actual Goal Count')
plt.scatter(playerList, moneyPuckxGList, color='green', label='Money Puck xGoal Count')
plt.scatter(playerList, stattrickxGList, color='purple', label='Natural Stat Trick xGoal Count')


# Add labels and title
plt.xlabel('Player')
plt.ylabel('Goal Count')
plt.title('Scatter Plot of Player Goals and xGoals (2022-2023 Regular Season)')
plt.legend()  # Show legend with labels

plt.xticks(rotation=45)

plt.tight_layout()


plt.figure('Model Comparison')
errorMyModel = np.average([round(abs((p-a)/a)*100,2) for p,a in zip(predictions_countList,actual_countList) ] )
errorMoneyPuck = np.average([round(abs((p-a)/a)*100,2) for p,a in zip(moneyPuckxGList,actual_countList) ] )
errorstattrick = np.average([round(abs((p-a)/a)*100,2) for p,a in zip(stattrickxGList,actual_countList) ] )


sizes1 = [errorMyModel,100-errorMyModel]
colors1 = ['grey','red']

sizes2 = [errorMoneyPuck,100-errorMoneyPuck]
colors2 = ['grey','green']

sizes3 = [errorstattrick,100-errorstattrick]
colors3 = ['grey','purple']

plt.subplot(1, 3, 1)  
plt.pie(sizes1,labels=['Percent Error','Accuracy'], colors=colors1, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  
plt.title('My Model')

plt.subplot(1, 3, 2)  
plt.pie(sizes2,labels=['Percent Error','Accuracy'], colors=colors2, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  
plt.title('Money Puck Model')

plt.subplot(1, 3, 3)  
plt.pie(sizes3,labels=['Percent Error','Accuracy'], colors=colors3, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  
plt.title('Natural Stat Trick Model')

plt.tight_layout()
plt.show()

