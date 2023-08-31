import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
scaler = StandardScaler()

shots2016 = pd.read_csv("shots_2016.csv")
shots2017 = pd.read_csv("shots_2017.csv")
shots2018 = pd.read_csv("shots_2018.csv")
shots2019 = pd.read_csv("shots_2019.csv")
shots2020 = pd.read_csv("shots_2020.csv")
shots2021 = pd.read_csv("shots_2021.csv")
trainingData = pd.concat([shots2016,shots2017,shots2018,shots2019,shots2020,shots2021])
testData = pd.read_csv("shots_2022.csv")

playerList = ["Auston Matthews","John Tavares","William Nylander","Mitchell Marner","Calle Jarnkrok","Ryan Reaves","Morgan Rielly","John Klingberg","Tyler Bertuzzi","Max Domi"]
predictions_countList=list()
actual_countList=list()
moneyPuckxGList = list()

for playerName in playerList:
    trainingData = pd.concat([shots2016,shots2017,shots2018,shots2019,shots2020,shots2021])
    testData = pd.read_csv("shots_2022.csv")
    moneyPuck = testData[testData["shooterName"] == playerName]
    moneyPuckxG = round(moneyPuck["xGoal"].sum(),1)


    trainingData = trainingData[trainingData["shooterName"] == playerName]
    trainingData=trainingData.drop(['teamCode','awayTeamCode','shotID','shooterName','gameOver','game_id','goalieIdForShot','goalieNameForShot','homeTeamCode','playerPositionThatDidEvent'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['homeWinProbability','homeTeamWon','id','isPlayoffGame','playerNumThatDidEvent','playerNumThatDidLastEvent'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['playoffGame','roadTeamCode','season','shooterLeftRight','shooterPlayerId','wentToOT','wentToShootout','xFroze','xGoal','xPlayContinuedInZone','xPlayContinuedOutsideZone','homeTeamScore'],axis=1,errors='ignore')
    trainingData=trainingData.drop(['xPlayStopped','xRebound','xShotWasOnGoal','event','lastEventCategory','lastEventTeam','roadTeamScore','shotGoalProbability','shotPlayContinued','penaltyLength','timeBetweenEvents'],axis=1,errors='ignore')
    trainingData.fillna(0,inplace=True)
    trainingData=pd.get_dummies(trainingData,columns=['shotType','team','location'])
    trainingData=trainingData.drop(['shotType_0','timeLeft'],axis=1,errors='ignore')


    testData = testData[testData["shooterName"] == playerName]
    testData=testData.drop(['teamCode','awayTeamCode','shotID','shooterName','gameOver','game_id','goalieIdForShot','goalieNameForShot','homeTeamCode','playerPositionThatDidEvent'],axis=1,errors='ignore')
    testData=testData.drop(['homeWinProbability','homeTeamWon','id','isPlayoffGame','playerNumThatDidEvent','playerNumThatDidLastEvent'],axis=1,errors='ignore')
    testData=testData.drop(['playoffGame','roadTeamCode','season','shooterLeftRight','shooterPlayerId','wentToOT','wentToShootout','xFroze','xGoal','xPlayContinuedInZone','xPlayContinuedOutsideZone'],axis=1,errors='ignore')
    testData=testData.drop(['xPlayStopped','xRebound','xShotWasOnGoal','event','homeTeamScore','lastEventCategory','lastEventTeam'],axis=1,errors='ignore')
    testData.fillna(0,inplace=True)
    testData=pd.get_dummies(testData,columns=['shotType','team','location'])
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

plt.scatter(playerList, predictions_countList, color='red', label='My Model xGoal Count')
plt.scatter(playerList, actual_countList, color='blue', label='Actual Goal Count')
plt.scatter(playerList, moneyPuckxGList, color='green', label='Money Puck xGoal Count')

# Add labels and title
plt.xlabel('Player')
plt.ylabel('Goal Count')
plt.title('Scatter Plot of Player Goals and xGoals (2022-2023 Regualr and Post Season)')
plt.legend()  # Show legend with labels

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()







