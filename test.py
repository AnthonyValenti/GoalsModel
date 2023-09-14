import pandas as pd 



shots2020 = pd.read_csv("shots_2020.csv")
shots2021 = pd.read_csv("shots_2021.csv")
trainingData = pd.concat([shots2020,shots2021])
trainingData=trainingData.drop(['teamCode','awayTeamCode','shotID','shooterName','gameOver','game_id','goalieNameForShot','homeTeamCode'],axis=1,errors='ignore')
trainingData=trainingData.drop(['homeWinProbability','homeTeamWon','id','isPlayoffGame','playerNumThatDidEvent','playerNumThatDidLastEvent'],axis=1,errors='ignore')
trainingData=trainingData.drop(['playoffGame','roadTeamCode','season','shooterPlayerId','wentToOT','wentToShootout','xFroze','xGoal','xPlayContinuedInZone','xPlayContinuedOutsideZone','homeTeamScore'],axis=1,errors='ignore')
trainingData=trainingData.drop(['xPlayStopped','xRebound','xShotWasOnGoal','event','roadTeamScore','shotGoalProbability','shotPlayContinued','penaltyLength','timeBetweenEvents'],axis=1,errors='ignore')
trainingData=trainingData.drop(['shotPlayContinuedOutsideZone','shotPlayContinuedInZone','shotGeneratedRebound','shotPlayStopped'],axis=1,errors='ignore')
trainingData.fillna(0,inplace=True)
trainingData=pd.get_dummies(trainingData,columns=['shotType','team','location','shooterLeftRight','playerPositionThatDidEvent','lastEventCategory','lastEventTeam'])
trainingData=trainingData.drop(['shotType_0','timeLeft'],axis=1,errors='ignore')
listData=trainingData.columns.to_list()

for data in listData:
    print(data)