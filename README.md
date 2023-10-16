# Expected Goals Model

Using NHL shot data dating back to 2016 I created a logistic regression model to predict if any given shot will result in a goal or not. The dataset contains over 100 features and was downloaded from Moneypuck.com. My model differs from other public models by predicting goals based only on shots taken by the individual shooter, instead of all shots in the league. The rational behind this is that shooters in the NHL have vastly different career shooting percentages (due to shot selection, ability and more) so a model predicting if a certain players shot is a goal should be trained using only that players past shots. This model works best for players with high shot volume, and will tend to over-fit players with low-shot totals and high shooting percentages.

# Results
![Screenshot 2023-10-16 at 2 45 30 PM](https://github.com/AnthonyValenti/GoalsModel/assets/57304403/81431437-9314-4522-bd2a-125be9ed7412)

This graph dispalys my models predictions for the top 30 goal scorers in the 2022-2023 NHL season and compares it against their actual goal count as well as other publical models.

# Model Comparison
![Screenshot 2023-10-16 at 2 47 09 PM](https://github.com/AnthonyValenti/GoalsModel/assets/57304403/282f8e5c-4340-4a9e-9210-84715d90ca3c)

My model was able to more accurately predict goals from the top 30 goals scoeres in the 2022-2023 when compared against the most popular public models.


