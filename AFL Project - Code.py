#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Establishing the connection with AFLTable and extracting game data

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import re
from selenium.webdriver.common.by import By
from lxml import html
 
url = 'https://afltables.com/afl/seas/2022.html'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')
 
urls = []
for link in soup.find_all('a'):
    thelink = link.get('href')
    urls.append(thelink)

game_url = []
for i in urls:
    i = str(i)
    if i[0:20] == "../stats/games/2022/":
        game_url.append(i)
        
baseurl = 'afltables.com/afl'
gameurls = []
for i in game_url:
    i = i[2:]
    gameurl = "http://" + baseurl + i
    gameurls.append(gameurl)
    

driver = webdriver.Chrome()

driver.get('https://afltables.com/afl/seas/2022.html')
driver.maximize_window()


# In[2]:


# Retrieving team sheets from AFLTables

team_sheets = []

for url in gameurls:
    driver.get(url)
    
    home_team = driver.find_element("id", 'sortableTable0')
    away_team = driver.find_element("id", 'sortableTable1')
    
    temporary_data = {'Home Rating': home_team.text,
                     'Away Rating': away_team.text}
    team_sheets.append(temporary_data)


# In[3]:


df = pd.DataFrame(team_sheets)

# Step 1: Split the string into lines
for i, text in df['Home Rating'].iteritems():
    lines = text.split("\n")
    # Step 2: Iterate over the lines and extract the player names
    player_names = []
    for line in lines:
        match = re.search(r'^\d+\s([A-Za-z\s,]+)', line)
        if match:
            player_names.append(match.group(1))
    df['Home Rating'][i] = player_names
    
for i, text in df['Away Rating'].iteritems():
    lines = text.split("\n")
    # Step 2: Iterate over the lines and extract the player names
    player_names = []
    for line in lines:
        match = re.search(r'^\d+\s([A-Za-z\s,]+)', line)
        if match:
            player_names.append(match.group(1))
    df['Away Rating'][i] = player_names


# In[4]:


# Feature 1: Player Ratings from previous season (May look at making this dynamic using fantasy football)

all_ratings = pd.read_csv('2022_PR.csv')
all_ratings['Player'] = all_ratings['Player'].astype(str)
right = all_ratings

# Assigning values to the team sheets

for i in range(len(df['Home Rating'])):
    home_sheet = pd.DataFrame({'Player': df['Home Rating'][i]})
    left = home_sheet.astype(str)
    for x in range(len(left['Player'])):
        left['Player'][x] = left['Player'][x][:-1]

    home_rating = pd.merge(left, right, on = 'Player', how='inner')
    df['Home Rating'][i] = home_rating['Scaled Score'].mean()


    away_sheet = pd.DataFrame({'Player': df['Away Rating'][i]})
    left = away_sheet.astype(str)
    for x in range(len(left['Player'])):
        left['Player'][x] = left['Player'][x][:-1]

    away_rating = pd.merge(left, right, on = 'Player', how='inner')
    df['Away Rating'][i] = away_rating['Scaled Score'].mean()
    
df['Rating Differential'] = df['Home Rating'] - df['Away Rating']


# In[5]:


# Retrieving team names

from collections import defaultdict

team_sheets = pd.DataFrame(team_sheets)

temp = []
teams = []
home_team = []
away_team = []

for i,row in team_sheets.iterrows():
    for x in row:
        x = x[:20]
        temp.append(x)
        
for i in temp:
    i = i.split()
    i = i[:2]
    if i[1] == 'Match':
        i = i[0]
    if len(i) == 2:
        i = ' '.join(i)
    teams.append(i)

count = 0
for i in teams:
    if i == "Greater Western":
        i = "Greater Western Sydney"
    if count % 2 == 0:
        home_team.append(i)
        count += 1
    else:
        away_team.append(i)
        count += 1
fin_teams = pd.DataFrame({"Home Team": home_team, "Away Team": away_team})
train_data = pd.concat([fin_teams, df], axis=1)


# In[6]:


# Feature 2+3: Incorporating prev season % Metric and Ladder Position

train_data = pd.concat([fin_teams, df], axis=1)

prev_finish_home = pd.read_csv('2021_Ladder.csv')
prev_finish_away = pd.read_csv('2021_Ladder_away.csv')

right_home = prev_finish_home.set_index('Home Team')[['Position Last Season', 'Percentage Last Season']].add_suffix('_prev')
right_away = prev_finish_away.set_index('Away Team')[['Away Position Last Season', 'Away Percentage Last Season']]

train_data = train_data.join(right_home, on='Home Team', how='outer')
train_data = train_data.join(right_away, on='Away Team', how='outer')

train_data['Position Differential Last Season'] = train_data['Position Last Season_prev'] - train_data['Away Position Last Season'] 

train_data['Percentage Differential Last Season'] = train_data['Percentage Last Season_prev'] - train_data['Away Percentage Last Season']

train_data = train_data.drop(['Position Last Season_prev', 'Percentage Last Season_prev', 'Away Position Last Season', 
                              'Away Percentage Last Season'], axis=1)

# Retrieving home team outcome. Will be categorising as Big Win, 
# Little Win, Little Loss, Big Loss

results = []
for url in gameurls:

    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table element
    table = soup.find('table')

    # Extract the data from each row
    for row in table.find_all('tr'):
        # Extract the data from each cell in the row
        cells = [cell.text for cell in row.find_all('td')]
        cells = cells[4:]
        results.append(cells)
        
        
fin_score = []
winners = []
index = 0
for i in results[3::6]:
    index += 1
    fin_score.append(i)

replacements = {'GE': 'Geelong', 'ME': 'Melbourne', 'HW': 'Hawthorn',
               'GC': 'Gold Coast', 'CA': 'Carlton', 'SY': 'Sydney',
               'CW': 'Collingwood', 'BL': 'Brisbane Lions',
                'FR': 'Fremantle', 'AD': 'Adelaide', 'PA': 'Port Adelaide',
               'SK': 'St Kilda', 'GW': 'Greater Western Sydney', 'RI': 'Richmond',
               'WC': 'West Coast', 'NM': 'North Melbourne', 'WB': 'Western Bulldogs',
               'ES': 'Essendon'}


for i in range(len(fin_score)):
    for j in range(len(fin_score[i])):
        words = fin_score[i][j].split(' by ')
        winners.append(words)
        for k in range(len(words)):
            if words[k] in replacements:
                words[k] = replacements[words[k]]
                
winners = pd.DataFrame(winners, columns = ['Winner', 'Margin'])

train_data = train_data.join(winners)

train_data['Result'] = pd.Series(dtype = 'str')


# In[7]:


train_data #NaN result is okay for now, we will add the outcome after


# In[8]:


# Adding the result to each game
train_data['Margin'] = pd.to_numeric(train_data['Margin'])

def get_results(row):
    if row['Winner'] == row['Home Team']:
        if row['Margin'] >= 39:
            return "Big Win"
        else:
            return "Small Win"
    elif row['Winner'] != row['Home Team']:
        if row['Margin'] >= 39:
            return "Big Loss"
        else:
            return "Small Loss"
    else:
        return 'Draw'

train_data['Result'] = train_data.apply(get_results, axis=1)


# In[9]:


df = train_data
df


# In[10]:


# Setting up sampling techniques and splitting data up into training and test
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit


ss = ShuffleSplit(n_splits=10, test_size=15, random_state=4)
kf = KFold(n_splits=10)
skf = StratifiedKFold(n_splits=10)

array = df.values
X1 = array[:,0:9] #This should be columns take 1
y1 = df['Result']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,
                                                    test_size=0.10, # use a teste sieve of 25%
                                                    random_state=4)


# In[11]:


# Test 1 - Decision Tree Classifier
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# encode categorical variables using one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(df.drop(columns=['Result']))
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# train a decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

# make predictions on the test set and calculate accuracy
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[12]:


# Test 2: XGBoost - CURRENTLY UNFINISHED
from sklearn.model_selection import train_test_split
import xgboost as xgb

# convert result column to binary
df['Result'] = df['Result'].apply(lambda x: 1 if x in ['Big Win', 'Small Win'] else 0)

# split into features and target variable
X = df.drop('Result', axis=1)
y = df['Result']

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define xgboost DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# define xgboost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# train xgboost model
xgb_model.fit(dtrain)

# make predictions on the test set and calculate accuracy
y_pred = xgb_model.predict(dtest)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy}")


# In[ ]:




