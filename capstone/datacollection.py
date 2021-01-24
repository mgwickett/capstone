import pandas as pd 
import numpy as np

pd.set_option('mode.chained_assignment', None)

wars = pd.read_csv("IntraStateWars.csv", encoding = 'latin-1')
constitutions = pd.read_csv("ConstitutionalEvents.csv", encoding = 'latin-1')

# total = wars[['WarName', 'WarType', 'CcodeA', 'SideA', 'SideB', 'StartYr1', 'EndYr1']].copy()

wars["FinalEnd"] = ""

for index, row in wars.iterrows(): 
    # find max end year 
    year_ends = row[['EndYr1', 'EndYr2', 'EndYr3', 'EndYr4']]
    end = year_ends.max(axis = 0)
    if (end < 0):
        wars.at[index, 'FinalEnd'] = 0
    else:
        wars.at[index, 'FinalEnd'] = end
    # print(row['FinalEnd'])



trimmed_wars = wars[['WarName', 'WarType', 'CcodeA', 'SideA', 'SideB', 'StartYr1', 'Outcome', 'FinalEnd']]
# trimmed_wars.to_csv('trimmedWars.csv')
trimmed_wars['CcodeA'] = trimmed_wars['CcodeA'].apply(str)
trimmed_wars['StartYr1'] = trimmed_wars['StartYr1'].apply(str)
trimmed_wars['FinalEnd'] = trimmed_wars['FinalEnd'].apply(str)
trimmed_wars["HelperStart"] = trimmed_wars[['CcodeA', 'StartYr1']].apply(lambda x: ''.join(x), axis=1)
trimmed_wars["HelperEnd"] = trimmed_wars[['CcodeA', 'FinalEnd']].apply(lambda x: ''.join(x), axis = 1)
trimmed_wars["HelperStart"] = trimmed_wars["HelperStart"].apply(int)
trimmed_wars["HelperEnd"] = trimmed_wars["HelperEnd"].apply(int)

constitutions['year'] = constitutions['year'].apply(str)
constitutions['cowcode'] = constitutions['cowcode'].apply(str)
constitutions['Helper'] = constitutions[['cowcode', 'year']].apply(lambda x: ''.join(x), axis = 1)
constitutions['Helper'] = constitutions['Helper'].apply(int)

constitutions['War'] = ""
constitutions['PostWar'] = ""

# now for the horrific part 
index, row = next(constitutions.iterrows())
c_year = row['Helper']
for i, r in trimmed_wars.iterrows():
    start = r['HelperStart']
    end = r['HelperEnd']
    a_end = end + 2
    print(start)
    print(end)
    print(c_year)
    if (start <= c_year <= end): 
        constitutions.at[index, 'War'] = 1
    else: 
        constitutions.at[index, 'War'] = 0
    if (end <= c_year <= a_end): 
        constitutions.at[index, 'PostWar'] = 1
    else: 
        constitutions.at[index, 'PostWar'] = 0

# constitutions.to_csv('combined.csv')
