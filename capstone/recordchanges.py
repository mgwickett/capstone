import numpy as np
import pandas as pd 

const_changes = pd.read_csv("const_events_v3.csv", error_bad_lines = False, dtype = str)
consts = pd.read_csv("ccpcnc_v2.csv", error_bad_lines = False, dtype = str)

consts = consts[consts.columns.drop(list(consts.filter(regex = '_comments')))]
consts = consts[consts.columns.drop(list(consts.filter(regex = '_article')))]

consts = consts.fillna(-1)

colnames = ["cowcode", "country", "year", "evnttype", "war", "postwar", "outcome", "changed_val", "old_val", "new_val"]
output = pd.DataFrame(columns = colnames)



ignore_changes = ["helper", "year", "evnt", "evntid", "evnttype", "length", "evntyear", "source", "langsrce", "translat", "doctit", "syst", "systid", "systyear"]

match = False




for index, row in const_changes.iterrows():
    event = row
    key_code = int(event['Helper'])

    old_const = consts[consts['helper'] == str(key_code - 1)]
    new_const = consts[consts['helper'] == str(key_code)] 

   
    for label, content in old_const.iteritems():
        # inserting biographical information in case there is a new row 
        if label in ignore_changes:
            continue
        
        cowcode = event['cowcode']
        country = event['country']
        year = event['year']
        evnttype = event['evnttype']
        war = event['War']
        postwar = event['PostWar']
        outcome = event['Outcome']

        old = old_const[label]
        new = new_const[label]

  

        if old.empty or new.empty:
            continue

        if old.values[0] == -1 or new.values[0] == -1:
            continue
        
        if old.values[0] == new.values[0]:
            match = True
        else:
            cv = label
            ov = old.values[0]
            nv = new.values[0]
            nr = np.array([cowcode, country, year, evnttype, war, postwar, outcome, cv, ov, nv])
            nr = np.reshape(nr, (1, 10))
            newrow = pd.DataFrame(nr, columns = colnames)
            print(newrow)
            output = output.append(newrow, ignore_index = True) 

output.to_csv("output_v4.csv")
