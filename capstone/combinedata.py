import numpy as np
import pandas as pd 

const_changes = pd.read_csv("const_events_v3.csv", error_bad_lines = False, dtype = str)
consts = pd.read_csv("ccpcnc_v2.csv", error_bad_lines = False, dtype = str)

consts = consts[consts.columns.drop(list(consts.filter(regex = '_comments')))]
consts = consts[consts.columns.drop(list(consts.filter(regex = '_article')))]

consts = consts.fillna(-1)

colnames = ["cowcode", "country", "year", "evnttype", "war", "postwar", "outcome", "changed_val", "old_val", "new_val"]
output = pd.DataFrame(columns = colnames)


# let's just do a proof of concept with USA 

# TO DO
# make a for loop with all civil war const changes
# create "well duh" list to check below 
# well duh = helper, year, evnt, evntid, evnttype, length 

ignore_changes = ["helper", "year", "evnt", "evntid", "evnttype", "length", "evntyear", "source", "langsrce", "translat", "doctit", "syst", "systid", "systyear"]

match = False


# key_code = 21865
# event = const_changes[const_changes['Helper'] == str(key_code)]
# old_const = consts[consts['helper'] == str(key_code - 1)]
# new_const = consts[consts['helper'] == str(key_code)]


for index, row in const_changes.iterrows():
    event = row
    key_code = int(event['Helper'])

    old_const = consts[consts['helper'] == str(key_code - 1)]
    new_const = consts[consts['helper'] == str(key_code)] 

    # print(event['country'])
    # print(old_const["evnttype"])
    

    for label, content in old_const.iteritems():
        # inserting biographical information in case there is a new row 
        if label in ignore_changes:
            continue
        
        # newrow = pd.DataFrame(columns=colnames)

        cowcode = event['cowcode']
        country = event['country']
        year = event['year']
        evnttype = event['evnttype']
        war = event['War']
        postwar = event['PostWar']
        outcome = event['Outcome']

        old = old_const[label]
        new = new_const[label]

        # print(newrow['country'])

        # print(old)
        # print(new)

        if old.empty or new.empty:
            continue

        if old.values[0] == -1 or new.values[0] == -1:
            continue
        
        # print(old.values[0])

        if old.values[0] == new.values[0]:
            match = True
        else:
            # don't add if the changed value is in "well duh" list
            cv = label
            ov = old.values[0]
            nv = new.values[0]
            nr = np.array([cowcode, country, year, evnttype, war, postwar, outcome, cv, ov, nv])
            nr = np.reshape(nr, (1, 10))
            newrow = pd.DataFrame(nr, columns = colnames)
            print(newrow)
            output = output.append(newrow, ignore_index = True) 
            # print(output)

output.to_csv("output_v4.csv")