
'''
Convert Yelp Academic Dataset from JSON to CSV
@author: Son
'''
 
import json
import pandas as pd
   
    
def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if k == 'text':
            ob[k] = ''.join(v)
        else:
            del ob[k]

    return ob
 
json_filename = 'yelp_academic_dataset_review.json';
csv_filename = 'review.csv'
file = open(json_filename)
print('Converting.....')
df = pd.DataFrame([convert(line) for line in file.readlines()])
df.to_csv(csv_filename, encoding='utf-8', index=False)  