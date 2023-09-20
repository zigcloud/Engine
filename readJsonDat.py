import os
import json

folders = [f for f in os.listdir('./Resources/jsons') if os.path.isdir(os.path.join('./Resources/jsons',f))]

files = []
for folder in folders:
    order = int(folder.split('_')[-1])
    for file in os.listdir(os.path.join('./Resources/jsons',folder)):
        if file.endswith('json'):
            files.append([os.path.join('./Resources/jsons',folder,file),order])

files.sort(key=lambda x:x[1])
files = [f[0] for f in files]

summaryJson = {}

for file in files:
    with open(file,'r') as js:
        data = json.load(js)
    for key, value in data.items():
        summaryJson.update({key: value})

with open('./Resources/summaryPhaseCurveTable.json', 'w') as sum:
    json.dump(summaryJson, sum, indent=3)
