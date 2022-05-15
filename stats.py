import pandas as pd
import pdb
import math

pdb.set_trace()
train = pd.read_csv("IAM/complete_monk/all_trainLabels - Copy.csv", header = 0)

for i in range(0,len(train)):
	try:
		if math.isnan(train.loc[i]['Label']):
			train.loc[i]['Label'] = ''
			train.loc[i]['Length'] = 0
			continue
	except:
		pass
	train.loc[i]['Label'] = train.loc[i]['Label'].lstrip()
	train.loc[i]['Label'] = train.loc[i]['Label'].rstrip()
	train.loc[i]['Length'] = len(train.loc[i]['Label'])

train.to_csv("IAM/complete_monk/all_trainLabels_filter.csv")  
