import pandas as pd
import numpy as np

if __name__ == '__main__':

	label_encoding_dictionary = {'<start>':0,'sp':2,'cu':3,'cy':4,'<end>':1} 

	# Read expressions file of three_ops
	data = pd.read_csv('../data/three_ops/expressions_with_labels.csv')
	data.drop(['Unnamed: 0'], inplace=True, axis=1)

	data['encoded_label']=''

	# add encoding on the basis of the defined dictionary.
	for i, j in enumerate(data.values):
	    label = j[1].split(',')

	    label_encoding = [0]
	    for k in label:
	        label_encoding.append(label_encoding_dictionary[k])

	    label_encoding.append(1)
	    data.at[i,'encoded_label'] = label_encoding

	# Save the encoded data.
	data.to_csv('../data/three_ops/label_with_encoding.csv')