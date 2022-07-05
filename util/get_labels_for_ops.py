import pandas as pd
import re
import argparse

def get_labels_for_three_op(file_destination,file_name):

    with open(file_destination + file_name,'r') as f:
        temp = [line.strip() for line in f.readlines()]
    
    df = pd.DataFrame(temp)
    df['label'] = ''
    df.rename(columns = {0:'Operation'}, inplace = True)
    
    for i, value in enumerate(df.values):
        sequence = value[0]
        string=re.sub('[^a-zA-Z]+', '', sequence)
        label = ','.join(string[i:i+2] for i in range(0,len(string),2))
        
        df.at[i,'label'] = label
        
    df.to_csv(file_destination + '/expressions_with_labels.csv')
    print('File Created and stored at : ' + file_destination + 'expressions_with_labels.csv')

def get_labels_for_one_op(file_destination,file_name):
    
    label_dictionary = {'(CY,CU)':0,'(CY,CY)':1,'(SP,CU)':2,'(SP,SP)':3,'(CU,CU)':4,'(SP,CY)':5}

    with open(file_destination + file_name,'r') as f:
        temp = [line.strip() for line in f.readlines()]
    
    df = pd.DataFrame(temp)
    df['label'] = ''
    df['label_encoding'] = ''
    df.rename(columns = {0:'Operation'}, inplace = True)
        
    for i,value in enumerate(df.values):
        sequence = value[0]

        if sequence.find('cu')!=-1 and sequence.find('cy')!=-1:
            label = '(CY,CU)'
            label_encoding = 0
        elif sequence.find('sp')!=-1 and sequence.find('cy')!=-1:
            label = '(SP,CY)'
            label_encoding = 5
        elif sequence.find('cu')!=-1 and sequence.find('sp')!=-1:
            label = '(SP,CU)'
            label_encoding = 2
        elif sequence.find('cu')!=-1 and sequence.find('cy')==-1 and sequence.find('sp')==-1:
            label = '(CU,CU)'
            label_encoding = 4
        elif sequence.find('cu')==-1 and sequence.find('cy')!=-1 and sequence.find('sp')==-1:
            label = '(CY,CY)'
            label_encoding = 1
        elif sequence.find('cu')==-1 and sequence.find('cy')==-1 and sequence.find('sp')!=-1:
            label = '(SP,SP)'
            label_encoding = 3
    
        df.at[i,'label'] = label
        df.at[i,'label_encoding'] = label_encoding
    
    df.to_csv(file_destination + '/expressions_with_labels.csv')
    print('File Created and stored at : ' + file_destination + 'expressions_with_labels.csv')


def main(args):

    if args.operation == 'one_op':
        get_labels_for_one_op(args.file_destination,args.file_name)
    elif args.operation == 'two_ops':
        print('Get labels for two ops.')
    elif args.operation == 'three_ops':
        get_labels_for_three_op(args.file_destination, args.file_name)



if __name__=='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--operation', type=str, default='one_op', help='For choosing the the operation for which labels need to be fetched. : one_op,two_ops,three_ops')
	parser.add_argument('--file_destination', type=str, default='../data/one_op/', help='Path for saving the labels of operations.')
	parser.add_argument('--file_name', type=str, default='expressions.txt')

	args = parser.parse_args()

	main(args)