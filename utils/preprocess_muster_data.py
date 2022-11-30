import os
import pandas as pd
from os.path import exists

def read_file(file_path, data_path):
    Musterdaten = pd.read_excel(file_path)
    Musterdaten.pop('AG-Nr')
    if data_path != '../data_vorplan/17002-211920':
        Musterdaten.pop('2D_3D bereitgestellt')
    #Musterdaten.pop('Auftrags-Pos')
    # Musterdaten.pop('Benennung')
    return Musterdaten


def get_data(file_path, data_path, vocab_dict, flag=True):
    
    df_musterdaten = read_file(file_path, data_path)
    Vorgänge = df_musterdaten['Arbeitsgang Bezeichnung'].unique()
    
    if data_path == '../data_vorplan/17002-211920':
        df_musterdaten.drop(df_musterdaten.index[df_musterdaten['Benennung'] == 'Beispieldatensatz'], inplace=True)
        df_musterdaten.reset_index(inplace=True,drop=True)
        
        df_musterdaten['file_number'] = df_musterdaten['Auftrags-Nr.'].astype(str)+'_'+ df_musterdaten['Auftrags-Pos.'].astype(str).str.zfill(5)
        df_musterdaten.pop('Auftrags-Pos.')
        df_musterdaten.pop('Auftrags-Nr.')
    else:
        df_musterdaten['file_number'] = df_musterdaten['Feinkalk-Nr.'].astype(str) + '_' + df_musterdaten['Auftrags-Pos'].astype(str)
        df_musterdaten.pop('Auftrags-Pos')
        df_musterdaten.pop('Feinkalk-Nr.')
        
    preprocessed_df = df_musterdaten.groupby('file_number')[["Arbeitsgang Bezeichnung"]].agg(lambda x: ['<start>']+list(x)+['<end>'])
    preprocessed_df.reset_index(inplace=True)
        
        
    if flag:
        vocab_dict = {Vorgänge[i]:i+2 for i in range(len(Vorgänge))}
        vocab_dict['<start>'] = 0
        vocab_dict['<end>'] = 1
        vocab_dict['<pad>'] = 99
    else:
        temp = {Vorgänge[i]:i+2 for i in range(len(Vorgänge))}
        j=sorted(vocab_dict.values())[-2]
        for i in temp:    
            if i in vocab_dict:
                continue
            else:
                j+=1
                vocab_dict[i] = j
    
    preprocessed_df['encoded_labels'] = preprocessed_df['Arbeitsgang Bezeichnung'].apply(lambda x: [vocab_dict[elem] for elem in x])
    
    return preprocessed_df, vocab_dict

def is_File(file_path, data_path):
    
    df_dataset = pd.read_csv(file_path)
    
    df = df_dataset['file_number'] + '.binvox'
    
    j=0
    flag = True
    for i in df:
        if not exists(data_path + i):
            if flag:
                print('Files that doesn\'t exists : ')
                flag=False
            j+=1
            print(' ' + str(j) + '. ' + i)


if __name__ == '__main__':

	data_path_1 = '../data_vorplan/1160035-1160712'
	excel_file_path_1 = '../data_vorplan/1160035-1160712/Musterdaten_KWS_220524.xlsx'

	data_path_2 = '../data_vorplan/1160738-1161972'
	excel_file_path_2 = '../data_vorplan/1160738-1161972/Musterdaten_KWS_220627.xlsx'

	data_path_3 = '../data_vorplan/17002-211920'
	excel_file_path_3 = '../data_vorplan/17002-211920/Beispiele_Arbeitspläne_211117.xlsx'

    data_path_4 = '../data_vorplan/1152033-1211207'
    excel_file_path_4 = '../data_vorplan/1152033-1211207/1152033-1211207/Musterdaten_KWS_220706.xlsx'

	vocab_dict = dict()

	pre_processed_data_1, vocab_dict = get_data(excel_file_path_1, data_path_1, vocab_dict)
	pre_processed_data_2, vocab_dict = get_data(excel_file_path_2, data_path_2, vocab_dict, False)
	pre_processed_data_3, vocab_dict = get_data(excel_file_path_3, data_path_3, vocab_dict, False)
    pre_processed_data_4, vocab_dict = get_data(excel_file_path_4, data_path_4, vocab_dict, False)

	preprocessed_df = pre_processed_data_1.append(pre_processed_data_2, ignore_index=True).append(pre_processed_data_3.append(pre_processed_data_4,ignore_index=True), ignore_index=True)
	
	try:
		preprocessed_df.to_csv('../data_vorplan/preprocessed_df.csv',index=False)
		print('\nCSV file containing all Voxels data is saved !!!\n')
		
		is_File('../data_vorplan/preprocessed_df.csv', '../data_vorplan/Voxels/')
	except:
		print('Error !!!')

