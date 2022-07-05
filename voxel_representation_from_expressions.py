import deepdish as dd
import pandas as pd
from src.Utils.train_utils import voxels_from_expressions
import matplotlib.pyplot as plt
import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser
import argparse

def main(args):
    # pre-rendered shape primitives in the form of voxels for better performance
    primitives = dd.io.load(args.primitive_file)
    
    data = pd.read_csv(args.voxel_expression_path)

    data.drop('Unnamed: 0', inplace=True, axis=1)
    
    for j,i in enumerate(data.values):
        expression = [i[0]]
        voxels = voxels_from_expressions(expression, primitives, max_len=7)
        voxels = voxels.reshape((64,64,64))

        vox = Vox.from_dense(voxels)
        VoxWriter(args.voxel_data_path + args.voxel_file_naming +str(j)+'.vox', vox).write()


if __name__=="__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--voxel_expression_path', type=str, default='./data/one_op/expressions_with_labels.csv', help='Path for fetching voxel expressions for x_ops (x - one, two, three)')
	parser.add_argument('--voxel_file_naming', type=str, default='voxel_', help='Set the file name according to the user requirement (eg. voxel_ : voxel_0, voxel_1, voxel_2,...)')
	parser.add_argument('--voxel_data_path', type=str, default='./data/one_op/voxel_representation/', help='Path for saving the voxel files')
	parser.add_argument('--primitive_file', type=str,default='./data/primitives.h5')

	args = parser.parse_args()

	main(args)

