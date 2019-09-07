#!/usr/bin/python3

import pandas as pd
from scipy import stats
import numpy as np

def preprocess(file_1, file_2):

	# Load files
	df_1 = pd.read_csv(file_1, index_col = False, nrows = 1000)
	df_2 = pd.read_csv(file_2, index_col = False, nrows = 1000)

	# Transpose and remove column label
	df_1 = df_1.transpose()
	df_1.drop(df_1.index[0], inplace = True)
	df_2 = df_2.transpose()
	df_2.drop(df_2.index[0], inplace = True)

	# Set class for each file
	df_1['class'] = 1
	df_2['class'] = 2

	# Concat into one matrix for further preprocessing and classification
	df = pd.concat([df_1, df_2])

	print("Table dimensions: " + str(df.shape))
	print("\tTable 1 dimensions: " + str(df_1.shape))
	print("\tTable 2 dimensions: " + str(df_2.shape))

	# Remove rows with only 0 values, don't add class value into sum
	df.drop(df[df.iloc[:, :-1].sum(axis=1) == 0].index, inplace = True)
	
	print("Table dimensions after removing 0s: " + str(df.shape))

	return df

def main():
	file_class_1 = "./data/024_Single-cell_RNA-seq_of_six_thousand _purified_CD3+_T_cells_from _human_primary_TNBCs_csv.csv";
	file_class_2 = "./data/025_Single-cell_RNA-seq_of_six_thousand _purified_CD3+_T_cells_from _human_primary_TNBCs_csv.csv";

	df = preprocess(file_class_1, file_class_2)

if __name__ == "__main__":
	main()