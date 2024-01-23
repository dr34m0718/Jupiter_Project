#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Creating Moons class
class Moons():
	#function to intiate the class instance
	def __init__(self, db_file = './jupiter.db', metadata = None):
		self.db_file = db_file
		self.metadata = metadata
		self.data = self.load_data()

	#function to load dataset from the SQL Database file into Pandas Dataframe
	def load_data(self):
		con = f"sqlite:///{self.db_file}"
		df = pd.read_sql_query("SELECT * from moons", con)
		return df

	#function to check the columns of data
	def columns(self):
		return(self.data.columns)

	#function to drop columns that are not needed
	def drop_columns(self, columns):
		self.data = self.data.drop(columns)

	#function to remove all rows with missing values
	def drop_na(self):
		self.data.dropna(inplace = True)

	#function to ensure data is stored in ascending order according to moon name
	def rearrange_data_by_col(self, column):
		self.data = self.data.sort_values(by = column).reset_index(drop = True)

	#funtion to add a new moon
	def add_moon(self, new_moon_data):
		new_data = pd.DataFrame(new_moon_data)
		self.data = pd.concat([self.data,new_data], ignore_index = True)
		self.rearrange_data_by_col('moon')

	#function to select moon according to given name
	def select_moon_by_name(self, moon):
		moon_data = self.data.loc[self.data['moon'] == moon]
		if moon_data.empty:
			print(f'No data found for {moon}.')
		else:
			return moon_data

	#function to select moon according to index
	def select_moon_by_idx(self, moon_idx):
		#check if the index given is out of range
		if moon_idx < 0 or moon_idx > len(self.data):
			raise IndexError(f'The index given is out of range.')
		else:
			moon_data = self.data.iloc[moon_idx]
			return moon_data

	#function to select required moons
	def select_data(self, moon):
		complete_data = pd.DataFrame()
		#check if moon is given by name
		if type(moon[0]) is str:
			for moon_name in moon:
				moon_data = self.select_moon_by_name(moon_name)
				if not moon_data.empty:
					complete_data = pd.concat([complete_data, moon_data])
		else:
			complete_data = self.select_moon_by_idx(moon)
		#ensure that the data return is remained in Moon class
		selected_moon = Moons()
		selected_moon.data = complete_data
		return selected_moon

	#function to print the moons and data
	def print_data(self, moon = []):
		if not moon:
			print(self.data)
		else:
			selected_moon_df = select.select_data(moon)
			print(selected_moon_df)

	#function for summary statistics
	def summary(self):
		return self.data.describe()

	#function to check the numbers of null values in each column
	def check_na(self):
		return self.data.isnull().sum()

	#function to check correlation between variables
	def correlation(self):
		return self.data.corr()

	#function to plot heatmap for the correlation matrix
	def plot_corr_matrix(self):
		corr = self.correlation()
		#set up figure
		f, ax = plt.subplots((figsize = (6,4))
		#choose a custom diverging colormap
		cmap = sns.diverging_palette(220,20, as_cmap = True)
		#plot the heatmap
		sns.heatmap(corr, cmap = cmap)

	#function to find minimum of selected column
	def max_min_by_col(self, column, type = None):
		#check if max or min is required
		if type == 'min'
			row_idx = self.data[column].idxmin()
		elif type == 'max':
			row_idx = self.data[column].idxmax()
		else:
			raise ValueError("Please specify type = 'max' or 'min'")
		return self.select_moon_by_idx(row_idx)

	#function to plot different graphs
	def plot(self, x = None, y = None, hue = None, plot = 'scatter'):
		#following is to plot a scatter plot
		if plot == 'scatter':
			if x and y is None:
				print("Datas to be plotted are not stated.")
			else:
				f, ax = plt.subplots(figsize = (8,4))
				sns.scatterplot(self.data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a histogram
		elif plot == 'histogram':
			if x is None:
				print("Data to be plot is not stated.")
			else:
				f, ax = plt.subplots(figsize = (8,5))
				sns.histplot(self.data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a boxplot
		elif plot == 'boxplot':
			if x is None:
				print("Data to be plotted is not stated.")
			else:
				f, ax = plt.subplots(figsize = (6,4))
				sns.boxplot(self.data, x = x, y = y, hue = hue, palette = 'pastel')
		#if plot is not available, raise an error
		else:
			raise ValueError(f"The plot requested -- {plot} is not available. (Try 'scatter', 'histogram' or 'boxplot'.)")

	#function to make a simple summary for the grouped analysis
	def grouped_analysis_summary(self, column = 'group'):
		grouped = self.data.groupby(column)
		group_count = grouped.size()
		print(f"Group count by {column} :\n {group_count}\n")
		mean = grouped.mean()
		print(f"Mean by {column} :\n {mean}\n")
		std = grouped.std()
		print(f"Standard Deviation by {column} :\n {std})")

