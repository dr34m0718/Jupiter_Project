#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Creating Moons class
class Moons():
	"""
	A class to modify, store and analyse data related to Jupiter moons

	Parameters:
	db_file(str) : Path to SQLite database file
	initial_data(DataFrame) : Data to be stored in the class
	"""

	def __init__(self, db_file = './jupiter.db', initial_data = None):
		"""
		Initialize Moons class
		"""
		self.__db_file = db_file
		if initial_data is not None:
			self.__data = initial_data
		else:
			self.__data = self.__load_data()

	def __load_data(self):
		"""
		Load data from SQLite database into a Pandas DataFrame
		"""
		con = f"sqlite:///{self.__db_file}"
		df = pd.read_sql_query("SELECT * from moons", con)
		return df

	def copy(self):
		"""
		Create a copy of the Moons class
		"""
		new_copy = Moons(db_file = self.__db_file, initial_data = self.__data.copy())
		return new_copy

	def columns(self):
		"""
		Check the columns of the data
		"""
		return self.__data.columns

	def drop_rows(self, rows):
		"""
		Drop specified rows from data
		"""
		self.__data.drop(rows, axis = 0, inplace = True)

	def drop_columns(self, columns):
		"""
		Drop specified columns from data
		"""
		self.__data.drop(columns, axis = 1, inplace = True)

	def drop_na(self):
		"""
		Drop rows with missing values
		"""
		self.__data.dropna(inplace = True)

	def rearrange_data_by_col(self, column):
		"""
		Sort data according to specified column
		"""
		self.__data = self.__data.sort_values(by = column).reset_index(drop = True)

	def add_moon(self, new_moon_data):
		"""
		Add data for a new moon to the class
		"""
		new_data = pd.DataFrame(new_moon_data)
		#merging original data with the new data
		self.__data = pd.concat([self.__data,new_data], ignore_index = True)
		#ensures data is in ascending order according to names
		self.rearrange_data_by_col('moon')

	def select_moon_by_name(self, moon):
		"""
		Select a single moon data by name
		"""
		moon_data = self.__data.loc[self.__data['moon'] == moon]
		if moon_data.empty:
			print(f'No data found for {moon}.')
		else:
			return moon_data

	def select_moon_by_idx(self, moon_idx):
		"""
		Select a single moon data by index
		"""
		#check if the index given is out of range
		if moon_idx < 0 or moon_idx > len(self.__data):
			raise IndexError(f'The index given is out of range.')
		else:
			moon_data = self.__data.iloc[[moon_idx]]
			return moon_data

	def select_data(self, moon):
		"""
		Select one or multiple moon data based on names or index to be created into a new Moons class
		"""
		complete_data = pd.DataFrame()
		#check if moon is given by name
		if type(moon[0]) is str:
			for moon_name in moon:
				moon_data = self.select_moon_by_name(moon_name)
				if not moon_data.empty:
					complete_data = pd.concat([complete_data, moon_data])
		else:
			for idx in moon:
				moon_data = self.select_moon_by_idx(idx)
				if not moon_data.empty:
					complete_data = pd.concat([complete_data, moon_data])
		#ensure that the data return is remained in Moon class
		selected_moon = Moons(db_file = self.__db_file, initial_data = complete_data)
		return selected_moon

	def print_data(self, moon = [], head = False):
		"""
		Print selected or all moon datas
		"""
		#check if moon is not specified, print the original data
		if not moon:
			#check if only data.head is required
			if head is False:
				return self.__data
			else:
				return self.__data.head()
		#if moon is specified, select the required data and print
		else:
			selected_moon_df = self.select_data(moon)
			if head is False:
				return selected_moon_df.__data
			else:
				return selected_moon_df.__data.head()

	def summary(self):
		"""
		Display summary statictics for data
		"""
		return self.__data.describe()

	def check_na(self):
		"""
		Check number of null values in each column
		"""
		return self.__data.isnull().sum()

	def correlation(self):
		"""
		Calculate correlation matrix of data
		"""
		return self.__data.corr()

	def plot_corr_matrix(self):
		"""
		Plot heatmap for correlation matrix
		"""
		#retrieve the correlation matrix
		corr = self.correlation()
		f, ax = plt.subplots(figsize = (6,4))
		#choose a custom diverging colormap
		cmap = sns.diverging_palette(220,20, as_cmap = True)
		#plot the heatmap
		sns.heatmap(corr, annot = True, cmap = cmap)

	def max_min_by_col(self, column, type = None):
		"""
		Find maximum or minimum values in specified column
		"""
		#check if max or min is required
		if type == 'min':
			row_idx = self.__data[column].idxmin()
		elif type == 'max':
			row_idx = self.__data[column].idxmax()
		#if type is not stated, raise a ValueError
		else:
			raise ValueError("Please specify type = 'max' or 'min'")
		#return data of the moon
		return self.select_moon_by_idx(row_idx)

	def plot(self, x = None, y = None, hue = None, plot = 'scatter'):
		"""
		Plot different types of graph
		"""
		#following is to plot a scatter plot
		if plot == 'scatter':
			#if data not specified, raise an error
			if x is None:
				raise ValueError("Datas to be plotted are not stated.")
			else:
				f, ax = plt.subplots(figsize = (4,4))
				#hue is to decide datas are grouped for different colours
				sns.scatterplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a histogram
		elif plot == 'histogram':
			if x is None:
				raise ValueError("Data to be plot is not stated.")
			else:
				f, ax = plt.subplots(figsize = (5,3))
				sns.histplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a boxplot
		elif plot == 'boxplot':
			if x is None:
				raise ValueError("Data to be plotted is not stated.")
			else:
				f, ax = plt.subplots(figsize = (5,3))
				sns.boxplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		elif plot == 'line':
			if x is None or y is None:
				raise ValueError("Datas to be plotted are not stated.")
			else:
				f, ax = plt.subplots(figsize = (4,4))
				sns.lineplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#if plot is not available, raise an error
		else:
			raise ValueError(f"The plot requested -- {plot} is not available. (Try 'scatter', 'histogram' or 'boxplot'.)")

	def grouped_analysis_summary(self, column = 'group'):
		"""
		Display simple summary of grouped analysis for specified column
		"""
		#obtain count of each group
		grouped = self.__data.groupby(column)
		group_count = grouped.size()
		print(f"Group count by {column} :\n {group_count}\n")
		#obtain mean
		mean = grouped.mean()
		print(f"Mean by {column} :\n {mean}\n")
		#obtain standard deviation
		std = grouped.std()
		print(f"Standard Deviation by {column} :\n {std})")

	def parameter_calculation(self,T,a):
		"""
		Calculate T^2 and a^3 according to Kepler's Third Law
		"""
		#conversion of units
		seconds_day = 24*60*60
		km_m = 1000
		#add columns for T^2 and a^3
		self.__data['T2_s'] = (self.__data[T]*seconds_day)**2
		self.__data['a3_m'] = (self.__data[a]*km_m)**3

	def split_train(self, x, y):
		"""
		Split data, build and train linear regression model, evaluate the predictions
		"""
		#defining the feature and target variable
		X = self.__data[[x]]
		Y = self.__data[y]
		#splitting data into train and test sets
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
		#initialise linear regression model
		self.model = LinearRegression()
		self.model.fit(x_train, y_train)
		#prediction on test set
		pred = self.model.predict(x_test)
		#obtain r2score and mse
		r2 = r2_score(y_test,pred)
		mse = mean_squared_error(y_test,pred)
		print(f"r2_score: {r2}")
		print(f"mean squared error: {mse}")
		return r2,mse

	def prediction_evaluation(self):
		"""
		Predict and print the mass of Jupiter
		"""
		#obtaining our coefficient from model
		coefficient = self.model.coef_[0]
		#gravity
		G = 6.67e-11
		#predicting M
		M_pred = (4*(np.pi**2))/(G*coefficient)
		print(f"Predicted Jupiter mass is {M_pred}.")
		return M_pred
