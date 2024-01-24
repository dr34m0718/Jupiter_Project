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
	#function to intiate the class instance
	def __init__(self, db_file = './jupiter.db', initial_data = None):
		self.__db_file = db_file
		if initial_data is not None:
			self.__data = initial_data
		else:
			self.__data = self.__load_data()

	#function to load dataset from the SQL Database file into Pandas Dataframe
	def __load_data(self):
		con = f"sqlite:///{self.__db_file}"
		df = pd.read_sql_query("SELECT * from moons", con)
		return df

	#function to make a copy of the Moons class
	def copy(self):
		new_copy = Moons(db_file = self.__db_file, initial_data = self.__data.copy())
		return new_copy

	#function to check the columns of data
	def columns(self):
		return self.__data.columns

	#function to drop columns that are not needed
	def drop_columns(self, columns):
		self.__data.drop(columns, axis = 1, inplace = True)

	#function to remove all rows with missing values
	def drop_na(self):
		self.__data.dropna(inplace = True)

	#function to ensure data is stored in ascending order according to moon name
	def rearrange_data_by_col(self, column):
		self.__data = self.__data.sort_values(by = column).reset_index(drop = True)

	#funtion to add a new moon
	def add_moon(self, new_moon_data):
		new_data = pd.DataFrame(new_moon_data)
		self.__data = pd.concat([self.__data,new_data], ignore_index = True)
		self.rearrange_data_by_col('moon')

	#function to select moon according to given name
	def select_moon_by_name(self, moon):
		moon_data = self.__data.loc[self.__data['moon'] == moon]
		if moon_data.empty:
			print(f'No data found for {moon}.')
		else:
			return moon_data

	#function to select moon according to index
	def select_moon_by_idx(self, moon_idx):
		#check if the index given is out of range
		if moon_idx < 0 or moon_idx > len(self.__data):
			raise IndexError(f'The index given is out of range.')
		else:
			moon_data = self.__data.iloc[[moon_idx]]
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
			for idx in moon:
				moon_data = self.select_moon_by_idx(idx)
				if not moon_data.empty:
					complete_data = pd.concat([complete_data, moon_data])
		#ensure that the data return is remained in Moon class
		selected_moon = Moons(db_file = self.__db_file, initial_data = complete_data)
		return selected_moon

	#function to print the moons and data
	def print_data(self, moon = [], head = False):
		if not moon:
			if head is False:
				return self.__data
			else:
				return self.__data.head()
		else:
			selected_moon_df = self.select_data(moon)
			if head is False:
				return selected_moon_df.__data
			else:
				return selected_moon_df.__data.head()

	#function for summary statistics
	def summary(self):
		return self.__data.describe()

	#function to check the numbers of null values in each column
	def check_na(self):
		return self.__data.isnull().sum()

	#function to check correlation between variables
	def correlation(self):
		return self.__data.corr()

	#function to plot heatmap for the correlation matrix
	def plot_corr_matrix(self):
		corr = self.correlation()
		#set up figure
		f, ax = plt.subplots(figsize = (6,4))
		#choose a custom diverging colormap
		cmap = sns.diverging_palette(220,20, as_cmap = True)
		#plot the heatmap
		sns.heatmap(corr, annot = True, cmap = cmap)

	#function to find minimum of selected column
	def max_min_by_col(self, column, type = None):
		#check if max or min is required
		if type == 'min':
			row_idx = self.__data[column].idxmin()
		elif type == 'max':
			row_idx = self.__data[column].idxmax()
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
				f, ax = plt.subplots(figsize = (4,4))
				sns.scatterplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a histogram
		elif plot == 'histogram':
			if x is None:
				print("Data to be plot is not stated.")
			else:
				f, ax = plt.subplots(figsize = (5,3))
				sns.histplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#following is to plot a boxplot
		elif plot == 'boxplot':
			if x is None:
				print("Data to be plotted is not stated.")
			else:
				f, ax = plt.subplots(figsize = (5,3))
				sns.boxplot(self.__data, x = x, y = y, hue = hue, palette = 'pastel')
		#if plot is not available, raise an error
		else:
			raise ValueError(f"The plot requested -- {plot} is not available. (Try 'scatter', 'histogram' or 'boxplot'.)")

	#function to make a simple summary for the grouped analysis
	def grouped_analysis_summary(self, column = 'group'):
		grouped = self.__data.groupby(column)
		group_count = grouped.size()
		print(f"Group count by {column} :\n {group_count}\n")
		mean = grouped.mean()
		print(f"Mean by {column} :\n {mean}\n")
		std = grouped.std()
		print(f"Standard Deviation by {column} :\n {std})")

	def parameter_calculation(self,T,a):
		#conversion of units
		seconds_day = 24*60*60
		km_m = 1000
		#add columns for T^2 and a^3
		self.__data['T2_s'] = (self.__data[T]*seconds_day)**2
		self.__data['a3_m'] = (self.__data[a]*km_m)**3

	def split_train(self, x, y):
		#defining the feature and target variable
		X = self.__data[[x]]
		Y = self.__data[y]
		#splitting data into train and test sets
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
		#initialise linear regression model
		self.model = LinearRegression()
		self.model.fit(x_train, y_train)
		#prediction on test set
		pred = self.model.prediction(x_test)
		#obtain r2score and mse
		r2 = r2_score(y_test,pred)
		mse = mean_squared_error(y_test,pred)
		print(f"r2_score: {r2}")
		print(f"mean squared error: {mse}")
		return r2,mse

	def prediction_evaluation(self):
		#obtaining our coefficient from model
		coefficient = self.model.coef_[0]
		#gravity
		G = 6.67e-11
		#predicting M
		M_pred = (4*(np.pi**2))/(G*coefficient)
		return M_pred
