#importing required libraries
import sqlite3
import pandas as pd


#Creating Moons class
class Moons():
	#function to intiate the class instance
	def __init__(self, db_file = 'jupiter.db', metadata = None):
		self.db_file = db_file
		self.metadata = metadata
		self.data = self.load_data()

	#function to load dataset from the SQL Database file into Pandas Dataframe
	def load_data(self):
		con = sqlite3.connect(self.db_file)
		df = pd.read_sql_query("SELECT * from moons", con)
		con.close()
		return df

	#function to check the columns of data
	def columns(self):
		return(self.data.columns)

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
			print(f'The index given is out of range.')
		else:
			moon_data = self.data.iloc[moon_idx]
			return moon_data

	#function to print required dataframe
	def print_data(self, moon = []):
		#if no specific moon selected, print all data
		if not moon:
			print(self.data)
		else:
			#create an empty dataframe
			complete_data = pd.DataFrame()
			#if given name of moons, select by name
			if type(moon[0]) is str:
				for moon_name in moon:
					moon_data = self.select_moon_by_name(moon_name)
					if not moon_data.empty:
						complete_data = pd.concat([complete_data,moon_data], ignore_index = True)
			#if given index of moons, select by index
			else:
				complete_data = self.select_moon_by_idx(moon)
		print(complete_data)


