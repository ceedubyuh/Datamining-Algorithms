import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import numpy as np
import scipy.stats as sp

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg


menu_options = {
	1: 'Trends of Pre, During, and Post Covid- How Many People went to the Food Pantry',
	2: 'Trends of How Many People go to the Food Pantry Yearly, Monthly, and/or Weekly',
	3: 'Linear Regression - Pending (Prediction not completed)',
	4: 'Time Series',
	0: 'Exit',
}


def type_of_data(argument, dataframe):
	if argument == 1:
		dataframe = dataframe.sort_values('date')
		pre_covid = dataframe[((dataframe["date"] > '2016-12-31') & (dataframe["date"] <= '2019-12-31'))]
		# month 
		pd.DatetimeIndex(pre_covid['date']).to_period("M")
		pre_covid_per = pre_covid['date'].dt.to_period("M")
		pre_covid_g = pre_covid.groupby(pre_covid_per)
		pre_covid_count_month = pre_covid_g.count() # count how many people went per month
		pre_covid_count_month = pre_covid_count_month.drop(['date'], axis=1)
		# year 
		pd.DatetimeIndex(pre_covid['date']).to_period("Y")
		pre_covid_per1 = pre_covid['date'].dt.to_period("Y")
		pre_covid_g1 = pre_covid.groupby(pre_covid_per1)
		pre_covid_count_year = pre_covid_g1.count()
		pre_covid_count_year = pre_covid_count_year.drop(columns=['date'])
		# week
		pd.DatetimeIndex(pre_covid['date']).to_period("W")
		pre_covid_per2 = pre_covid['date'].dt.to_period("W")
		pre_covid_g2 = pre_covid.groupby(pre_covid_per2)
		pre_covid_count_week = pre_covid_g2.count() # count how many people went per month
		pre_covid_count_week = pre_covid_count_week.drop(['date'], axis=1)

		during_covid = dataframe[((dataframe["date"] > '2018-12-31') & (dataframe["date"] <= '2021-12-31'))]
		# month 
		pd.DatetimeIndex(during_covid['date']).to_period("M")
		during_covid_per = during_covid['date'].dt.to_period("M")
		during_covid_g = during_covid.groupby(during_covid_per)
		during_covid_count_month = during_covid_g.count() # count how many people went per month
		during_covid_count_month = during_covid_count_month.drop(['date'], axis=1)
		# year 
		pd.DatetimeIndex(during_covid['date']).to_period("Y")
		during_covid_per1 = during_covid['date'].dt.to_period("Y")
		during_covid_g1 = during_covid.groupby(during_covid_per1)
		during_covid_count_year = during_covid_g1.count()
		during_covid_count_year = during_covid_count_year.drop(columns=['date'])
		# week
		pd.DatetimeIndex(during_covid['date']).to_period("W")
		during_covid_per2 = during_covid['date'].dt.to_period("W")
		during_covid_g2 = during_covid.groupby(during_covid_per2)
		during_covid_count_week = during_covid_g2.count() # count how many people went per month
		during_covid_count_week = during_covid_count_week.drop(['date'], axis=1)

		post_covid = dataframe[((dataframe["date"] > '2020-12-31') & (dataframe["date"] <= '2023-12-31'))]
		# month 
		pd.DatetimeIndex(post_covid['date']).to_period("M")
		post_covid_per = post_covid['date'].dt.to_period("M")
		post_covid_g = post_covid.groupby(post_covid_per)
		post_covid_count_month = post_covid_g.count() # count how many people went per month
		post_covid_count_month = post_covid_count_month.drop(['date'], axis=1)
		# year 
		pd.DatetimeIndex(post_covid['date']).to_period("Y")
		post_covid_per1 = post_covid['date'].dt.to_period("Y")
		post_covid_g1 = post_covid.groupby(post_covid_per1)
		post_covid_count_year = post_covid_g1.count()
		post_covid_count_year = post_covid_count_year.drop(columns=['date'])
		# week
		pd.DatetimeIndex(post_covid['date']).to_period("W")
		post_covid_per2 = post_covid['date'].dt.to_period("W")
		post_covid_g2 = post_covid.groupby(post_covid_per2)
		post_covid_count_week = post_covid_g2.count() # count how many people went per month
		post_covid_count_week = post_covid_count_week.drop(['date'], axis=1)

		print("")
		dateoption = input("Enter if you want data for yearly(Y/y), month(M/m), or weekly(W/w): ")  or 'Y'
		print("")

		if dateoption == 'Y' or dateoption == 'y' or dateoption == 'yearly' or dateoption == 'Yearly':
			pre_covid_count_year.plot(y='totalpoints', use_index=True, title='How Many People Came - Pre-Covid Yearly')
			during_covid_count_year.plot(y='totalpoints', use_index=True, title='How Many People Came - During Covid Yearly')
			post_covid_count_year.plot(y='totalpoints', use_index=True, title='How Many People Came - Post-Covid Yearly')
			plt.show()
		if dateoption == 'M' or dateoption == 'm' or dateoption == 'monthly' or dateoption == 'Monthly':
			pre_covid_count_month.plot(y='totalpoints', use_index=True, title='How Many People Came - Pre-Covid Monthly')
			during_covid_count_month.plot(y='totalpoints', use_index=True, title='How Many People Came - During Covid Monthly')
			post_covid_count_month.plot(y='totalpoints', use_index=True, title='How Many People Came - Post-Covid Monthly')
			plt.show()
		if dateoption == 'W' or dateoption == 'w' or dateoption == 'weekly' or dateoption == 'Weekly':
			pre_covid_count_week.plot(y='totalpoints', use_index=True, title='How Many People Came - Pre-Covid Weekly')
			during_covid_count_week.plot(y='totalpoints', use_index=True, title='How Many People Came - During Covid Weekly')
			post_covid_count_week.plot(y='totalpoints', use_index=True, title='How Many People Came - Post-Covid Weekly')
			plt.show()
		return argument

	if argument == 2:
		print("")
		dataframe = dataframe.sort_values('date')
		# get data per month since the start date of the data
		pd.DatetimeIndex(dataframe['date']).to_period("M")
		per = dataframe['date'].dt.to_period("M")
		g = dataframe.groupby(per)
		sum_month = g.sum() #sums per 1 month
		count_month = g.count() # count how many people went per month
		count_month = count_month.drop(['date'], axis=1)
		how_many_months = count_month.count() # how many months of data
	
		# get data per year since the start date of the data
		pd.DatetimeIndex(dataframe['date']).to_period("Y")
		per1 = dataframe['date'].dt.to_period("Y")
		g1 = dataframe.groupby(per1)
		sum_year = g1.sum()
		count_year = g1.count()
		count_year = count_year.drop(columns=['date'])
		how_many_years = count_year.count() 

		# get data per week since the start date of the data
		pd.DatetimeIndex(dataframe['date']).to_period("W")
		per2 = dataframe['date'].dt.to_period("W")
		g2 = dataframe.groupby(per2)
		sum_week = g2.sum()
		count_week = g2.count()
		count_week = count_week.drop(columns=['date'])
		how_many_weeks = count_week.count()

		count_year.plot(y='totalpoints', use_index=True, title='How Many People Came Per Year')
		count_month.plot(y='totalpoints', use_index=True, title='How Many People Came Per Month')
		count_week.plot(y='totalpoints', use_index=True, title='How Many People Came Per Week')
		plt.show()
		return argument

	if argument == 3:
		arg3_df = dataframe.sample(frac=1)
		print("")
		dateoption = input("Enter if you want data for yearly(Y/y), month(M/m), or weekly(W/w): ")  or 'Y'
		dateoption2 = int(input("Enter if you want data for 3 point items (3), 2 point items (3), or 1 point items (1): ")) 
		print("")

		if dateoption == 'Y' or dateoption == 'y' or dateoption == 'yearly' or dateoption == 'Yearly':
			pd.DatetimeIndex(arg3_df['date']).to_period("Y")
			arg3_per = arg3_df['date'].dt.to_period("Y")
			arg3_g = arg3_df.groupby(arg3_per)
			arg3_sum_year = arg3_g.sum()
			arg3_sum_year = arg3_sum_year.reset_index('date')
			arg3_sum_year['date'] = arg3_sum_year['date'].astype(str)
			arg3_sum_year['date'] = pd.to_datetime(arg3_sum_year['date'])
			arg3_sum_year.set_index('date', inplace=True)
			arg3_df = arg3_sum_year
			if dateoption2 == 3:
				y=np.array(arg3_df['3points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['3points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['3points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 3 points')
				ax.legend();
				plt.show()
			if dateoption2 == 2:
				y=np.array(arg3_df['2points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['2points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['2points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 2 points')
				ax.legend();
				plt.show()
			if dateoption2 == 1:
				y=np.array(arg3_df['1points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['1points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['1points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 1 points')
				ax.legend();
				plt.show()
		if dateoption == 'M' or dateoption == 'm' or dateoption == 'monthly' or dateoption == 'Monthly':
			pd.DatetimeIndex(arg3_df['date']).to_period("M")
			arg3_per2 = arg3_df['date'].dt.to_period("M")
			arg3_g2 = arg3_df.groupby(arg3_per2)
			arg3_sum_month = arg3_g2.sum()
			arg3_sum_month = arg3_sum_month.reset_index('date')
			arg3_sum_month['date'] = arg3_sum_month['date'].astype(str)
			arg3_sum_month['date'] = pd.to_datetime(arg3_sum_month['date'])
			arg3_sum_month.set_index('date', inplace=True)
			arg3_df = arg3_sum_month
			if dateoption2 == 3:
				y=np.array(arg3_df['3points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['3points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['3points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 3 points')
				ax.legend();
				plt.show()
			if dateoption2 == 2:
				y=np.array(arg3_df['2points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['2points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['2points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 2 points')
				ax.legend();
				plt.show()
			if dateoption2 == 1:
				y=np.array(arg3_df['1points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['1points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['1points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 1 points')
				ax.legend();
				plt.show()
		if dateoption == 'W' or dateoption == 'w' or dateoption == 'weekly' or dateoption == 'Weekly':
			pd.DatetimeIndex(arg3_df['date']).to_period("Y")
			arg3_per3 = arg3_df['date'].dt.to_period("Y")
			arg3_g3 = arg3_df.groupby(arg3_per3)
			arg3_sum_week = arg3_g3.sum()
			arg3_sum_week = arg3_sum_week.reset_index('date')
			arg3_sum_week['date'] = arg3_sum_week['date'].astype(str)
			arg3_sum_week['date'] = pd.to_datetime(arg3_sum_week['date'])
			arg3_sum_week.set_index('date', inplace=True)
			arg3_df = arg3_sum_week
			if dateoption2 == 3:
				y=np.array(arg3_df['3points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['3points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['3points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 3 points')
				ax.legend();
				plt.show()
			if dateoption2 == 2:
				y=np.array(arg3_df['2points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['2points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['2points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 2 points')
				ax.legend();
				plt.show()
			if dateoption2 == 1:
				y=np.array(arg3_df['1points'].dropna().values, dtype=float)
				x=np.array(pd.to_datetime(arg3_df['1points'].dropna()).index.values, dtype=float)
				slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
				xf = np.linspace(min(x),max(x),100)
				xf1 = xf.copy()
				xf1 = pd.to_datetime(xf1)
				yf = (slope*xf)+intercept
				print(' r = ', r_value, '\n', 'p = ', p_value, '\n', 's = ', std_err)
				print("Equation: y =", slope, "x +", intercept)
				f, ax = plt.subplots(1, 1)
				ax.plot(xf1, yf,label='Linear fit', lw=3)
				arg3_df['1points'].dropna().plot(ax=ax,marker='o', ls='')
				plt.ylabel('# of items of 1 points')
				ax.legend();
				plt.show()
		print("")
		return argument

	if argument == 4:
		plt.style.use('fivethirtyeight')
		plt.rcParams['lines.linewidth'] = 1.5
		data = dataframe
		data = data.set_index('date')
		data.index = pd.DatetimeIndex(data.index).to_period('M')
		data = data.sort_index()
		#data = data.sort_values(by='date')

		steps = 100
		data_train = data[:-steps]
		data_train = data_train.to_timestamp()
		data_test  = data[-steps:]
		data_test = data_test.to_timestamp()

		print(f"Train dates : {data_train.index.min()} / {data_train.index.max()}  (n={len(data_train)})")
		print(f"Test dates  : {data_test.index.min()} / {data_test.index.max()}  (n={len(data_test)})")

		forecaster = ForecasterAutoreg(
            regressor = RandomForestRegressor(random_state=123),
            lags = 600
        )

		y = data_train['totalpoints'].resample('D').mean().apply(lambda l: l if not np.isnan(l) else np.random.choice([1, 16]))
		forecaster.fit(y)
		steps = 100
		predictions = forecaster.predict(steps=steps)
		error_mse = mean_squared_error(
						y_true = data_test['totalpoints'],
						y_pred = predictions
					)
		print(f"Test error (mse): {error_mse}")

		fig, ax = plt.subplots(figsize=(9, 4))
		data_train['totalpoints'].plot(ax=ax, label='train')
		data_test['totalpoints'].plot(ax=ax, label='test')
		predictions.plot(ax=ax, label='predictions')
		ax.legend()
		plt.show()

		print(predictions.to_string())
		print("")
		return argument
			
	return 0
			


def print_menu():
	for key in menu_options.keys():
		print (key, '--', menu_options[key] )


def main ():

	print("Starting...")
	print("")

	#ask for user input of the name of the csv file
	if len(sys.argv) == 1:
		print ("You can also give filename as a command line argument. e.g. python3 python_filename.py csv_filename.csv")
		filename = input("Enter a CSV Filename: ")  or 'MOCK_DATA.csv'
	else:
		filename = sys.argv[1]

	print("")
	print("Filename being used:", filename)
	print("")

	# get data and format dates
	data = pd.read_csv(filename)
	df = pd.DataFrame(data)
	df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
	
	 
	option = -1

	while(option != 0):
		print_menu()
		option = int(input('Enter your choice: '))
		type_of_data(option, df)
	
	print("")
	print("Ending...")



if __name__ == '__main__':
	import sys
	main()



if __name__ == '__main__':
	import sys
	main()