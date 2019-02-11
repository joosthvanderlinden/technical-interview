# --------------------------------------------------------------------------------------------------
# Technical interview
# By Joost van der Linden, February 11th, 2019
#
# File contains: abstracted utility code
# 
# --------------------------------------------------------------------------------------------------
import pandas as pd

# --------------------------------------------------------------------------- SANITY CHECKS --------
def sanity_check(df):
	''' Runs a quick check of variable types, zeros and NaNs on the dataframes provided.

	Args:
		df (:obj:`pandas.DataFrame`): Dataframe to check.
	
	Returns: 
		df (:obj:`pandas.DataFrame`): Dataframe of .describe() and custom checks.
	'''

	return (pd.concat([df.dtypes, 			# dtypes
					   df.isnull().sum(), 	# nan count
					   (df == 0).sum()],	# zero count
					  keys = ['dtypes', 'nan-count', 'zero-count'], axis = 1)
			  .join(df.describe().T)
			  .fillna('-'))


# --------------------------------------------------------------------------- PREPROCESSING --------
def add_graduation_year(df):
	''' Calculates the estimated graduation year, using the age and education columns. The key
	assumptions (and probable sources of error) in this calculation are:
		(1) Most people graduate within the allotted timeframe for a degree
		(2) Over time, actual graduation ages have not fluctuated
		(3) Canadian hair style (McGill, Dalhousie, St Mary's) was similar to US hair style (Purdue) 

	Args:
		df (:obj:`pandas.DataFrame`): dataframe containing 'age' and 'education' columns.

	Returns:
		grad_year (:obj:`pandas.Series`): assumed graduation years.
	'''

	# Hard-coded (assumed!) graduation ages. Source: Google & https://collegescorecard.ed.gov/.
	graduation_ages = {
		'Preschool':    4,
		'1st-4th':      11, 
		'5th-6th':      13, 
		'7th-8th':      14,
		'9th':          15, 
		'10th':         16, 
		'11th':         17, 
		'12th':         18, 
		'HS-grad':      18, 
		'Assoc-voc':    20, 
		'Assoc-acdm':   20, 
		'Some-college': 21, 
		'Bachelors':    22, 
		'Masters':      25, 
		'Prof-school':  30, 
		'Doctorate':    30
	}

	# The census was taken in 1994.
	census_year = 1994

	# Graduation year = 1994 - [age] + [graduation age, based on [education]]
	grad_year = census_year - df['age'] + df['education'].map(graduation_ages)

	# Handle edge case of graduates graduating faster than expected. Assume graduation in 1994.
	grad_year[grad_year > census_year] = census_year

	return grad_year
