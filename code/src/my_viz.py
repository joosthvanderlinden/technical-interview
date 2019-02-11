# --------------------------------------------------------------------------------------------------
# Technical interview
# By Joost van der Linden, February 11th, 2019
#
# File contains: abstracted visualization code
# 
# --------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

def distribution_overview(df, is_numeric):
	''' Plots the distribution of each variable in df.

	Args:
		df (:obj:`pandas.DataFrame`): Dataframe with columns to plot.
		is_numeric (:obj:`bool`): If True, assumes all columns in df are numeric.
	
	Returns: 
		fig (:obj:`plt.fig`): matplotlib figure object.
		axs (:obj:`list` of :obj:`plt.ax`): list of matplotlib axes.
	'''
	num_rows = int(np.ceil(len(df.columns) / 4))
	fig, axs = plt.subplots(num_rows, 4, figsize = (12, 3 * num_rows))
	for i,c in enumerate(df.columns):
		row = int(np.floor(i/4))
		col = i % 4
		if is_numeric:
			df[c].plot(kind = 'hist', ax = axs[row, col])
		else:
			df[c].value_counts().plot(kind = 'bar', color = 'b', ax = axs[row, col])

		axs[row, col].set_title(c)
		axs[row, col].set_ylabel('')
		sns.despine(ax = axs[row, col], left = True)

	# Hide remaining subplots
	col += 1
	while col < 4:
		axs[row, col].axis('off')
		col += 1

	fig.suptitle('Numeric variable distributions')

	return fig, axs

def correlation_overview(df, columns, target, is_numeric):
	''' Visualizes the correlations of the variables in df with a target variable. 
	Assumes target is a categorical variable.

	Args:
		df (:obj:`pandas.DataFrame`): Dataframe containing columns and target.
		columns (:obj:`list` of :obj:`str`): Columns to correlate with
		target (:obj:`str`): Target variable to compare against 
		is_numeric (:obj:`bool`): If True, assumes columns in df are numeric.
	
	Returns: 
		fig (:obj:`plt.fig`): matplotlib figure object.
		axs (:obj:`list` of :obj:`plt.ax`): list of matplotlib axes.
	'''
	num_rows = int(np.ceil(len(columns) / 4))
	fig, axs = plt.subplots(num_rows, 4, figsize = (12, 3 * num_rows))
	for i,c in enumerate(columns):
		row = int(np.floor(i/4))
		col = i % 4
		if is_numeric:
			sns.boxplot(x = df[target], y = df[c], ax = axs[row, col], showfliers=False)
		else:
			freqs = (df.groupby(c)[target]
					   .value_counts(normalize = True)
					   .unstack()
					   .sort_values(by = True))

			sns.heatmap(freqs, annot=True, fmt = '.0%', ax = axs[row, col])

		axs[row, col].set_title(c)
		axs[row, col].set_ylabel('')
		sns.despine(ax = axs[row, col], left = True)

	# Hide remaining subplots
	col += 1
	while col < 4:
		axs[row, col].axis('off')
		col += 1

	fig.suptitle('Correlation overview')

	return fig, axs

def stacked_categorical(df, var, legend_loc = (0.9, 0.3)):
	''' Visualizes how var varies by age
	
	Args:
		df (:obj:`pandas.DataFrame`): Dataframe containing var and 'age'.
		target (:obj:`str`): Variable in df
	
	Returns: 
		fig (:obj:`plt.fig`): matplotlib figure object.
		axs (:obj:`list` of :obj:`plt.ax`): list of matplotlib axes.

	'''
	fig, ax = plt.subplots(1, 1, figsize = (6,4))

	((df.groupby('age')[var]
        .value_counts(normalize = True)
        .unstack()
        .fillna(0) * 100)
        .plot(kind = 'area', stacked = True, ax = ax)
        .legend(bbox_to_anchor = legend_loc, facecolor = 'w', edgecolor = 'w'))

	ax.set_ylabel('Fraction')
	sns.despine(ax = ax, left = True, bottom = True)

	return fig, ax



