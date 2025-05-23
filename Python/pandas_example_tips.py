#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# =============================================================================
# 1. PANDAS CORE DATA STRUCTURES
# =============================================================================

# 1.1. SERIES
# ===========
# Series: 1D labeled array capable of holding any data type

# Creating Series
s1 = pd.Series([1, 2, 3, 4])
# Output:
# 0    1
# 1    2
# 2    3
# 3    4
# dtype: int64

s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# Output:
# a    1
# b    2
# c    3
# d    4
# dtype: int64

s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})
# Creates Series from dictionary (keys become indices)

s4 = pd.Series(5, index=['a', 'b', 'c'])
# Output: Creates Series with the same value replicated
# a    5
# b    5
# c    5
# dtype: int64

# Series Attributes
# s1.values - Returns array of values
# s1.index - Returns index object
# s1.dtype - Returns dtype of values
# s1.shape - Returns shape of Series
# s1.size - Returns number of elements
# s1.name - Name of Series (can be assigned)
# s1.index.name - Name of index (can be assigned)

# Series Methods
# s1.head(n) - First n elements (default 5)
# s1.tail(n) - Last n elements (default 5)
# s1.sample(n) - Random sample of n elements
# s1.describe() - Summary statistics describe()
# s1.count() - Count non-NA/null values
# s1.unique() - Unique values in Series
# s1.value_counts() - Count occurrence of each value Value_counts
# s1.sort_values() - Sort by values
# s1.sort_index() - Sort by index
# s1.reset_index() - Reset index, moving it to columns

# Accessing elements
# By position (integer)
val = s1[0]  # 1

# By label (if index is defined)
val = s2['a']  # 1

# Using .loc (label-based)
val = s2.loc['a']  # 1

# Using .iloc (position-based)
val = s2.iloc[0]  # 1

# Using .at (single label access)
val = s2.at['a']  # 1

# Using .iat (single position access)
val = s2.iat[0]  # 1

# Series slicing
slice1 = s1[1:3]  # Series with elements at positions 1, 2
slice2 = s2['a':'c']  # Series with elements having labels 'a', 'b', 'c'

# EXCEPTION: KeyError if accessing non-existent label
try:
    s2['z']  # Raises KeyError
except KeyError:
    pass  # Handle the exception

# EXCEPTION: IndexError if accessing out-of-bounds position
try:
    s1[10]  # Raises IndexError
except IndexError:
    pass  # Handle the exception

# Operations on Series
s5 = s1 + 2  # Element-wise addition
s6 = s1 * 3  # Element-wise multiplication
s7 = s1 ** 2  # Element-wise power

# Series with mixed data types
s_mixed = pd.Series([1, 'a', 3.14, True])
# Output:
# 0       1
# 1       a
# 2    3.14
# 3    True
# dtype: object


# 1.2. DATAFRAME
# ==============
# DataFrame: 2D labeled data structure with columns of potentially different types

# Creating DataFrame from dictionary
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
    'C': [4.0, 5.0, 6.0]
})
# Output:
#    A  B    C
# 0  1  a  4.0
# 1  2  b  5.0
# 2  3  c  6.0

# Creating DataFrame from list of lists/tuples
df2 = pd.DataFrame([
    [1, 'a', 4.0],
    [2, 'b', 5.0],
    [3, 'c', 6.0]
], columns=['A', 'B', 'C'])

# Creating DataFrame from NumPy array
df3 = pd.DataFrame(
    np.random.randn(3, 3),
    columns=['A', 'B', 'C'],
    index=['row1', 'row2', 'row3']
)

# Creating DataFrame from Series
df4 = pd.DataFrame([s1, s2])

# Creating DataFrame with MultiIndex
df_multi = pd.DataFrame(
    np.random.randn(4, 2),
    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
    columns=['A', 'B']
)
# Output:
#        A         B
# a 1  0.123456  0.234567
#   2 -0.345678  0.456789
# b 1  0.567890 -0.678901
#   2  0.789012  0.890123

# DataFrame Attributes
# df1.values - Returns 2D array of values Values
# df1.columns - Returns Index object for columns columns
# df1.index - Returns Index object for rows Index
# df1.dtypes - Returns Series with dtypes of each column dtypes
# df1.shape - Returns tuple of (rows, columns) shape
# df1.size - Returns number of elements Size
# df1.T - Transpose of DataFrame Transponse

# DataFrame Methods
# df1.head(n) - First n rows (default 5)
# df1.tail(n) - Last n rows (default 5)
# df1.sample(n) - Random sample of n rows
# df1.describe() - Summary statistics for numeric columns describe()
# df1.info() - Concise summary of DataFrame info 
# df1.count() - Count non-NA cells for each column count
# df1.sort_values(by) - Sort by values in specified column(s) Sort_values()
# df1.sort_index() - Sort by index sort_index()
# df1.reset_index() - Reset index, moving it to columns reset_index()
# df1.drop(labels) - Drop specified labels (rows or columns) drop

# Accessing columns
col_a = df1['A']  # Returns Series with column A
col_a_b = df1[['A', 'B']]  # Returns DataFrame with columns A and B

# Setting new column
df1['D'] = [7, 8, 9] # we add New column D and corresponding Values
df1['E'] = df1['A'] + df1['C']

# Accessing rows by position
row0 = df1.iloc[0]  # First row as Series iloc[start:stop:step], iloc[row_idex],
rows01 = df1.iloc[0:2]  # First two rows as DataFrame

# Accessing rows by label
if 'row1' in df3.index:
    row1 = df3.loc['row1']  # Row with label 'row1' as Series

# Accessing specific cells
val = df1.iloc[0, 0]  # Value at first row, first column
val = df1.loc[0, 'A']  # Value at row 0, column 'A'
val = df1.at[0, 'A']  # Faster for single label access
val = df1.iat[0, 0]  # Faster for single integer access

# Boolean indexing
df_filtered = df1[df1['A'] > 1]  # Rows where column A > 1
df_filtered = df1[(df1['A'] > 1) & (df1['C'] < 6)]  # Compound conditions

# EXCEPTION: KeyError if accessing non-existent column
try:
    df1['Z']  # Raises KeyError
except KeyError:
    pass  # Handle the exception

# =============================================================================
# 2. DATA MANIPULATION
# =============================================================================

# 2.1. DATA INSPECTION
# ====================

# Basic info
# df1.dtypes - Data types of each column dtypes
# df1.info() - Summary including dtypes and non-null values info()
# df1.describe() - Statistical summary of numeric columns describe()
# df1.shape - Dimensions (rows, columns) shape
# df1.columns - Column labels columns
# df1.index - Row labels index

# Additional inspection methods
# df1.count() - Non-NA count for each column count()
# df1.nunique() - Number of unique values in each column nunique()
# df1.value_counts() - Counts of unique values (for Series) value_count()
# df1.isnull().sum() - Count of missing values per column
# df1.memory_usage() - Memory usage of each column

# Example: Creating data with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': ['a', 'b', 'c', None]
})

# Checking for missing values
missing_mask = df_missing.isnull() # Bool mask of missing values isnull()
missing_counts = df_missing.isnull().sum() # Count of missing values per column isnull.sum()
total_missing = df_missing.isnull().sum().sum() # Total count of missing values  isnull.sum().sum()

# 

# 2.2. DATA CLEANING
# ==================

# Handling missing data
df_dropped_rows = df_missing.dropna()  # Drop rows with any missing values dropna()
df_dropped_rows = df_missing.dropna(how='all')  # Drop rows with all missing values 
df_dropped_cols = df_missing.dropna(axis=1)  # Drop columns with any missing values drop(axis=1)
df_cleaned = df_missing.fillna(0)  # Fill missing values with 0 fillna(0)
df_cleaned = df_missing.fillna({'A': 0, 'B': 5, 'C': 'unknown'})  # Fill by column 

# Forward fill (propagate last valid observation forward)
df_ffilled = df_missing.fillna(method='ffill')

# Backward fill (use next valid observation to fill gap)
df_bfilled = df_missing.fillna(method='bfill')

# Interpolation (linear by default)
df_interp = df_missing.interpolate()

# Removing duplicates
df_with_dupes = pd.DataFrame({
    'A': [1, 1, 2, 3],
    'B': ['a', 'a', 'b', 'c']
})
df_no_dupes = df_with_dupes.drop_duplicates()
df_no_dupes = df_with_dupes.drop_duplicates(subset=['A'])  # Based on column A only

# Replacing values
df_replaced = df1.replace(1, 100)  # Replace all 1s with 100
df_replaced = df1.replace({1: 100, 2: 200})  # Replace using mapping
df_replaced = df1.replace(['a', 'b'], ['A', 'B'])  # Replace list of values

# Data type conversion
df_converted = df1.astype({'A': float, 'B': str})  # Convert by column
df_converted = df1['A'].astype(float)  # Convert single column

# String methods (applied to string columns)
if 'B' in df1.columns:
    upper_case = df1['B'].str.upper()
    contains_a = df1['B'].str.contains('a')
    split_result = df1['B'].str.split('_')
    
# 2.3. DATA TRANSFORMATION
# ========================

# Applying functions
doubled = df1['A'].apply(lambda x: x * 2) # apply to single column
df_transformed = df1.apply(lambda x: x * 2)  # Apply to each column
df_transformed = df1['A'].applymap(lambda x: str(x).upper())  # Apply to each element

# Mapping values
mapping = {1: 'One', 2: 'Two', 3: 'Three'}
mapped = df1['A'].map(mapping)

# One-hot encoding
dummies = pd.get_dummies(df1['B'])

# Cut and qcut (binning)
bins = [0, 2, 4, 6, 8]
labels = ['Low', 'Medium', 'High', 'Very High']
binned = pd.cut(df1['A'], bins, labels=labels)
quantile_binned = pd.qcut(df1['A'], 4)  # Quartiles

# Pivoting
# Create sample data for pivot
df_for_pivot = pd.DataFrame({
    'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
    'B': ['one', 'two', 'one', 'two', 'one', 'two'],
    'C': [1, 2, 3, 4, 5, 6],
    'D': [10, 20, 30, 40, 50, 60]
})

pivoted = df_for_pivot.pivot(index='A', columns='B', values='C')
# Output:
# B    one  two
# A
# bar    5    6
# foo    2    2

# Unpivoting (melt)
df_for_melt = pd.DataFrame({
    'A': ['foo', 'bar'],
    'B': [1, 2],
    'C': [3, 4]
})

melted = pd.melt(df_for_melt, id_vars=['A'], value_vars=['B', 'C'])
# Output:
#      A variable  value
# 0  foo        B      1
# 1  bar        B      2
# 2  foo        C      3
# 3  bar        C      4

# 2.4. DATA GROUPING AND AGGREGATION
# ==================================

# Create sample data for grouping
df_for_group = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
    'B': [1, 2, 3, 4, 5, 6],
    'C': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
})

# Simple groupby
grouped = df_for_group.groupby('A')
group_sum = grouped.sum()
# Output:
#        B     C
# A
# bar  12.0  16.5
# foo   9.0  13.5

# Multiple aggregations
agg_result = grouped.agg({'B': 'sum', 'C': ['sum', 'mean', 'std']})

# Using named aggregation (pandas 0.25+)
named_agg = grouped.agg(
    b_sum=('B', 'sum'),
    c_mean=('C', 'mean')
)

# Using multiple columns for grouping
multi_group = df_for_group.groupby(['A', 'B']).sum()

# Groupby with transformation
standardized = grouped.transform(lambda x: (x - x.mean()) / x.std())

# Filter groups based on a condition
filtered_groups = grouped.filter(lambda x: x['B'].sum() > 10)

# Get group sizes
group_sizes = grouped.size()

# 2.5. MERGING AND JOINING
# ========================

# Create dataframes for merging
left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

right = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K4'],
    'C': ['C0', 'C1', 'C2', 'C4'],
    'D': ['D0', 'D1', 'D2', 'D4']
})

# Different join types
inner_join = pd.merge(left, right, on='key')  # Inner join (default)
left_join = pd.merge(left, right, on='key', how='left')  # Left join
right_join = pd.merge(left, right, on='key', how='right')  # Right join
outer_join = pd.merge(left, right, on='key', how='outer')  # Full outer join

# Joining on indexes
indexed_join = left.set_index('key').join(right.set_index('key'))

# Merging with different column names
merge_diff_cols = pd.merge(
    left, right, 
    left_on='key', right_on='key'
)

# Handling duplicates during merge (use suffixes)
merge_with_suffix = pd.merge(
    left, right, 
    on='key', 
    suffixes=('_left', '_right')
)

# 2.6. CONCATENATION
# =================

# Simple concatenation (row-wise, default)
concat_rows = pd.concat([left, right])

# Column-wise concatenation
concat_cols = pd.concat([left, right], axis=1)

# Handling indexes during concatenation
concat_ignore_index = pd.concat([left, right], ignore_index=True)

# Join types in concatenation
concat_inner = pd.concat([left, right], join='inner')  # Keep only shared columns

# 2.7. RESHAPING
# ==============

# Stacking and unstacking
# Create a multi-index DataFrame
multi_idx = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
}, index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]])

# Stacking: pivots DataFrame from columns to index
stacked = multi_idx.stack()
# Output:
# a  1  A    1
#       B    5
#    2  A    2
#       B    6
# b  1  A    3
#       B    7
#    2  A    4
#       B    8
# dtype: int64

# Unstacking: pivots from index to columns
unstacked = stacked.unstack()
# Returns to original form

# Exploding lists in cells
df_with_lists = pd.DataFrame({
    'A': [[1, 2], [3, 4]],
    'B': [['a', 'b'], ['c', 'd']]
})
exploded = df_with_lists.explode('A')
# Output:
#      A          B
# 0    1  [a, b]
# 0    2  [a, b]
# 1    3  [c, d]
# 1    4  [c, d]

# =============================================================================
# 3. TIME SERIES FUNCTIONALITY
# =============================================================================

# Creating date ranges
date_range = pd.date_range(start='2023-01-01', end='2023-01-10')
# DatetimeIndex(['2023-01-01', '2023-01-02', ..., '2023-01-09', '2023-01-10'], dtype='datetime64[ns]', freq='D')

date_range = pd.date_range(start='2023-01-01', periods=10, freq='D')  # Daily frequency
date_range = pd.date_range(start='2023-01-01', periods=10, freq='B')  # Business days
date_range = pd.date_range(start='2023-01-01', periods=10, freq='W')  # Weekly
date_range = pd.date_range(start='2023-01-01', periods=10, freq='M')  # Month end
date_range = pd.date_range(start='2023-01-01', periods=10, freq='Q')  # Quarter end
date_range = pd.date_range(start='2023-01-01', periods=10, freq='A')  # Year end
date_range = pd.date_range(start='2023-01-01', periods=10, freq='H')  # Hourly

# Creating time series data
ts = pd.Series(np.random.randn(10), index=date_range)

# Time-based indexing
if len(ts) > 0:
    val = ts['2023-01-05']  # Get value for specific date
    subset = ts['2023-01-05':'2023-01-08']  # Get date range

# Time series methods
# ts.shift(1) - Shift values by 1 period
# ts.diff() - First difference
# ts.pct_change() - Percentage change
# ts.resample('M').mean() - Resample to monthly frequency (taking mean)
# ts.rolling(3).mean() - Rolling mean with window of 3
# ts.expanding().mean() - Expanding window mean

# Date/time components
if len(ts) > 0:
    year = ts.index.year
    month = ts.index.month
    day = ts.index.day
    dayofweek = ts.index.dayofweek
    quarter = ts.index.quarter

# Time zone handling
ts_utc = pd.Series(np.random.randn(10), index=pd.date_range('2023-01-01', periods=10, tz='UTC'))
ts_ny = ts_utc.tz_convert('America/New_York')

# Period functionality
period_range = pd.period_range('2023-01', periods=10, freq='M')
period_series = pd.Series(np.random.randn(10), index=period_range)

# Converting between timestamps and periods
ts_to_period = ts.to_period('M')  # Convert DatetimeIndex to PeriodIndex
period_to_ts = period_series.to_timestamp()  # Convert PeriodIndex to DatetimeIndex

# Handling timedeltas
timedeltas = pd.Series(pd.to_timedelta(np.arange(10), unit='d'))  # 0-9 days
ts_plus_10d = ts.index + pd.Timedelta(days=10)  # Add 10 days to each timestamp

# =============================================================================
# 4. INPUT/OUTPUT OPERATIONS
# =============================================================================

# 4.1. READING DATA
# ================

# CSV files
# df = pd.read_csv('file.csv')
# df = pd.read_csv('file.csv', sep=',', header=0, index_col=0)
# df = pd.read_csv('file.csv', skiprows=2, nrows=10)
# df = pd.read_csv('file.csv', na_values=['NA', 'Missing'])
# df = pd.read_csv('file.csv', parse_dates=['date_column'])
# df = pd.read_csv('file.csv', dtype={'col1': 'int32', 'col2': 'float64'})

# Excel files
# df = pd.read_excel('file.xlsx')
# df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
# df = pd.read_excel('file.xlsx', sheet_name=[0, 1])  # Multiple sheets

# SQL databases
# import sqlite3
# conn = sqlite3.connect('database.db')
# df = pd.read_sql('SELECT * FROM table', conn)
# df = pd.read_sql_query('SELECT * FROM table WHERE column > 5', conn)
# df = pd.read_sql_table('table_name', conn)

# JSON files
# df = pd.read_json('file.json')
# df = pd.read_json('file.json', orient='records')

# HTML tables
# dfs = pd.read_html('https://example.com/tables.html')
# df = dfs[0]  # First table

# Other formats
# df = pd.read_pickle('file.pkl')
# df = pd.read_hdf('file.h5', 'key')
# df = pd.read_feather('file.feather')
# df = pd.read_parquet('file.parquet')
# df = pd.read_sas('file.sas7bdat')
# df = pd.read_stata('file.dta')
# df = pd.read_fwf('file.txt', widths=[10, 10, 6])  # Fixed-width files

# 4.2. WRITING DATA
# ================

# df.to_csv('output.csv')
# df.to_csv('output.csv', index=False)
# df.to_csv('output.csv', columns=['A', 'B'])
# df.to_csv('output.csv', sep='\t')
# df.to_csv('output.csv', na_rep='NULL')
# df.to_csv('output.csv', float_format='%.2f')
# df.to_csv('output.csv', quoting=csv.QUOTE_NONNUMERIC)

# df.to_excel('output.xlsx', sheet_name='Sheet1')
# df.to_excel('output.xlsx', index=False)
# with pd.ExcelWriter('output.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='Sheet1')
#     df2.to_excel(writer, sheet_name='Sheet2')

# df.to_sql('table_name', conn, if_exists='replace', index=False)

# df.to_json('output.json')
# df.to_json('output.json', orient='records')
# df.to_json('output.json', date_format='iso')

# df.to_pickle('output.pkl')
# df.to_hdf('output.h5', key='df')
# df.to_feather('output.feather')
# df.to_parquet('output.parquet', compression='gzip')

# =============================================================================
# 5. ADDITIONAL FUNCTIONALITY
# =============================================================================

# 5.1. VISUALIZATION
# =================

# Basic plotting
df_plot = pd.DataFrame({
    'A': np.random.randn(100).cumsum(),
    'B': np.random.randn(100).cumsum(),
    'C': np.random.randn(100).cumsum()
})

# Line plot
# df_plot.plot()

# Various plot types
# df_plot.plot.line()
# df_plot.plot.bar()
# df_plot.plot.barh()
# df_plot.plot.hist(bins=20)
# df_plot.plot.box()
# df_plot.plot.kde()
# df_plot.plot.density()
# df_plot.plot.area()
# df_plot.plot.scatter(x='A', y='B')
# df_plot.plot.hexbin(x='A', y='B', gridsize=20)
# df_plot.plot.pie(subplots=True)

# 5.2. COMPUTATION & STATISTICS
# ============================

# Statistical methods
# df.corr() - Correlation between columns
# df.cov() - Covariance between columns
# df.kurtosis() - Kurtosis for each column
# df.skew() - Skewness for each column
# df.rank() - Rank of values in each column
# df.pct_change() - Percentage change between adjacent elements
# df.quantile([0.25, 0.5, 0.75]) - Quantiles

# Window functions
# df.rolling(window=3).mean() - Rolling mean with window size 3
# df.rolling(window=3).sum() - Rolling sum
# df.rolling(window=3).std() - Rolling standard deviation
# df.expanding().mean() - Expanding mean
# df.ewm(alpha=0.3).mean() - Exponentially weighted moving average

# 5.3. ADVANCED INDEXING
# =====================

# MultiIndex creation
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
multi_index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'number'))
df_multi = pd.DataFrame(np.random.randn(4, 2), index=multi_index, columns=['X', 'Y'])

# Accessing with MultiIndex
val = df_multi.loc[('A', 1), 'X']  # Access specific value
subset = df_multi.loc['A']  # All rows with first level 'A'
subset = df_multi.xs('A', level='letter')  # Cross-section by level name

# Manipulating MultiIndex
swapped = df_multi.swaplevel('letter', 'number')  # Swap index levels
sorted_idx = df_multi.sort_index(level='letter')  # Sort by level

# 5.4. APPLYING CUSTOM FUNCTIONS
# ============================

def custom_func(x):
    return x * 2 if isinstance(x, (int, float)) else str(x).upper()

# Apply to a Series
if 'A' in df1.columns:
    applied = df1['A'].apply(custom_func)

# Apply to each column of a DataFrame
applied_df = df1.apply(lambda x: custom_func(x) if x.name == 'A' else x)

# Apply to each element
applied_elements = df1.applymap(custom_func)

# Apply with additional arguments
if 'A' in df1.columns:
    applied_with_args = df1['A'].apply(lambda x, y, z: x * y + z, args=(2, 1))

# Using the .pipe method for method chaining
piped = (df1
         .pipe(lambda df: df[df['A'] > 1] if 'A' in df.columns else df)
         .pipe(lambda df: df.assign(Z=df.sum(axis=1))))

# 5.5. HANDLING CATEGORICAL DATA
# ============================

# Creating categorical variables
cat_df = pd.DataFrame({
    'A': pd.Categorical(['a', 'b', 'c', 'a']),
    'B': pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c', 'd']),
    'C': pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'], ordered=True)
})

# Properties of categorical data
if 'A' in cat_df.columns:
    cats = cat_df['A'].cat.categories
    codes = cat_df['A'].cat.codes
    ordered = cat_df['A'].cat.ordered

# Manipulating categories
if 'A' in cat_df.columns:
    cat_df['A'] = cat_df['A'].cat.add_categories(['d', 'e'])
    cat_df['A'] = cat_df['A'].cat.remove_categories(['a'])
    cat_df['A'] = cat_df['A'].cat.set_categories(['b', 'c', 'd', 'e', 'f'])
    cat_df['A'] = cat_df['A'].cat.reorder_categories(['c', 'b', 'd', 'e', 'f'], ordered=True)
    cat_df['A'] = cat_df['A'].cat.as_ordered()  # Set as ordered
    cat_df['A'] = cat_df['A'].cat.as_unordered()  # Set as unordered

# 5.6. PERFORMANCE OPTIMIZATION
# ===========================

# Using optimized data types
df_optimized = pd.DataFrame({
    'A': np.random.randint(0, 100, 10000),
    'B': np.random.choice(['x', 'y', 'z'], 10000)
})

# Convert to categoricals for memory efficiency
df_optimized['B'] = df_optimized['B'].astype('category')

# Numeric downcasting
df_optimized['A'] = pd.to_numeric(df_optimized['A'], downcast='integer')

# Get memory usage
mem_usage = df_optimized.memory_usage(deep=True)

# Efficient iteration (avoid as much as possible)
# df.iterrows() - Iterator over (index, Series) pairs
# df.itertuples() - Iterator over rows as namedtuples (faster than iterrows)
# df.values - Convert to NumPy array for fastest iteration

# Using eval() for complex expressions (uses numexpr package)
if 'A' in df1.columns and 'C' in df1.columns:
    result = pd.eval('df1.A + df1.C')
    
    # Using query() for filtering (also uses numexpr)
    filtered = df1.query('A > 1 and C < 5')

# 5.7. EXTENSION ARRAYS
# ====================

# pandas provides extension arrays for specialized data types

# Example 1: Integer arrays with NA values
from pandas import Int64Dtype
df_int_na = pd.DataFrame({
    'A': pd.array([1, 2, None, 4], dtype=Int64Dtype())
})
# This allows for integer columns with NA values, unlike native NumPy arrays

# Example 2: Boolean arrays with NA values
from pandas import BooleanDtype
df_bool_na = pd.DataFrame({
    'A': pd.array([True, False, None, True], dtype=BooleanDtype())
})

# Example 3: String arrays
from pandas import StringDtype
df_string = pd.DataFrame({
    'A': pd.array(['a', 'b', None, 'd'], dtype=StringDtype())
})

# =============================================================================
# 6. COMMON EXCEPTIONS AND ERROR HANDLING
# =============================================================================

# KeyError - Accessing a non-existent label
try:
    val = df1['Non_existent_column']
except KeyError as e:
    # Handle the error
    error_message = f"Column not found: {e}"

# ValueError - Many operations can raise this
try:
    pd.to_datetime(['not_a_date'])
except ValueError as e:
    # Handle the error
    error_message = f"Value error: {e}"

# IndexError - Out-of-bounds indexing
try:
    val = df1.iloc[1000, 0]  # Assuming df1 has fewer than 1000 rows
except IndexError as e:
    # Handle the error
    error_message = f"Index out of bounds: {e}"

# TypeError - Wrong type for operation
try:
    result = df1['A'] + 'string'  # Trying to add string to numeric column
except TypeError as e:
    # Handle the error
    error_message = f"Type error: {e}"

# AttributeError - Accessing non-existent attribute
try:
    df1.non_existent_method()
except AttributeError as e:
    # Handle the error
    error_message = f"Attribute not found: {e}"

# =============================================================================
# 7. BEST PRACTICES AND TIPS
# =============================================================================

# 1. Use vectorized operations whenever possible instead of loops
#    Example: df['A'] * 2 instead of iterating over rows

# 2. Chain methods for cleaner code
#    Example: (df.query('A > 1')
#                .assign(D=lambda x: x.A + x.C)
#                .sort_values('D'))

# 3. Use .loc, .iloc, .at, .iat for accessing data, not [] where possible
#    Example: df.loc[df.A > 1, 'B'] instead of df[df.A > 1]['B']

# 4. Set proper data types early to save memory
#    Example: df['category_col'] = df['category_col'].astype('category')

# 5. Use pd.read_csv with appropriate parameters for large files
#    Example: Use chunksize, usecols, nrows to limit memory usage

# 6. Use appropriate NA representations (pd.NA, np.nan, None) consistently

# 7. Understand index operations and when to reset_index()

# 8. Profile memory usage for large dataframes
#    Example: df.info(memory_usage='deep')

# 9. Use categorical data types for string columns with few unique values

# 10. Be aware of the copy vs view behavior
#     Example: df_slice = df[['A']].copy() to avoid SettingWithCopyWarning