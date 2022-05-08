#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:54:00 2022

@author: alexanderng
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from pandas import read_csv

fred_dir = "../fred/"

def process_fred_dateseries( csv_file, series_name ):

#  header = 0  the header row is located at row index 0
#  parse_dates - dates are at column index 0
#  index_col  -  first column is the index for the Time Series object
#
# ----------------------------------------------------
    
    df = pd.read_csv(csv_file, header = 0, parse_dates = [0], index_col = 0 )
    
    # convert blank or missing data to nan and other values from string to numeric
    df[series_name] = pd.to_numeric( df[series_name], errors = 'coerce')
    
    # Fill forward any missing data on a business holiday.
    #
    df.fillna( method = "ffill", inplace = True )
    
    df.reset_index(inplace=True)
    
    return df

#
#   Series filenames are <SERIES-KEY>.csv
#
#   Series file content:  <date>, <series-key>
# ----------------------------------------------------------------------------
def load_and_display_fred_dateseries( series_list , show_plot = False):
    
    dict_series = { }
    
    for series in series_list:
    
        sfilename = fred_dir + series + ".csv"
        
        df_series = process_fred_dateseries( sfilename, series)
        
        dict_series.update({ series : df_series } )
        
        if show_plot:
            df_series.plot( y = series )
            plt.show()
        
    return dict_series

#
#  All the data series to be used in regression analysis.  
# -------------------------------------------------------------------------------------------
series_list = ["BAA10Y", "T10YIE", "WILL5000IND", "DGS10", "UNRATE","CIVPART",   "PCEPILFE"]

#
# Store the data series in a dictionary with:
#    key:   series key
#    value: DataFrame of dates, raw time series valuees
# 
dict_series = load_and_display_fred_dateseries(series_list, show_plot = False)



def transform_fred_dateseries( series_dict, series_key, period, is_pct , col_name , forward_shift = 1):
    
    df_series = series_dict[ series_key]
    
    if is_pct:
        df_series[ col_name ] = df_series[series_key].pct_change( periods = period )
        if period < 0:
            df_series[ col_name ] = -1 * df_series[ col_name]
        df_series[ col_name ] = df_series[col_name ].shift( periods = forward_shift)  # Forward shift
    else:
        df_series[ col_name ] = df_series[series_key].diff( periods = period )
        
        if period < 0:
            df_series[ col_name ] = -1 * df_series[ col_name]
        df_series[ col_name ] = df_series[col_name ].shift( periods = forward_shift)  # Forward shift

transform_fred_dateseries( dict_series, "BAA10Y", -2,  is_pct = True, col_name = "fp2")
transform_fred_dateseries( dict_series, "BAA10Y", -5,  is_pct = True, col_name = "fp5")

transform_fred_dateseries( dict_series, "T10YIE", -2,  is_pct = True, col_name = "fp2")
transform_fred_dateseries( dict_series, "T10YIE", -5,  is_pct = True, col_name = "fp5")

transform_fred_dateseries( dict_series, "WILL5000IND", -2,  is_pct = True, col_name = "fp2")
transform_fred_dateseries( dict_series, "WILL5000IND", -5,  is_pct = True, col_name = "fp5")

transform_fred_dateseries( dict_series, "DGS10", -2,  is_pct = True, col_name = "fp2")
transform_fred_dateseries( dict_series, "DGS10", -5,  is_pct = True, col_name = "fp5")

#
# Look back 1 month for Labor Participation Rate.  Use absolute rate change.
# 
# Join the monthly FOMC meeting to the same-month-CIVPART value and compare:
# sentiment Y of the FOMC and its prior-period change dY vs.  bd1 of the same-month-CIVPART 
#  
# Check if bd1 predicts dY
transform_fred_dateseries( dict_series, "CIVPART", 1,  is_pct = False, col_name = "bd1")

# Join the monthly FOMC meeting to the same-month-UNRATE value and compare:
# sentiment Y of the FOMC and its prior-period change dY vs.  bd1 of the same-month-UNRATE
#  
# Check if bd1 predicts dY
transform_fred_dateseries( dict_series, "UNRATE", 1,  is_pct = False, col_name = "bd1")

#
#  PCEPILFE is released monthly on a backward looking basis near the end of the following month.
#  E.g.
#        PCE[ Jan 2022] is published late Feb 2022 and would be available for an March 2022 FOMC meeting.
#
#  So we would shift the 12 month change in Jan 2022 PCEILFE to March 2022 bp12 data point.
#  i.e. use a 2 month forward shift
#
#  Regress if bp12 predicts change in sentiment dY
transform_fred_dateseries( dict_series, "PCEPILFE", 12,  is_pct = True, col_name = "bp12", forward_shift = 2)


#
# Display plots of all the time series and their transforms.
# 
# Outer loop is the series key
for v in dict_series:
    
    # inner loop is all data columns
    # only works if date is an index - not a column
    for u in dict_series[v]:
        
        if u != "DATE":
            dict_series[v].plot( x = "DATE", y = u)
            plt.title(v + " " + u)
            plt.show()


#
# Load the FOMC dates and make dummy sentiment index values
#
derived_data_dir = "../derived"

fomc_statements_file = derived_data_dir + "/" + "FOMC_statements.csv"

df_fomc_statements_raw = pd.read_csv( fomc_statements_file , parse_dates=[1], header = 0)

#  Construct a sentiment dates dataframe with dummy values
#  made using a random number generator.
# ------------------------------------------------------
num_fomc_dates = len(df_fomc_statements_raw)

np.random.seed(1029)

df_sentiment = pd.DataFrame( df_fomc_statements_raw["date"] )

df_sentiment["sentiment"] = np.sin( 0.09 * np.arange( num_fomc_dates ) ) +  0.2 * np.random.rand(num_fomc_dates)

df_sentiment.plot(x = "date", y = "sentiment")
plt.title("Dummy Sentiment Index")
plt.show()

df_sentiment[ "d1sentiment" ] = df_sentiment["sentiment"].diff( periods = 1 )

#
#  Merge dates of the Sentiment and Market Indicator to construct a regression.
#
#  Let sentiment variable be denoted Z
#  
#  For macroeconomic variables Y:   we estimate the model  Z = f(Y).   Explain sentiment in terms of macroeconomic variable.
#
#  For financial market variable X:  we estimate the model X = g(Z).  Explain financial variable in terms of sentiment variable.
#
#  Date Alignment for macroeconomic model fit:
#      for each FOMC date f(i) and the prior FOMC meeting date f(i-1) 
#           find the macroeconomic variable for the same month and year.
#
#      Estimate a model based on the change in sentiment Z[f(i)] - Z[f(i-1)]
#
#      Compare to the prior change in macroeconomic bd1 (backward difference 1 lag) or 
#      bp12 (backward proportional change 12 months)
# -------------------------------------------------------------------

CIVPART = dict_series["CIVPART"]

df_sent_CIVPART = pd.merge_asof(df_sentiment, CIVPART, left_on = "date", right_on = "DATE")

print(df_sent_CIVPART.tail(20) )

sns.lmplot( y= 'd1sentiment', x = "bd1", data = df_sent_CIVPART, fit_reg=True, legend=True)
plt.title("CIVPART vs Sentiment (Changes)")
plt.show()


UNRATE = dict_series["UNRATE"]

df_sent_UNRATE = pd.merge_asof(df_sentiment, UNRATE, left_on = "date", right_on = "DATE")

print(df_sent_UNRATE.tail(20) )

sns.lmplot( y= 'd1sentiment', x = "bd1", data = df_sent_UNRATE, fit_reg=True, legend=True)

plt.title("Unemployment vs Sentiment (Changes)")
plt.show()



PCEPILFE = dict_series["PCEPILFE"]

df_sent_PCEPILFE = pd.merge_asof(df_sentiment, PCEPILFE, left_on = "date", right_on = "DATE")

print(df_sent_PCEPILFE.tail(20) )

sns.lmplot( y= 'd1sentiment', x = "bp12", data = df_sent_PCEPILFE, fit_reg=True, legend=True)

plt.title("Annualized PCEILFE vs Sentiment (Changes)")
plt.show()

#  Date Alignment for financial model fit:
#      for each FOMC date f(i):
#           find the financial variable for the date.  Most variables are available daily.
#
#      Estimate a model on the change in the financial variable as a result of the change in sentiment.
#
#      One important problem is that FOMC statement is released during the market hours at 2pm EST.
#      Thus, the financial variable recorded at end of day on the same day as the FOMC statement already reflects
#      the sentiment impact.
#      Therefore, we need to compare the change in response of the financial variable across
#      a date before the FOMC statement versus 1 or more days after the FOMC statement.
#
#      We use 2 comparison periods to measure the change:
#          bp2:   change from 1 day before FOMC date to 1 day AFTER FOMC date
#          bp5:   change from 1 day before FOMC date to 4 days AFTER FOMC date
#      For ease of date alignment, we shifted both bp2 and bp5 by 1 day forward.
#      So the value of bp2 and bp5 on a FOMC date is actually spanning the intended time period.
# ----------------------------------------------------------

BAA10Y = dict_series["BAA10Y"]

df_sent_BAA10Y = pd.merge_asof(df_sentiment, BAA10Y, left_on = "date", right_on = "DATE")

print(df_sent_BAA10Y.tail(20) )

sns.lmplot( x= 'd1sentiment', y = "fp2", data = df_sent_BAA10Y, fit_reg=True, legend=True)

plt.title("BAA10Y 2-day vs Sentiment (Changes)")
plt.show()


sns.lmplot( x= 'd1sentiment', y = "fp5", data = df_sent_BAA10Y, fit_reg=True, legend=True)

plt.title("BAA10Y 5-day vs Sentiment (Changes)")
plt.show()


T10YIE = dict_series["T10YIE"]

df_sent_T10YIE = pd.merge_asof(df_sentiment, T10YIE, left_on = "date", right_on = "DATE")

print(df_sent_T10YIE.tail(20) )

sns.lmplot( x= 'd1sentiment', y = "fp2", data = df_sent_T10YIE, fit_reg=True, legend=True)

plt.title("T10YIE 2-day vs Sentiment (Changes)")
plt.show()


sns.lmplot( x= 'd1sentiment', y = "fp5", data = df_sent_T10YIE, fit_reg=True, legend=True)

plt.title("T10YIE 5-day vs Sentiment (Changes)")
plt.show()



WILL5000IND = dict_series["WILL5000IND"]

df_sent_WILL5000IND = pd.merge_asof(df_sentiment, WILL5000IND, left_on = "date", right_on = "DATE")

print(df_sent_WILL5000IND.tail(20) )

sns.lmplot( x= 'd1sentiment', y = "fp2", data = df_sent_WILL5000IND, fit_reg=True, legend=True)

plt.title("WILL5000IND 2-day vs Sentiment (Changes)")
plt.show()


sns.lmplot( x= 'd1sentiment', y = "fp5", data = df_sent_WILL5000IND, fit_reg=True, legend=True)

plt.title("WILL5000IND 5-day vs Sentiment (Changes)")
plt.show()



DGS10 = dict_series["DGS10"]

df_sent_DGS10 = pd.merge_asof(df_sentiment, DGS10, left_on = "date", right_on = "DATE")

print(df_sent_DGS10.tail(20) )

sns.lmplot( x= 'd1sentiment', y = "fp2", data = df_sent_DGS10, fit_reg=True, legend=True)

plt.title("DGS10 2-day vs Sentiment (Changes)")
plt.show()


sns.lmplot( x= 'd1sentiment', y = "fp5", data = df_sent_DGS10, fit_reg=True, legend=True)

plt.title("DGS10 5-day vs Sentiment (Changes)")
plt.show()
