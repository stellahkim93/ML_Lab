#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:01:45 2019

@author: stellakim
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# cd Documents/NYC\ Data\ Science\ Academy/Week\ 8/feb25/ML_Lab

Orders = pd.read_csv("data/Orders.csv")
#Returns = pd.read_csv("data/Returns.csv")

Orders.hist()
Orders.columns
################################  PART 1 ######################################
##############################  PROBLEM 1 #####################################
columns = ['Profit', 'Sales']
for col in columns:
    Orders[col] = Orders[col].str.replace("[$,]","").astype("float")
del (col, columns)

Orders.dtypes

columns = ['Order.Date', 'Ship.Date']
for col in columns:
    Orders[col] = pd.to_datetime(Orders[col], infer_datetime_format = True).dt.date
del (col, columns)

##############################  PROBLEM 2 #####################################
# Problem 2, Question 1
Orders['Order.Month'] = pd.Series(list(map(lambda x: x.month, Orders['Order.Date'])))
monthly_orders = Orders.groupby("Order.Month").agg({"Quantity": "sum"})
Months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_orders.index = Months
sns.barplot(x = monthly_orders.index, y = monthly_orders['Quantity'])

# Problem 2, Question 2
monthly_cat_orders = Orders.groupby(["Order.Month" ,"Category"]).agg({"Quantity": "sum"})
monthly_cat_orders = monthly_cat_orders.reset_index(level = "Category")
monthly_cat_orders.index =  pd.Index(np.repeat(Months, 3), name = monthly_cat_orders.index.name)
monthly_cat_orders.reset_index(inplace = True)

fg = sns.FacetGrid(monthly_cat_orders, col = "Category")
fg = fg.map(sns.barplot, "Order.Month", 'Quantity')


##############################  PROBLEM 3 #####################################
Returns = pd.read_csv("data/Returns.csv")


merged_returns = pd.merge(Orders, Returns, how = "inner", left_on = "Order.ID",
                  right_on = "Order ID").drop(["Order ID", "Region_y"], axis = 1)

# Problem 3, Question 1
merged_returns['Profit'].sum()
# $61370.75 in profits returned


# Problem 3, Question 2
name_count = {}
for name in merged_returns['Customer.Name']:
    name_count[name] = sum(merged_returns['Customer.Name'].str.find(name) != -1)

len({key:value for (key, value) in name_count.items() if value > 1})
# 449 customers returned more than one time
len({key:value for (key, value) in name_count.items() if value > 5})
# 124 customers returned more than five times

# Problem 3, Question 3
merged_returns['Region_x'].value_counts()

# Problem 3, Question 4
merged_returns['Sub.Category'].value_counts()

################################  PART 2 ######################################
##############################  PROBLEM 4 #####################################


