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

# cd ~/Documents/NYC\ Data\ Science\ Academy/Week\ 8/feb25/ML_Lab

Orders = pd.read_csv("data/Orders.csv")
#Returns = pd.read_csv("data/Returns.csv")

#Orders.hist()
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
#sns.barplot(x = monthly_orders.index, y = monthly_orders['Quantity'])

# Problem 2, Question 2
monthly_cat_orders = Orders.groupby(["Order.Month" ,"Category"]).agg({"Quantity": "sum"})
monthly_cat_orders = monthly_cat_orders.reset_index(level = "Category")
monthly_cat_orders.index =  pd.Index(np.repeat(Months, 3), name = monthly_cat_orders.index.name)
monthly_cat_orders.reset_index(inplace = True)

#fg = sns.FacetGrid(monthly_cat_orders, col = "Category")
#fg = fg.map(sns.barplot, "Order.Month", 'Quantity')


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

## Step 1
#Orders['Order.ID'].isin(Returns['Order ID'])
merged_returns = pd.merge(Orders, Returns, how = "outer", left_on = "Order.ID",
                  right_on = "Order ID").drop(["Order ID", "Region_y"], axis = 1)
merged_returns['Returned'].fillna("No", inplace = True)


## Step 2
merged_returns['Process.Time'] = merged_returns['Ship.Date'] - merged_returns['Order.Date']
merged_returns['Process.Time'] = list(map(lambda x: x.days, merged_returns['Process.Time']))

# Step 3
merged_returns[merged_returns['Returned'] == "Yes"].groupby('Product.ID').agg({'Quantity':'sum'})
tmp = merged_returns.groupby(['Product.ID', 'Returned']).agg({'Quantity': "sum"}).reset_index(['Returned','Product.ID'])
tmp = tmp.rename(columns = {"Quantity": "Quantity.Returned"})

for row in range(0, len(tmp)):
    if (tmp.loc[row, 'Returned'] == "No"):
        tmp.loc[row, 'Quantity.Returned'] = 0

merged_returns = pd.merge(merged_returns, tmp, how = "outer", left_on = ["Product.ID", "Returned"],
         right_on = ["Product.ID", "Returned"])



##############################  PROBLEM 5 #####################################
merged_returns.columns

# If city shows up <= 50 times, categorize as "Other"
others = list(merged_returns['City'].value_counts().index[merged_returns['City'].value_counts() <= 50])
merged_returns['City'] = ['Other' if x in others else x for x in merged_returns['City']]

# If state shows up <= 25 times, categorize as "Other"
others = list(merged_returns['State'].value_counts().index[merged_returns['State'].value_counts() <= 25])
merged_returns['State'] = ['Other' if x in others else x for x in merged_returns['State']]

# If country shows up <= 10 times, categorize as "Other"
others = list(merged_returns['Country'].value_counts().index[merged_returns['Country'].value_counts() <= 10])
merged_returns['Country'] = ['Other' if x in others else x for x in merged_returns['Country']]


# Keeping Customer ID over Customer Name, in case there are two customers with the same name
# Product ID is probably more reliable than Product Name, but same information
# Postal code has many NAs, probably too granular for this purpose anyway
# Customer's probably aren't going to consider a company's profits when making a return
drop_columns = ['Row.ID', 'Order.ID', 'Customer.Name',
                'Postal.Code', 'Market', 'Product.Name',
                'Profit']

merged_returns.drop(columns = drop_columns, inplace= True)

y = merged_returns['Returned']

#Categorical Variable DF
merged_returns['Order.Month'] = merged_returns['Order.Month'].astype('str')
merged_returns['Process.Time'] = merged_returns['Process.Time'].astype('str')

CAT_DF = merged_returns[['Ship.Mode','Customer.ID','Segment',
                        'City','State','Country','Region_x',
                       'Product.ID','Category', 'Sub.Category',
                       'Order.Priority', 'Order.Month','Process.Time']]

#Dummify
dummy_df = pd.get_dummies(CAT_DF, prefix=['Ship.Mode','Customer.ID','Segment',
                    'City','State','Country','Region_x',
                  'Product.ID','Category','Sub.Category','Order.Priority',
                  'Order.Month','Process.Time'])

merged_return = merged_returns[['Sales','Quantity','Discount',
                               'Shipping.Cost']]

return_df = pd.concat([merged_return, dummy_df], axis=1, sort=False)
return_df.sample(10)
X = return_df


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

############################### Logistic Regression ###########################
cv5fold = StratifiedKFold(n_splits=5, random_state=0)
logit = LogisticRegression()
scores = cross_val_score(estimator = logit, X = X, y = y, cv = cv5fold)

scores
np.mean(scores)












