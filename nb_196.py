#!/usr/bin/env python
# coding: utf-8

#  <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png" width = 300, align = "center"></a>
# 
# <h1 align=center><font size = 5>Data Analysis with Python</font></h1>

# # House Sales in King County, USA

# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

# <b>id</b> : A notation for a house
# 
# <b> date</b>: Date house was sold
# 
# 
# <b>price</b>: Price is prediction target
# 
# 
# <b>bedrooms</b>: Number of bedrooms
# 
# 
# <b>bathrooms</b>: Number of bathrooms
# 
# <b>sqft_living</b>: Square footage of the home
# 
# <b>sqft_lot</b>: Square footage of the lot
# 
# 
# <b>floors</b> :Total floors (levels) in house
# 
# 
# <b>waterfront</b> :House which has a view to a waterfront
# 
# 
# <b>view</b>: Has been viewed
# 
# 
# <b>condition</b> :How good the condition is overall
# 
# <b>grade</b>: overall grade given to the housing unit, based on King County grading system
# 
# 
# <b>sqft_above</b> : Square footage of house apart from basement
# 
# 
# <b>sqft_basement</b>: Square footage of the basement
# 
# <b>yr_built</b> : Built Year
# 
# 
# <b>yr_renovated</b> : Year when house was renovated
# 
# <b>zipcode</b>: Zip code
# 
# 
# <b>lat</b>: Latitude coordinate
# 
# <b>long</b>: Longitude coordinate
# 
# <b>sqft_living15</b> : Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# 
# <b>sqft_lot15</b> : LotSize area in 2015(implies-- some renovations)

# You will require the following libraries: 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Module 1: Importing Data Sets 

#  Load the csv:  

# In[10]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# 
# We use the method <code>head</code> to display the first 5 columns of the dataframe.

# In[3]:


df.head()


# ### Question 1 
# Display the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image. 

# In[4]:


df.dtypes


# We use the method describe to obtain a statistical summary of the dataframe.

# In[5]:


df.describe()


# # Module 2: Data Wrangling

# ### Question 2 
# Drop the columns <code>"id"</code>  and <code>"Unnamed: 0"</code> from axis 1 using the method <code>drop()</code>, then use the method <code>describe()</code> to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the <code>inplace</code> parameter is set to <code>True</code>

# In[11]:


df.drop('id', axis=1, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.describe()


# We can see we have missing values for the columns <code> bedrooms</code>  and <code> bathrooms </code>

# In[12]:


print(("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum()))
print(("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum()))


# 
# We can replace the missing values of the column <code>'bedrooms'</code> with the mean of the column  <code>'bedrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code>inplace</code> parameter to <code>True</code>

# In[13]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# 
# We also replace the missing values of the column <code>'bathrooms'</code> with the mean of the column  <code>'bathrooms' </code> using the method <code>replace()</code>. Don't forget to set the <code> inplace </code>  parameter top <code> True </code>

# In[14]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[15]:


print(("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum()))
print(("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum()))


# # Module 3: Exploratory Data Analysis

# ### Question 3
# Use the method <code>value_counts</code> to count the number of houses with unique floor values, use the method <code>.to_frame()</code> to convert it to a dataframe.
# 

# In[53]:


df_floor_counts = df['floors'].value_counts().to_frame()
df_floor_counts.reset_index(inplace=True)
df_floor_counts.rename(columns = {'floors': 'houses', 'index': 'floors'}, inplace = True)
df_floor_counts


# ### Question 4
# Use the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[54]:


sns.boxplot(x='waterfront', y='price', data=df)


# ### Question 5
# Use the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price.

# In[56]:


sns.regplot(x='sqft_above', y='price', data=df)


# 
# We can use the Pandas method <code>corr()</code>  to find the feature other than price that is most correlated with price.

# In[57]:


df.corr()['price'].sort_values()


# # Module 4: Model Development

# 
# We can Fit a linear regression model using the  longitude feature <code>'long'</code> and  caculate the R^2.

# In[58]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# ### Question  6
# Fit a linear regression model to predict the <code>'price'</code> using the feature <code>'sqft_living'</code> then calculate the R^2. Take a screenshot of your code and the value of the R^2.

# In[60]:


X1 = df[['sqft_living']]
lm1 = LinearRegression()
lm1.fit(X1, Y)
lm1.score(X1, Y)


# ### Question 7
# Fit a linear regression model to predict the <code>'price'</code> using the list of features:

# In[61]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# Then calculate the R^2. Take a screenshot of your code.

# In[62]:


lm2 = LinearRegression()
lm2.fit(df[features], Y)
lm2.score(df[features], Y)


# ### This will help with Question 8
# 
# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 
# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# The second element in the tuple  contains the model constructor 
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>
# 

# In[63]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# ### Question 8
# Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list <code>features</code>, and calculate the R^2.

# In[64]:


pipe = Pipeline(Input)
pipe.fit(df[features], Y)
pipe.score(df[features], Y)


# # Module 5: Model Evaluation and Refinement

# Import the necessary modules:

# In[65]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# We will split the data into training and testing sets:

# In[66]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print(("number of test samples:", x_test.shape[0]))
print(("number of training samples:",x_train.shape[0]))


# ### Question 9
# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data. 
# 

# In[67]:


from sklearn.linear_model import Ridge


# In[76]:


RidgeModel1 = Ridge(alpha = 0.1)
RidgeModel1.fit(x_train, y_train)
RidgeModel1.score(x_test, y_test)


# ### Question 10
# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

# In[75]:


RidgeModel2 = Ridge(alpha=0.1)
poly_train = PolynomialFeatures(degree=2)
poly_test = PolynomialFeatures(degree=2)
poly_train.fit_transform(x_train)
poly_test.fit_transform(x_test)
RidgeModel2.fit(x_train, y_train)
RidgeModel2.score(x_test, y_test)


# <p>Once you complete your notebook you will have to share it. Select the icon on the top right a marked in red in the image below, a dialogue box should open, and select the option all&nbsp;content excluding sensitive code cells.</p>
#         <p><img width="600" src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/save_notebook.png" alt="share notebook"  style="display: block; margin-left: auto; margin-right: auto;"/></p>
#         <p></p>
#         <p>You can then share the notebook&nbsp; via a&nbsp; URL by scrolling down as shown in the following image:</p>
#         <p style="text-align: center;"><img width="600"  src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/url_notebook.png" alt="HTML" style="display: block; margin-left: auto; margin-right: auto;" /></p>
#         <p>&nbsp;</p>

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/">Michelle Carey</a>, <a href="www.linkedin.com/in/jiahui-mavis-zhou-a4537814a">Mavis Zhou</a> 

# In[ ]:




