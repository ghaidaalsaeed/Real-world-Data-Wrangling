#!/usr/bin/env python
# coding: utf-8

# # Real-world Data Wrangling

# ### **1.1.** Problem Statement

# ## In this project, I aim to analyze the relationship between passenger demographics and their survival rates in the Titanic dataset. I will also explore the factors affecting wine quality in the Wine Quality dataset.

# #### **Dataset 1** Titanic Dataset
# 
# Type: CSV File
# Method: Download data manually from Kaggle.
# Dataset variables:
# PassengerId: Unique identifier for passengers.
# Survived: 1 if the passenger survived, 0 if not.
# Pclass: Passenger class (1, 2, or 3).
# Name: Passenger's name.
# Sex: Passenger's gender.
# Age: Passenger's age.
# SibSp: Number of siblings or spouses aboard.
# Parch: Number of parents or children aboard.
# Ticket: Ticket number.
# Fare: Ticket fare.
# Cabin: Cabin number.
# Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

# In[32]:


#FILL IN 1st data gathering and loading method
import pandas as pd
titanic = pd.read_csv('train.csv')
titanic.head(10)


# #### Dataset 2: Wine Quality Dataset
# 
# Type: CSV File
# Method: Programmatically downloaded from an online source.
# Dataset variables:
# type: Type of wine (red or white).
# fixed acidity: Fixed acidity level.
# volatile acidity: Volatile acidity level.
# citric acid: Citric acid content.
# residual sugar: Residual sugar content.
# chlorides: Chloride content.
# free sulfur dioxide: Free sulfur dioxide content.
# total sulfur dioxide: Total sulfur dioxide content.
# density: Density of the wine.
# pH: pH level.
# sulphates: Sulfates content.
# alcohol: Alcohol content.
# quality: Wine quality rating.
# 

# In[33]:


#FILL IN 2nd data gathering and loading method
wine = pd.read_csv('winequality.csv')
wine.head(10)


# ## 2. Assess data

# ### Quality Issue 1:

# In[34]:


#FILL IN - Inspecting the dataframe visually
titanic.describe()


# In[35]:


#FILL IN - Inspecting the dataframe programmatically
titanic.isnull().sum()


# 
# ### Issue: Missing values in the "Age" column of the Titanic dataset.
# ### Justification: This issue may affect our analysis of passenger demographics.

# ### Quality Issue 2:

# In[36]:


#FILL IN - Inspecting the dataframe visually
wine.describe()


# ### Tidiness Issue 1:

# In[37]:


# Fill missing values in the 'quality' column in the 'wine' DataFrame with the mean value
mean_quality = wine['quality'].mean()
wine['quality'].fillna(mean_quality, inplace=True)

# Fill missing values in the 'alcohol' column in the 'wine' DataFrame with the mean value
mean_alcohol = wine['alcohol'].mean()
wine['alcohol'].fillna(mean_alcohol, inplace=True)


# ### Issue: Filling missing values with the mean can distort the distribution of the variable. It assumes that missing values have the same characteristics as the observed values, which may not be true.
# 
# ### Justification: This approach is often chosen because it's simple and quick. It's justifiable when the assumption of missing-at-random holds, meaning that missing values are unrelated to the values themselves.
# 

# In[38]:


# Check the data types of columns in the Titanic dataset
titanic_dtypes = titanic.dtypes

# Check the data types of columns in the Wine Quality dataset
wine_dtypes = wine.dtypes

# Print the data types
print("Titanic Dataset Data Types:")
print(titanic_dtypes)

print("\nWine Quality Dataset Data Types:")
print(wine_dtypes)


# ### Tidiness Issue 2: 

# In[41]:


# Create a common key column with the same value for all rows in both DataFrames
titanic['key'] = 1
wine['key'] = 1

# Merge the DataFrames on the common key
combined_df = pd.merge(titanic, wine, on='key')
combined_df.drop('key', axis=1, inplace=True)


# In[42]:


#FILL IN - Inspecting the dataframe programmatically
# Create a new column 'dataset' to identify the source of each record
titanic['dataset'] = 'titanic'
wine['dataset'] = 'wine_quality'

# Concatenate the two datasets vertically
combined_dataset = pd.concat([titanic, wine], ignore_index=True)

# Display the first few rows of the combined dataset
combined_dataset.head(1000)


# ### Issue: Combining both datasets to answer the research questions.
# 
# ### Justification: Combining datasets allows us to explore relationships between passenger demographics, wine quality, and survival rates.

# ## 3. Clean data

# In[43]:


# FILL IN - Make copies of the datasets to ensure the raw dataframes 
# are not impacted
titanic_copy = titanic.copy()
wine_copy = wine.copy()


# ### **Quality Issue 1: FILL IN**

# In[44]:


# FILL IN - Apply the cleaning strategy
# Fill missing values in the "Age" column with the mean age
mean_age = titanic['Age'].mean()
titanic['Age'].fillna(mean_age, inplace=True)


# In[45]:


# FILL IN - Validate the cleaning was successful
titanic.isnull().sum()


# 
# Cleaning: Fill missing values in the "Age" column with the mean age.
# Validation: Check that there are no more missing values in the "Age" column.

# ### **Tidiness Issue 2: FILL IN**

# ### **Remove unnecessary variables and combine datasets**
# 
# Depending on the datasets, you can also peform the combination before the cleaning steps.

# In[46]:


#FILL IN - Remove unnecessary variables and combine datasets
#Remove unnecessary columns
titanic = titanic[['Age', 'Sex','Pclass','Survived']]
wine = wine[['alcohol', 'quality']]

# Step 3: Combine the datasets vertically (assuming the columns are in the same order)
combined_dataset = pd.concat([titanic, wine], ignore_index=True)


# In[47]:


combined_dataset.head(1000)


# ## 4. Update your data store
# Update your local database/data store with the cleaned data, following best practices for storing your cleaned data:
# 
# - Must maintain different instances / versions of data (raw and cleaned data)
# - Must name the dataset files informatively
# - Ensure both the raw and cleaned data is saved to your database/data store

# In[48]:


#FILL IN - saving data
raw_wine_filename = "raw_wine_data.csv"
cleaned_wine_filename = "cleaned_wine_data.csv"
raw_titanic_filename = "raw_titanic_data.csv"
cleaned_titanic_filename = "cleaned_titanic_data.csv"


# ## 5. Answer the research question
# 
# ### **5.1:** Define and answer the research question 
# Going back to the problem statement in step 1, use the cleaned data to answer the question you raised. Produce **at least** two visualizations using the cleaned data and explain how they help you answer the question.

# *Research question:* FILL IN from answer to Step 1 
# How does passenger demographics (age, gender, class) relate to survival rates in the Titanic dataset, and what factors affect wine quality in the Wine Quality dataset?

# In[49]:


#Visual 1 - FILL IN
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate survival rates by passenger class
survival_rates = titanic.groupby('Pclass')['Survived'].mean().reset_index()

# Create a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=survival_rates, palette='viridis')
plt.title('Survival Rates by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# *Answer to research question:* FILL IN 
# Visualization: A bar chart comparing survival rates for different passenger classes.
# Answer: We can observe if there's a correlation between passenger class and survival rates.

# In[52]:


#Visual 2 - FILL IN
# Create a scatterplot to visualize the relationship between alcohol content and wine quality
plt.figure(figsize=(8, 6))
sns.scatterplot(x='alcohol', y='quality', data=wine, color='green', alpha=0.7)
plt.title('Relationship Between Alcohol Content and Wine Quality')
plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.show()


# *Answer to research question:* FILL IN
# Visualization: A scatterplot showing the relationship between alcohol content and wine quality.
# Answer: We can determine if alcohol content affects wine quality.
# 

# ### **5.2:** Reflection
# If I had more time for this project, I would:
# 
# Investigate further data quality issues, such as outliers.
# Explore additional research questions, such as the impact of family size on survival rates.
# Perform more in-depth statistical analysis.
# Create more visualizations to gain insights.
# Conduct hypothesis testing to validate findings.
# 
