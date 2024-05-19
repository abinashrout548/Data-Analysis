#!/usr/bin/env python
# coding: utf-8

# ## UBER Data analysis 

# In[1]:


##To importing all these libraries, we can use the  below code :


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


##Once downloaded, you can import the dataset using the pandas library.


# In[4]:


dataset = pd.read_csv(r"C:\Users\Abinash rout\Documents\Data Analyst\Uber Request Data.csv")


# In[5]:


dataset


# In[6]:


dataset.head()


# In[7]:


dataset.tail()


# In[8]:


dataset.head(1)


# In[9]:


dataset.shape


# In[10]:


dataset.info()


# In[11]:


##Data Preprocessing
##As we understood that there are a lot of null values in PURPOSE column, so for that we will me filling the null values with a NOT keyword. You can try something else too.


# In[12]:


dataset['Pickup point'].fillna("ISRO", inplace=True)


# In[13]:


dataset.info()


# In[14]:


##Changing the START_DATE and END_DATE to the date_time format so that further it can be use to do analysis.


# In[17]:


dataset['Request_timestamp'] = pd.to_datetime(dataset['Request_timestamp'], errors='coerce')
dataset['Drop_timestamp'] = pd.to_datetime(dataset['Drop_timestamp'], errors='coerce')


# In[18]:


##Splitting the START_DATE to date and time column and then converting the time into four different categories i.e. Morning, Afternoon, Evening, Night


# In[19]:


from datetime import datetime

dataset['date'] = pd.DatetimeIndex(dataset['Request_timestamp']).date
dataset['time'] = pd.DatetimeIndex(dataset['Request_timestamp']).hour


# In[20]:


###changing into categories of day and night
dataset['day_night'] = pd.cut(x=dataset['time'], bins = [0, 10,15,19,24] , labels = ['Morning', 'Afternoon', 'Evening', 'Night'])


# In[21]:


##Once we are done with creating new columns, we can now drop rows with null values.


# In[22]:


dataset.dropna(inplace=True)


# In[23]:


##It is also important to drop the duplicates rows from the dataset. To do that, refer the code below.


# In[24]:


dataset.drop_duplicates(inplace=True)


# In[25]:


#Data Visualization
##In this section, we will try to understand and compare all columns.

##Let’s start with checking the unique values in dataset of the columns with object datatype.


# In[26]:


obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_values = {}

for col in object_cols:
    unique_values[col] = dataset[col].unique().size
unique_values


# In[27]:


##Now, we will be using matplotlib and seaborn library for countplot the Status and Pickup point columns.


# In[28]:


plt.figure(figsize=(10,5))
 
plt.subplot(1,2,1)
sns.countplot(dataset['Status'])
plt.xticks(rotation=90)
 
plt.subplot(1,2,2)
sns.countplot(dataset['Pickup point'])
plt.xticks(rotation=90)


# In[29]:


##Let’s do the same for time column, here we will be using the time column which we have extracted above.


# In[30]:


sns.countplot(dataset['day_night'])
plt.xticks(rotation=90)


# In[31]:


#Now, we will be comparing the two different day_night along with the Pickup point of the user.


# In[32]:


plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='Pickup point', hue='day_night')
plt.xticks(rotation=90)
plt.show()


# In[33]:


#As we have seen that CATEGORY and PURPOSE columns are two very important columns. So now we will be using OneHotEncoder to categories them.


# In[34]:


from sklearn.preprocessing import OneHotEncoder
object_cols = ['Pickup point', 'day_night']
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)


# In[35]:


plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), 
            cmap='BrBG', 
            fmt='.2f', 
            linewidths=2, 
            annot=True)


# In[36]:


#Now, as we need to visualize the month data. This can we same as done before (for hours). 


# In[38]:


dataset['MONTH'] = pd.DatetimeIndex(dataset['Request_timestamp']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
               5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
               9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)
 
mon = dataset.MONTH.value_counts(sort=False)
 
# Month total rides count vs Month ride max count
df = pd.DataFrame({"MONTHS": mon.values,
                   "VALUE COUNT": dataset.groupby('MONTH',
                                                  sort=False)['x1_Morning'].max()})
 
p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")


# In[39]:


dataset.info()


# In[41]:


dataset['DAY'] = dataset.Request_timestamp.dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)


# In[42]:


day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label);
plt.xlabel('DAY')
plt.ylabel('COUNT')


# In[43]:


#We can use boxplot to check the distribution of the column.


# In[46]:


sns.boxplot(dataset['x0_Airport'])


# In[47]:


#As the graph is not clearly understandable. Let’s zoom in it for values lees than 100.


# In[49]:




sns.boxplot(dataset[dataset['x0_Airport']<100]['x0_Airport'])


# In[50]:


#It’s bit visible. But to get more clarity we can use distplot for values less than 40.


# In[52]:


sns.distplot(dataset[dataset['x0_Airport']<40]['x0_Airport'])


# In[ ]:




