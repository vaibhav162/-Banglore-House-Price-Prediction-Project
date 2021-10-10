#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]= (20,10)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression


# In[2]:


df= pd.read_csv(r"C:\Users\shruti\Desktop\Bangalore House Price Prediction\Bengaluru_House_Data.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# # Exploratory Data Analysis

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


# Groupby operation on Area Type

df.groupby("area_type")["area_type"].agg("count")


# In[10]:


df.info()


# In[11]:


# Dropping less important features

df= df.drop(["area_type", "society", "balcony", "availability"], axis=1)


# In[12]:


df.head()


# In[13]:


# Dropping Null values

df= df.dropna()


# In[14]:


df.isnull().sum()


# # Feature Engineering

# ### Analyzing "size"

# In[15]:


# Applying unique function on "size" feature

df["size"].unique()


# In[16]:


# Size has two different name method; BHK and Bedroom
# Creatring a new column to split it as integer

df["BHK"]= df["size"].apply(lambda x: int(x.split(" ")[0]))


# In[17]:


df.head()


# ### Analyzing "total_sqft"

# In[18]:


df.total_sqft.unique()


# In[19]:


# Exploring "total_sqft"

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[20]:


df[~df["total_sqft"].apply(is_float)].head(10)


# In[21]:


# Converting "sqft" to number and drop rows for "Sq. Meter"

def convert_sqft_to_number(x):
    tokens=x.split("-")
    if len(tokens)== 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[22]:


df= df.copy()
df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_number)
df.head(10)


# In[23]:


# Adding a new column named "Price_per_sqft"

df= df.copy()
df["price_per_sqft"]=df["price"]*100000/df["total_sqft"]


# In[24]:


df.head()


# In[25]:


# Reducing number of Laptions by using Dimensionalilty Reduction

df.location= df.location.apply(lambda x: x.strip())
location_stats= df["location"].value_counts(ascending=False)
location_stats


# In[26]:


len(location_stats[location_stats<=10])


# In[27]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[28]:


df.location= df.location.apply(lambda x:"other" if x in location_stats_less_than_10 else x)
len(df.location.unique())


# In[29]:


df.head()


# In[30]:


df[df.total_sqft/df.BHK<300].head()


# In[31]:


df= df[~(df.total_sqft/df.BHK<300)]
df.shape


# In[32]:


df.describe()


# In[33]:


# Removing Outliers using Mean and Standard Deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df = remove_pps_outliers(df)
df.shape


# # Data Visualization

# In[34]:


df.head()


# In[35]:


# Plotting 2 BHK & 3 BHK in a Scatter chart

def plot_scatter_chart(df, location):
    bhk2= df[(df.location==location) & (df.BHK==2)]
    bhk3= df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams["figure.figsize"]= (8,6)
    plt.scatter(bhk2.total_sqft, bhk2.price, color="blue", label="2 BHK", s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker="+", color="red", label="3 BHK", s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df, "1st Phase JP Nagar")


# In[36]:


# Plotting histogram for Price per SqureFeet Vs Count

plt.hist(df.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[37]:


# Plotting histogram for Price per SqureFeet Vs Count

plt.hist(df.bath, rwidth=0.8)
plt.xlabel("Numbers of Bathroom")
plt.ylabel("Count")


# In[38]:


df[df.bath>10]


# In[39]:


df[df.bath>df.BHK+2]


# In[40]:


df.head()


# In[41]:


df.shape


# # Using One Hot Encoding for Location

# In[42]:


dummies= pd.get_dummies(df.location)
dummies.head(20)


# # Concatinating both DataFrames together

# In[43]:


df= pd.concat([df,dummies.drop("other",axis="columns")],axis="columns")
df.head()


# In[44]:


df= df.drop("location", axis="columns")
df.head()


# In[45]:


X= df.drop(["price"], axis="columns")
df.head()


# In[46]:


X= df.drop(["size"], axis="columns")
df.head()


# In[47]:


y= df.price
y.head()


# In[48]:


X= X.drop(["price_per_sqft"], axis="columns")
df.head()


# In[49]:


X= X.drop(["price"], axis="columns")
df.head()


# In[50]:


X.shape


# In[51]:


y.shape


# # Train-Test Split

# In[52]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)


# In[53]:


lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[54]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)


# # Model Building

# In[55]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


# # Model Evaluation

# In[56]:


find_best_model_using_gridsearchcv=(X,y)


# # Model Testing

# In[57]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[58]:


predict_price("1st Phase JP Nagar", 1000, 2, 2)


# In[59]:


predict_price("Banashankari Stage V",2000, 3, 3)


# In[60]:


predict_price("2nd Stage Nagarbhavi",5000, 2, 2)


# In[61]:


predict_price("Indira Nagar",1500, 3, 3)


# # Conclusion
# 
# 
# ### From all the above models, we can clearly say that Linear Regression perform best for this dataset.
