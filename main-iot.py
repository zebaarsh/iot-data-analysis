
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
data = pd.read_csv("/Users/rosy/Desktop/data.txt", sep = ' ',header = None, names = ['sdate', 'stime','epoch','moteid','temp','humidity','light','voltage'],index_col=False, dtype='unicode')
data.head()


# In[5]:


data.tail()


# In[6]:


#data.columns = ['sdate', 'stime','epoch','moteid','temp','humidity','light','voltage']
data.to_csv('inteldata.csv')
#data.head(2000).to_csv('intelshort.csv')



# In[7]:


data.fillna(0, inplace=True)
#print(data)
data['epoch'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
data['epoch'] = data['epoch'].astype(int)
data.to_csv('intelclean.csv')


# In[8]:


#data['date'] = '2004-02-28'
#data['time'] = '00:59:16.02785'
#data.loc[data['date' == '2004-02-28']]
#data.groupby([''])
data

#df['timestamp'] = df[['sdate', 'stime']].apply(lambda x: ' '.join(x.astype(str)), axis=1)#


# In[9]:


df = data
df[['epoch', 'moteid','temp','humidity','light','voltage']] = df[['epoch', 'moteid','temp','humidity','light','voltage']].astype(float)
#df
df.describe()


# In[10]:




data['timestamp'] = data[['sdate', 'stime']].apply(lambda x: ' '.join(x.astype(str)), axis=1)


# In[11]:


df1= data
#df_1 = data
del df1['sdate']
del df1['stime']
df1


# In[12]:


df1.timestamp = pd.to_datetime(df1.timestamp)
df1.set_index('timestamp', inplace=True)
#df1.head()
df1.info()
df1.head()


# In[13]:


df1[['moteid','temp','humidity','light','voltage']] = df1[['moteid','temp','humidity','light','voltage']].apply(pd.to_numeric)


# In[14]:


df1.info()


# In[15]:


grouped_data = df1.groupby(['moteid'])
grouped_data['temp'].describe()


# In[18]:


grouped_data['humidity'].describe()


# In[19]:


grouped_data['light'].describe()


# In[20]:


grouped_data['voltage'].describe()


# In[21]:



df2= df1.groupby(['moteid']).corr()
df2.fillna(0, inplace=True)
df2


# ### Variation in humidity, temp, light, voltage with epoch

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(60,20))

for xcol, ax in zip(['humidity', 'temp', 'light','voltage'], axes):
    df.plot(kind='scatter', x='epoch', y=xcol, ax=ax, alpha=1, color='r')


# ### Variation in humidity, light, voltage with temperature

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,20))

for xcol, ax in zip(['humidity', 'light','voltage'], axes):
    df.plot(kind='scatter', x='temp', y=xcol, ax=ax, alpha=1, color='b')


# ### Variation in  temp, light, voltage with humidity

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,20))

for xcol, ax in zip(['temp', 'light','voltage'], axes):
    df.plot(kind='scatter', x='humidity', y=xcol, ax=ax, alpha=1, color='y')


# ### Variation in temp, humidity, voltage with light

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,20))

for xcol, ax in zip(['temp', 'humidity','voltage'], axes):
    df.plot(kind='scatter', x='light', y=xcol, ax=ax, alpha=1, color='g')


# ### Variation in humidity, temp, light with voltage

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60,20))

for xcol, ax in zip(['temp', 'light','humidity'], axes):
    df.plot(kind='scatter', x='voltage', y=xcol, ax=ax, alpha=1, color='r')


# ### Variation in temperature over time(averaged per hour)

# In[28]:


df1.index
index_hour = df1.index.hour
df1_by_hour =df1['temp'].groupby(index_hour).mean()
df1_by_hour.plot()
plt.show()


# In[29]:


ax = df1.plot(colormap='Dark2', figsize=(14, 7))

ax.set_xlabel('Date')

ax.set_ylabel('Variation with time')

df_summary = df1.describe()

# Specify values of cells in the table
ax.table(cellText=df_summary.values, 
          # Specify width of the table
          colWidths=[0.3]*len(df.columns), 
          # Specify row labels
          rowLabels=df_summary.index, 
          # Specify column labels
          colLabels=df_summary.columns, 
          # Specify location of the table
          loc='top') 

plt.show()


# ### Plot of variation in epoch, moteid, temp, humidity, light and voltage against timestamp(in the given order)

# In[30]:


df1.plot(subplots=True,
                linewidth=0.5,
                layout=(2, 4),
                figsize=(40, 20),
                sharex=False,
                sharey=False)

plt.show()


# ## Correlation between multivariate time series using pearson and spearman method 

# In[31]:


corr_mat_p = df1.corr(method='pearson')
print('************correlation matrix using pearson*************')
corr_mat_p




# In[32]:


corr_mat_sp = df1.corr(method='spearman')
print('************correlation matrix using pearson*************')
corr_mat_sp


# ### The following heatmap has the similar time series placed together

# In[33]:



import seaborn as sns
sns.clustermap(corr_mat_p)



# # Resampling time interval: per 31 seconds to per hour

# In[35]:


#find the variation in data per hour 


df1.index = df1.index.floor('1H')
df1.to_csv('intelhour.csv')
# df_time_mote= df1.groupby(['timestamp'])
# df_time_mote



# # Variation in temperature readings over time for moteid's: 2 and 10

# In[50]:


from matplotlib import pyplot as plt 
d_m2 = data.loc[data['moteid'] == 2.0]
d_m10 = data.loc[data['moteid'] == 10.0]

fig2 = plt.figure(figsize = (10,5))
d_m2['temp'].plot(label='temperature for moteid=2.0')
d_m10['temp'].plot(label='temperature for moteid=10.0')
fig2.suptitle('Variation in temperature over time for moteid= 2.0 and 10.0', fontsize=10)
plt.xlabel('timestamp', fontsize=10)
plt.ylabel('temperature', fontsize=10)
plt.legend()



# # Variation in light readings over time for moteid's: 2 and 10

# In[51]:


from matplotlib import pyplot as plt 
d_m2 = data.loc[data['moteid'] == 2.0]
d_m10 = data.loc[data['moteid'] == 10.0]

fig2 = plt.figure(figsize = (10,5))
d_m2['light'].plot(label='light for moteid=2.0')
d_m10['light'].plot(label='light for moteid=10.0')
fig2.suptitle('Variation in light over time for moteid= 2.0 and 10.0', fontsize=10)
plt.xlabel('timestamp', fontsize=10)
plt.ylabel('light', fontsize=10)
plt.legend()


# # Anomaly Detection using moving average method 

# ### For moteid:10 and window size: 20, we calculate the mean and standard deviation of the data.If the next entry in the dataframe lies between mean(+-)sd*2, it is considered normal else it is considered an anamoly

# In[73]:


from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
def mov_average(data, window_size):

    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')



# In[80]:


def find_anomalies(y, window_size, sigma=1.0):
    avg = mov_average(y, window_size).tolist()
    residual = y - avg
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}
    


# In[81]:


def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
   
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 40000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)
    events = {}
    events = find_anomalies(y, window_size=window_size, sigma=sigma_value)
    

    x_anom = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anom = np.fromiter(events['anomalies_dict'].values(), dtype=float,count=len(events['anomalies_dict']))
    plt.plot(x_anom, y_anom, "r*")
    print(x_anom)
    plt.grid(True)
    plt.show()


# In[82]:


x = d_m10['epoch']
Y = d_m10['temp']
plot_results(x, y=Y, window_size=50, text_xlabel="Date", sigma_value=3,text_ylabel="temperature")


# ### Note: Red points represent anomalies in temperature for moteid= 10

# ### Future work: In similar fashion, we can calculate anamolies in the humidity, light and voltage readings for different moteids. Also, using LSTM or RNN, we can predict future temperature,humidity,light and voltage readings based on the given time series data. 
