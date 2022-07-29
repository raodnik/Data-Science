#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


x=np.linspace(0,5,11)
y=x**2


# In[6]:


x


# In[7]:


y


# In[8]:


plt.plot(x,y,'b-')
plt.xlabel('X Label')
plt.ylabel('Y Label')

plt.title('title')


# In[11]:


#subplots
plt.subplot(1,2,1)
plt.plot(x,y,'g-')

plt.subplot(1,2,2)
plt.plot(y,x,'y-')


# In[14]:


fig=plt.figure() #empty canvas

axes=fig.add_axes([0.1,0.1,0.8,0.8]) #set of axes
axes.plot(x,y)
axes.set_xlabel('X label')
axes.set_ylabel('Y label')


# In[16]:


fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes


axes1.plot(x, y, 'b')
axes1.set_xlabel('X_axes2')
axes1.set_ylabel('Y_axes2')
axes1.set_title('Axes 1')

axes2.plot(y, x, 'r')
axes2.set_xlabel('X_axes2')
axes2.set_ylabel('Y_axes2')
axes2.set_title('Axes 2');


# In[18]:


fig,axes=plt.subplots(nrows=1,ncols=2)

for curr_ax in axes: #iterating through array
    curr_ax.plot(x,y)


# In[19]:


fig,axes=plt.subplots(nrows=2,ncols=2)

plt.tight_layout()

axes[0,0].plot(x,y) 
axes[1,1].plot(y,x)
axes[0,1].plot(x,y**2)
axes[1,0].plot(y,x**3)


# In[20]:


fig.savefig('my_picture.png',dpi=210)


# In[21]:


fig=plt.figure()

ax=fig.add_axes([0,0,1,1])
ax.set_title('title')
ax.plot(x,x**3,label='x_lab')
ax.plot(x,x**2,label='y_lab')

ax.legend(loc=0) #default location of legend  


# In[22]:


fig=plt.figure()

ax=fig.add_axes([0,0,1,1])
ax.plot(x,y,color='purple',linewidth=3,linestyle='-',marker='o',alpha=0.8,markersize=20,markerfacecolor='yellow',markeredgecolor='red')
ax.set_xlim([0,10])
ax.set_ylim([0,30])


# In[23]:


x = np.arange(0,100)
y = x*2
z = x**2


# In[24]:


x


# In[25]:


y


# In[26]:


z


# In[29]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')


# In[30]:


from random import sample
data=sample(range(0,1000),100)
plt.hist(data, density=True,align='left',orientation='horizontal',rwidth=0.5,stacked=True)


# # Seaborn

# Distribution plot

# In[32]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


tips=sns.load_dataset('Tips')
tips.head()


# In[35]:


sns.displot(tips['total_bill'],kde=False,bins=50)


# In[36]:


#bivariate distplot
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')


# In[37]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')


# In[39]:


#for numerical columns
sns.pairplot(tips,hue='sex',palette='rainbow')


# In[40]:


tips.head()


# In[41]:


sns.rugplot(tips['total_bill'])


# In[43]:


#Classification
import numpy as np
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)


# In[44]:


tips.head(2)


# In[45]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='sex',palette='rainbow')


# In[46]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',split=True,palette='rainbow')


# In[47]:


sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',palette='Set1',dodge=True)


# In[48]:


sns.swarmplot(x='day',y='total_bill',data=tips,hue='sex',palette='Set1',dodge=True)


# In[49]:


sns.violinplot(x="tip", y="day", data=tips,palette='rainbow')
sns.swarmplot(x="tip", y="day", data=tips,color='black',size=3)


# # Matrixplot

# In[50]:


flights=sns.load_dataset('flights')
flights.head()


# In[51]:


tips.head(2)


# In[52]:


#heatmap
sns.heatmap(tips.corr(),cmap='rainbow',annot=True)


# In[53]:


#Pivot Table
ptflights=flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(ptflights,cmap='magma',linewidths=2)


# In[54]:


sns.clustermap(ptflights,cmap='jet',standard_scale=1)


# #Grids

# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


iris=sns.load_dataset('iris')
iris.head()


# In[58]:


#pairgrid
g=sns.PairGrid(iris)
g.map(plt.scatter)


# In[64]:


sns.pairplot(data=iris,hue='species',palette='rainbow')


# In[71]:


F=sns.FacetGrid(tips,col='time',row='smoker',hue='sex')
F.map(plt.scatter,"total_bill","tip").add_legend()


# #Regression Plot

# In[77]:


sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='rainbow',markers=['o','v'],scatter_kws={'s':50})


# In[79]:


sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',col='sex',palette='rainbow',markers=['o','v'],scatter_kws={'s':50})


# In[81]:


sns.lmplot(x='total_bill',y='tip',data=tips,row='time',col='sex')


# #Style and Color

# In[83]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


sns.set_style('white')
sns.countplot(x='sex',data=tips,palette='deep')


# In[87]:


sns.set_style('ticks')
sns.countplot(x='sex',data=tips,palette='deep')
sns.despine()


# In[90]:


sns.set_context('poster',font_scale=0.5)
sns.countplot('sex',data=tips,palette='coolwarm')


# # Data Visualization of Titanoc dataset using Seaborn

# In[91]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


titanic=sns.load_dataset('titanic')
titanic.head()


# In[94]:


sns.jointplot(x='fare',y='age',data=titanic)


# In[95]:


sns.displot(titanic['fare'],kde=False,color='red',bins=30)


# In[96]:


F=sns.FacetGrid(titanic,col='sex')
F.map(plt.hist,'age')


# # Pandas Built-in Data visualization

# In[97]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[108]:


df1=pd.read_csv('df1',index_col=0)
df2=pd.read_csv('df2')


# In[109]:


df1.head()


# In[111]:


df2.head()


# In[120]:


import matplotlib.pyplot as  plt
plt.style.use('ggplot')
plt.style.use('dark_background')
df1['A'].hist()


# In[122]:


plt.style.use('dark_background')
df2.plot.area(alpha=0.8)


# In[124]:


plt.style.use('dark_background')
df2.plot.bar(staticmethodcked=True)


# In[129]:


df1.plot.scatter(x='A',y='B')


# In[134]:


df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm')


# In[135]:


df2.plot.density()


# In[136]:


f=plt.figure()
df2.iloc[0:30].plot.area(alpha=0.4)
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.show()


# # Choropleth Maps

# In[5]:


import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[6]:


#to show graphs in the notebook
init_notebook_mode(connected=True)


# In[28]:


Map_data=dict(type='choropleth',
         locations=['AZ','NY','CA'],
         locationmode='USA-states',
         colorscale='Portland',
         text=['Arizona','NewYork','California'],
         z=[1.0,2.0,3.0],
         colorbar = {'title':'Colorbar Title'})
layout = dict(geo = {'scope':'usa'})


# In[29]:


choromap=go.Figure(data=[Map_data],layout=layout)


# In[30]:


iplot(choromap)


# # 2011 US AGRI Exports Us Map choropleth

# In[31]:


df = pd.read_csv('2011_US_AGRI_Exports')
df.head()


# In[36]:


#Building data dictionary
map_data = dict(type = 'choropleth',
             locations = df['code'],
             locationmode = 'USA-states',
             z = df['total exports'],
             text = df['text'],
             colorscale = 'ylorrd',
             colorbar = {'title':'Millions USD'},
             marker = dict(line = dict(color ='rgb(255,255,255)', width = 2)))


# In[42]:


layout = dict(title = '2011 US Agri Exports',
             geo={'scope':'usa','showlakes':True,'lakecolor':'rgb(85,173,240)'})


# In[43]:


choromap_agri = go.Figure(data=[map_data], layout=layout)


# In[45]:


iplot(choromap_agri)


# # 2014 World GDP Choropleth Map

# In[46]:


df1 = pd.read_csv('2014_World_GDP')
df1.head()


# In[47]:


data_1 = dict(type = 'choropleth',
             locations = df1['CODE'],
             z = df1['GDP (BILLIONS)'],
             colorbar = {'title':'GDP Billions USD'},
             text = df1['COUNTRY'])


# In[59]:


layout_1 = dict(title = '2014 World GDP',
                geo = dict(showframe=False, projection = {'type':'mercator'}))


# In[60]:


choromap_GDP = go.Figure(data = [data_1],layout = layout_1)


# In[61]:


iplot(choromap_GDP)


# In[ ]:




