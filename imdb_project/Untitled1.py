
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from IPython.core.display import HTML
import seaborn as sns
import re
##get_ipython().magic('matplotlib inline')

css = open('style-table.css').read() + open('style-notebook.css').read()
HTML('<style>{}</style>'.format(css))


# In[2]:


titles = pd.DataFrame.from_csv('movie_metadata.csv', index_col=None)


# In[3]:


titles=titles.set_index('movie_title')


# In[4]:


titles['ratio']=titles.gross/titles.budget


# In[5]:


titles=titles=titles.dropna(axis=0, subset=['imdb_score','gross', 'budget', 'ratio', 'director_facebook_likes' , 'content_rating' , 'title_year' , 'movie_facebook_likes' ])


# In[6]:


titles.columns


# In[7]:


titles.content_rating.value_counts()


# In[8]:


((titles.imdb_score//1)).value_counts()


# In[9]:


titles.genres=titles.genres.str.replace("\|.*","")


# In[10]:


titles.genres.value_counts().plot(kind = 'bar')


# In[11]:


def normalize(array1, name):
    print ("normalizing "+name);
    array=array1
    array=array/array.std()
    array=array-array.mean()
    return(array);


# In[12]:


plt.scatter(titles.actor_1_facebook_likes , titles.ratio*100)
plt.xlim([0,2500])
plt.ylim([0,100000])
plt.show()


# In[13]:


plt.scatter(titles.imdb_score , titles.ratio*100)
#plt.xlim([0,2500])
plt.ylim([0,40000])
plt.show()


# In[14]:


plt.scatter(titles.director_facebook_likes , titles.ratio*100)
#plt.xlim([0,10000])
plt.ylim([0,40000])
plt.show()


# In[15]:


titles.imdb_score.plot.kde()


# In[16]:


titles.ratio.plot.kde(xlim=[-100,100])


# In[17]:


## removing movies with a very high profit ratio for which are mostly low budget movies
titles=titles.drop(titles[(titles.ratio>2000)].index)


# In[18]:


titles.ratio.mean()


# In[19]:


titles.ratio.plot.kde(xlim=[-10,100])


# In[20]:


print(titles.ratio.mean(),titles.ratio.std() )


# In[21]:


len(titles[(titles.ratio > 18)])


# In[22]:


(titles.ratio*100).plot.hist(xlim=[-10,1000],ylim=[0,100],bins=10000)


# In[23]:


(titles.ratio*100).plot.kde(xlim=[-10,1000],ylim=[0,100])


# In[24]:


(titles.ratio).plot.kde(xlim=[-10,10])


# In[25]:


len(titles[(titles.ratio > 10)])


# In[26]:


titles.ratio.std()


# In[27]:


titles['profit_making']= (titles.ratio > 1.2)


# In[28]:


titles.profit_making.value_counts()


# In[222]:


##  sending budget and likes through normalizer
titles.columns


# In[225]:


for column in ['budget','movie_facebook_likes','cast_total_facebook_likes','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','duration','num_voted_users', 'num_user_for_reviews', 'gross']:
    if np.issubdtype(titles[column].dtype, np.number):
        print (column + ": true")
        titles[column]=normalize(array1=titles[column],name=column)
    else:
        print (column + ":false")


# In[230]:


titles.budget.median()


# In[231]:


titles2 = pd.DataFrame.from_csv('movie_metadata.csv', index_col=None)


# In[236]:


titles2.budget.plot.kde(xlim=[-.4e9,.4e9])


# In[237]:


titles2.budget.mean()


# In[238]:


titles2.budget.std()


# In[247]:


temp1=((titles2.budget//1e6)*(1e6)).value_counts().sort_index()


# In[258]:


temp1=((titles.ratio//.5)*.5).value_counts().sort_index()


# In[310]:


temp1.plot.hist(bins=1000 , xlim=[0,100]  )


# In[337]:


temp1=((titles2.budget//1000000)*(1000000)).value_counts().sort_index()


# In[338]:


temp1.plot.hist(bins=1000 , xlim=[0,100]  )


# In[341]:


titles2.budget.mean()


# In[807]:


titles[titles.country.str.match("South Korea")].budget


# In[509]:


titles_highb = titles[(titles.budget > -.114)]


# In[510]:


titles_lowb = titles[(titles.budget <= -.114)]


# In[511]:


len(titles_highb)


# In[512]:


len(titles_lowb)


# In[513]:


titles_highb.profit_making.mean()


# In[809]:


titles_highb.sort_values('budget', ascending=False)


# In[514]:


titles_lowb.profit_making.mean()


# In[515]:


from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


# In[516]:


titles_highb=shuffle(titles_highb)


# In[517]:


titles_highb.head()


# In[518]:


titles_highb=titles_highb[['imdb_score','gross', 'budget', 'ratio', 'director_facebook_likes' , 'content_rating' , 'title_year' , 'movie_facebook_likes' ,'actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes']]


# In[519]:


titles_highb=titles_highb.dropna(axis=0, subset = ['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes'])


# In[520]:


titles_highb_train=titles_highb[:1500]
titles_highb_test=titles_highb[1500:]


# In[521]:


linear_regression_highb= LinearRegression()


# In[529]:


linear_regression_highb.fit(titles_highb_train[['title_year','budget','imdb_score','director_facebook_likes' ,  'actor_1_facebook_likes', 'actor_2_facebook_likes' , 'actor_3_facebook_likes']] , titles_highb_train.ratio)


# In[530]:


linear_regression_highb.predict(titles_highb_test[['title_year','budget','imdb_score','director_facebook_likes' ,  'actor_1_facebook_likes', 'actor_2_facebook_likes' , 'actor_3_facebook_likes']])[:10]


# In[531]:


titles_highb_test.ratio[:10]


# In[532]:


linear_regression_highb.score(titles_highb_test[['title_year','budget','imdb_score','director_facebook_likes' ,  'actor_1_facebook_likes', 'actor_2_facebook_likes' , 'actor_3_facebook_likes' ]],titles_highb_test.ratio)


# In[533]:


linear_regression_highb.coef_


# In[536]:


new=titles.genres.drop_duplicates()


# In[538]:


new.sort_values


# In[973]:


titles_cat=titles


# In[821]:


from pandas import get_dummies 


# In[822]:


type (titles_cat)


# In[827]:


titles_cat=pd.get_dummies(titles_cat , columns=["genres"])


# In[825]:


titles_cat.genres_Action.head()


# In[828]:


titles_cat.columns


# In[829]:


titles_cat.content_rating.value_counts()


# In[830]:


titles_cat.sort_values('budget', ascending=False).country


# In[831]:


titles_cat=titles_cat.drop(titles_cat[titles_cat.content_rating.str.match("NC-17|Passed|M|GP")].index)


# In[832]:


titles_cat.content_rating.value_counts()


# In[833]:


titles_cat=pd.get_dummies(titles_cat , columns=["content_rating"])


# In[939]:


titles_cat_highb = titles_cat[(titles_cat.budget > -.114)]


# In[940]:


titles_cat_lowb = titles_cat[(titles_cat.budget <= -.114)]


# In[941]:


len(titles_cat_highb)


# In[942]:


len(titles_cat_lowb)


# In[943]:


titles_cat_highb.profit_making.mean()


# In[944]:


titles_cat_lowb.profit_making.mean()


# In[946]:


titles_cat_highb=shuffle(titles_cat_highb)


# In[947]:


titles_cat_highb.columns


# In[948]:


#titles_cat_highb.drop(titles_cat_highb.country==)


# In[975]:


len(titles_cat_highb[titles_cat_highb.country != "USA"])


# In[950]:


titles_cat_highb=titles_cat_highb.drop(titles_cat_highb[titles_cat_highb.country != "USA"].index)


# In[951]:


#titles_cat_highb=titles_cat_highb.drop(['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes'],axis=1)


# In[974]:


titles_cat_highb.columns


# In[953]:


titles_cat_ml_highb=titles_cat_highb.select_dtypes(exclude=['object']) 


# In[954]:


titles_cat_ml_highb=titles_cat_ml_highb.select_dtypes(exclude=['bool']) 


# In[955]:


titles_cat_ml_highb=titles_cat_ml_highb.dropna()


# In[956]:


titles_cat_ml_highb=shuffle(titles_cat_ml_highb)


# In[957]:


titles_cat_ml_highb.columns


# In[959]:


titles_cat_ml_highb_train=titles_cat_ml_highb[:1200]
titles_cat_ml_highb_test=titles_cat_ml_highb[1200:]


# In[960]:


titles_cat_ml_highb_train_x=titles_cat_ml_highb_train.drop(['gross', 'ratio'],axis=1 )


# In[961]:


titles_cat_ml_highb_test_x=titles_cat_ml_highb_test.drop( ['ratio' , 'gross' ],axis=1)


# In[962]:


titles_cat_ml_highb_train_y=pd.DataFrame(titles_cat_ml_highb_train['ratio'])


# In[963]:


titles_cat_ml_highb_test_y=pd.DataFrame(titles_cat_ml_highb_test[ 'ratio' ])


# In[964]:


linear_regression_highb_cat= LinearRegression()


# In[965]:


linear_regression_highb_cat.fit(titles_cat_ml_highb_train_x ,titles_cat_ml_highb_train_y )


# In[966]:


linear_regression_highb_cat.predict(titles_cat_ml_highb_train_x)[:10]


# In[967]:


titles_cat_ml_highb_train_y[:10]


# In[968]:


linear_regression_highb_cat.score(titles_cat_ml_highb_test_x,titles_cat_ml_highb_test_y)


# In[969]:


linear_regression_highb_cat.coef_


# In[970]:


titles_cat_ml_highb_test_x.columns


# In[732]:


titles2 = pd.DataFrame.from_csv('movie_metadata.csv', index_col=None)


# In[733]:


titles2=titles2.set_index('movie_title')


# In[734]:



titles2['ratio']=titles2.gross/titles2.budget


# In[745]:


plt.ylim=[0,500]
plt.xlim=[0,.2e10]
plt.scatter(titles2.budget , titles2.ratio)


# In[898]:


titles2=titles2.drop(titles2[titles2.country!="USA"].index)


# In[903]:


plt.ylim=[0,500]
plt.xlim=[0,.2e10]
plt.scatter(titles2.budget , titles2.ratio)


# In[905]:


titles2.plot.scatter(x='imdb_score',y='gross')


# In[906]:


titles2.plot.scatter(x='budget',y='gross')


# In[916]:


titles2.plot.scatter(x='actor_1_facebook_likes',y='gross')


# In[918]:


titles2.actor_1_facebook_likes.std()


# In[971]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[972]:


logistic_regression_1= LogisticRegression()


# In[ ]:


logistic_regression_1.fit()


# In[985]:


titles_cat_ml_highb=titles_cat_highb.select_dtypes(exclude=['object']) 


# In[986]:


titles_cat_ml_highb_ratio=titles_cat_ml_highb.ratio


# In[987]:


titles_cat_logistic_highb=titles_cat_ml_highb.drop(['ratio'], axis=1)


# In[988]:


titles_cat_logistic_highb=titles_cat_logistic_highb.dropna()


# In[989]:


titles_cat_logistic_highb=shuffle(titles_cat_logistic_highb)


# In[990]:


titles_cat_logistic_highb_train=titles_cat_logistic_highb[:1200]
titles_cat_logistic_highb_test=titles_cat_logistic_highb[1200:]


# In[1117]:


titles_cat_logistic_highb_train_x.columns


# In[1118]:


titles_cat_logistic_highb_train_x=titles_cat_logistic_highb_train.drop(['gross', 'profit_making','movie_facebook_likes','num_voted_users'],axis=1 )


# In[1119]:


titles_cat_logistic_highb_test_x=titles_cat_logistic_highb_test.drop( ['gross', 'profit_making','movie_facebook_likes','num_voted_users' ],axis=1)


# In[1120]:


titles_cat_logistic_highb_train_y=pd.DataFrame(titles_cat_logistic_highb_train['profit_making'])


# In[1121]:


titles_cat_logistic_highb_test_y=pd.DataFrame(titles_cat_logistic_highb_test[ 'profit_making' ])


# In[1122]:


logistic_regression_1.fit(titles_cat_logistic_highb_train_x,titles_cat_logistic_highb_train_y.profit_making)


# In[1123]:


logistic_regression_1.predict(titles_cat_logistic_highb_test_x)[:10]


# In[1124]:


titles_cat_logistic_highb_test_y[:10]


# In[1125]:


accuracy_score(logistic_regression_1.predict(titles_cat_logistic_highb_test_x),titles_cat_logistic_highb_test_y.profit_making)


# In[1003]:


titles_cat_logistic_highb_test_x.index[:10]


# In[1027]:


logistic_regression_2=LogisticRegression()


# In[1010]:


titles_cat_lowb=titles_cat_lowb.drop(titles_cat_lowb[titles_cat_lowb.country!="USA"].index)


# In[1012]:


titles_cat_ml_lowb=titles_cat_lowb.select_dtypes(exclude=['object']) 


# In[1013]:


titles_cat_ml_lowb_ratio=titles_cat_ml_lowb.ratio


# In[1019]:


titles_cat_logistic_lowb=titles_cat_ml_lowb.drop(['ratio'], axis=1)


# In[1020]:


titles_cat_logistic_lowb=titles_cat_logistic_lowb.dropna()


# In[1021]:


titles_cat_logistic_lowb=shuffle(titles_cat_logistic_lowb)


# In[1022]:


titles_cat_logistic_lowb_train=titles_cat_logistic_lowb[:1200]
titles_cat_logistic_lowb_test=titles_cat_logistic_lowb[1200:]


# In[1126]:


titles_cat_logistic_lowb_train_x=titles_cat_logistic_lowb_train.drop(['gross', 'profit_making','movie_facebook_likes','num_voted_users'],axis=1 )


# In[1127]:


titles_cat_logistic_lowb_test_x=titles_cat_logistic_lowb_test.drop( ['gross', 'profit_making','movie_facebook_likes','num_voted_users'],axis=1)


# In[1128]:


titles_cat_logistic_lowb_train_y=pd.DataFrame(titles_cat_logistic_lowb_train['profit_making'])


# In[1129]:


titles_cat_logistic_lowb_test_y=pd.DataFrame(titles_cat_logistic_lowb_test[ 'profit_making' ])


# In[1130]:


logistic_regression_2.fit(titles_cat_logistic_lowb_train_x,titles_cat_logistic_lowb_train_y.profit_making)


# In[1131]:


logistic_regression_2.predict(titles_cat_logistic_lowb_test_x)[:10]


# In[1132]:


titles_cat_logistic_lowb_test_y[:10]


# In[1133]:


accuracy_score(logistic_regression_2.predict(titles_cat_logistic_lowb_test_x),titles_cat_logistic_lowb_test_y.profit_making)


# In[1134]:


array_coef_logistic=pd.DataFrame(logistic_regression_2.coef_)


# In[1135]:


array_coef_logistic.loc[1]=titles_cat_logistic_lowb_test_x.columns


# In[1136]:


array_coef_logistic=array_coef_logistic.T


# In[1137]:


array_coef_logistic=array_coef_logistic.set_index([1])


# In[1138]:


array_coef_logistic.sort_values(by=[0])


# In[1034]:


#get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import sklearn.model_selection

c0=sns.color_palette()[0]
c1=sns.color_palette()[1]
c2=sns.color_palette()[2]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=True, colorscale=cmap_light, 
                cdiscrete=cmap_bold, alpha=0.1, psize=10, zfunc=False, predicted=False):
    h = .02
    X=np.concatenate((Xtr, Xte))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    #plt.figure(figsize=(10,6))
    if zfunc:
        p0 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        p1 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z=zfunc(p0, p1)
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ZZ = Z.reshape(xx.shape)
    print(ZZ)
    if mesh:
        plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light, alpha=alpha, axes=ax)
    if predicted:
        showtr = clf.predict(Xtr)
        showte = clf.predict(Xte)
    else:
        showtr = ytr
        showte = yte
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=showtr-1, cmap=cmap_bold, 
               s=psize, alpha=alpha,edgecolor="k")
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=showte-1, cmap=cmap_bold, 
               alpha=alpha, marker="s", s=psize+10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax,xx,yy,ZZ

def points_plot_prob(ax, Xtr, Xte, ytr, yte, clf, colorscale=cmap_light, 
                     cdiscrete=cmap_bold, ccolor=cm, psize=10, alpha=0.1):
    ax,xx,yy,zz = points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=False, 
                           colorscale=colorscale, cdiscrete=cdiscrete, 
                           psize=psize, alpha=alpha, predicted=True) 
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=ccolor, alpha=.2, axes=ax)
    cs2 = plt.contour(xx, yy, Z, cmap=ccolor, alpha=.6, axes=ax)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14, axes=ax)
    return ax 

