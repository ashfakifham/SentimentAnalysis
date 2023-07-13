#!/usr/bin/env python
# coding: utf-8

# # MOHAMED ASHFAK MOHAMED IFHAN
# # 500196894
# # MIDTERM SENTIMENT ANALYSIS MODEL
# 
# 
# ## ADDITIONAL COMMENTS
# ###### I'm not confident with the deepNN implementation or understand how everything conncets in a deep NN architecture, so i have not attempted it. 
# ###### The below code consists of Naive Baye's model and data cleaning.

# #### Importing libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas.core.common import SettingWithCopyWarning
import warnings
from sklearn.metrics import roc_auc_score
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[2]:


## Importing the data set

tweets_df = pd.read_csv('sentiment_tweets_2022.csv', low_memory=False) #importing into a tweets dataframe
tweets_df.head()

#we dont want to see any warnings so suppressing it, importing straight gave  warning suggestion to implemet low_memory=false,
#which has been implemented


# # Basic Data Cleaning and EDA

# In[3]:


## since we want to analyze sentiments, we can easily disregard details such as username,location, etc.
## so we will create a new simpler dataframe

tweets_df = tweets_df[['Tweet','Sentiment']].copy() #taking only the raw tweet and the sentiment for each tweet.
tweets_df.head(5)


# In[4]:


## It was noticed on EXCEL that sentiments ranged between -1 to 1,
## so for simpler training, lets replace all above 0 with a 1 for positive,
## and all below 0 with a 0, for negative sentiment.

tweets_df.loc[tweets_df['Sentiment'] > 0, 'Sentiment'] = 1
tweets_df.loc[tweets_df['Sentiment'] < 0, 'Sentiment'] = 0
        
        
        
#reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
#https://stackoverflow.com/questions/31511997/pandas-dataframe-replace-all-values-in-a-column-based-on-condition


# In[5]:


tweets_df.head(5)


# In[6]:


#now that it is replaced and consistent, lets take a look at how many values we have in each sentiment

tweets_df.Sentiment.value_counts()


# ### Balancing the data sample

# #### we notice that there is about twice the data in positive sentiment than negative, and the data set is also huge
# #### so we take a smaller sample from each class
# #### but we need to make sure it is sorted in order

# In[7]:



tweets_df= tweets_df.sort_values(by =['Sentiment'],ascending=False) # sorting the sentiment

#create two df, one for positive and oen for negative
new_tweets_pos = tweets_df.head(10000)
new_tweets_neg = tweets_df.tail(10000)


new_tweets_pos = pd.DataFrame(new_tweets_pos) #output from .head() or .tail() was not a df, so converting it
new_tweets_neg = pd.DataFrame(new_tweets_neg) #output from .head() or .tail() was not a df, so converting it

#join the sample dataframe of smaller values
tweets_subset = pd.concat([new_tweets_pos, new_tweets_neg])
tweets_subset = tweets_subset.reset_index(drop=True) # as regex library gives error for out of order index


#https://stackoverflow.com/questions/37787698/how-to-sort-pandas-dataframe-from-one-column
#https://www.geeksforgeeks.org/get-last-n-records-of-a-pandas-dataframe/
#https://pandas.pydata.org/docs/reference/api/pandas.concat.html
#https://www.geeksforgeeks.org/get-last-n-records-of-a-pandas-dataframe/
#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


# In[8]:


#checking the balance of sentiments 
tweets_subset.Sentiment.value_counts() 


# In[9]:


#get a general idea of how many words per tweet. 

words_per_tweet= tweets_subset.Tweet.apply(lambda x: len(x.split(" "))) 
words_per_tweet.hist(bins = 30)

#we get the idea that most tweets range less than 20 words
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# In[10]:


#it is insightful to see a wordcloud of the most occuring words in a tweet
#it can be done for positive tweets and negative tweets using the wordcloud library

word_cloud_tweet = ''.join(new_tweets_pos.Tweet)
#joining all the words to create a word corpus ( one big sentence of all the words)
word_cloud = WordCloud(max_font_size=200,max_words=150,background_color="black",scale=10,width=800,height=800
                      ).generate(word_cloud_tweet)



plt.figure()
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#documentation says that it will auto convert it to tokens using regex
#https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# In[11]:


#Negative wordcloud

word_cloud_tweet = ''.join(new_tweets_neg.Tweet)
#joining all the words to create a word corpus ( one big sentence of all the words)
word_cloud = WordCloud(max_font_size=220,max_words=150,background_color="black",scale=10,width=800,height=800
                      ).generate(word_cloud_tweet)



plt.figure()
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#documentation says that it will auto convert it to tokens using regex
#https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# #### We can see some https words and a few elements that dont really make sense, so we will do some more data cleaning with regex later. We will observe how the model performs without any further cleaning

# #### Splitting the data for modeling

# In[12]:


## create x and y variables for the training and split


x = tweets_subset.Tweet
y = tweets_subset.Sentiment


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)
#splitting the data by 80-20 ratio


# # Model Building

# #### Vectorization, very important, as a model doesnt understand text
# #### this vectorization does its own tokenization inside, that is why we pass the entire tweet into it, rather than tokens

# In[13]:




vectorizer = TfidfVectorizer(max_features = 100, ngram_range = (1,2))
train_x_matrix = vectorizer.fit_transform(train_x)
test_x_matrix = vectorizer.transform(test_x)

train_x_matrix = train_x_matrix.toarray()  
test_x_matrix = test_x_matrix.toarray()


#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# ## Model initiation

# ### Naive Bayes

# In[14]:


#https://www.tutorialspoint.com/how-to-build-naive-bayes-classifiers-using-python-scikit-learn
#https://scikit-learn.org/stable/modules/naive_bayes.html
#https://www.tutorialspoint.com/how-to-build-naive-bayes-classifiers-using-python-scikit-learn
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


from sklearn.naive_bayes import GaussianNB
NBC = GaussianNB()  #initiate model , gaussian model is used for normally distributed data sets.

# Train the model:
NBmodel = NBC.fit(train_x_matrix, train_y)
print("Accuracy Naive Baye's Model 1 : ", NBC.score(test_x_matrix,test_y)) # print out accuracy of the model


# #### We see that the baseline model gives an accuracy of 68 % We could improve this with some data cleaning later
# 
# #### Lets look at some metrics

# In[15]:


# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report

predictions = NBC.predict(test_x_matrix)
confusion_matrix(predictions,test_y) #print confusion matrix

#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials
#https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6


# In[16]:


def confusion(ytest,y_pred):
    names=["Negative","Positive"]
    cm=confusion_matrix(ytest,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    return


# In[17]:


#gives a classification report of all the scores
print(classification_report(predictions,test_y))  


confusion(test_y,predictions)

#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# #### ROC value

# In[18]:


roc = roc_auc_score(test_y, NBC.predict_proba(test_x_matrix)[:, 1])
print(f"ROC AUC: {roc:.4f}")


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html


# ### Cleaning off the hyperlink values, special characters and userhandles in the tweets to observe how the model behaves

# #### Removing Hyperlinks

# In[54]:


rows = tweets_subset.shape[0]
hyperlinks = []
location = []
for twts in range(rows):
    x = re.findall(r'https?:\/\/.*[\r\n]*',tweets_subset.Tweet[twts]) #returns a list with hyperlink sequence found
    if x !=[]:
        hyperlinks.append(x)
        location.append(twts)#saving backups of the hyperlink and the location in row it belongs to
        tweets_subset.Tweet[twts] = re.sub(r'https?:\/\/.*[\r\n]*',"",tweets_subset.Tweet[twts])
    


# #### Removing special characters

# In[55]:


rows = tweets_subset.shape[0]
s_char = []
location_s_char = []
for twts in range(rows):
    x = re.findall(r'[^A-Za-z0-9 ]+',tweets_subset.Tweet[twts])
    if x !=[]:
        s_char.append(x)
        location_s_char.append(twts)#saving backups of the hyperlink and the location in row it belongs to
        tweets_subset.Tweet[twts] = re.sub(r'[^A-Za-z0-9 ]+',"",tweets_subset.Tweet[twts])
    


# #### Removing Userhandles

# In[56]:


rows = tweets_subset.shape[0]
user_handles = []
location_userhandles = []
for twts in range(rows):
    x = re.findall(r'@([a-zA-Z0-9_]{1,50})',tweets_subset.Tweet[twts])
    if x !=[]:
        user_handles.append(x)
        location_userhandles.append(twts)#saving backups of the hyperlink and the location in row it belongs to
        tweets_subset.Tweet[twts] = re.sub(r'@([a-zA-Z0-9_]{1,50})',"",tweets_subset.Tweet[twts])


# #### Making sure the HTTP links were removed by checking the wordcloud again, but here we use the overall subset

# In[19]:


#it is insightful to see a wordcloud of the most occuring words in a tweet
#it can be done for positive tweets and negative tweets using the wordcloud library

word_cloud_tweet = ''.join(tweets_subset.Tweet)
#joining all the words to create a word corpus ( one big sentence of all the words)
word_cloud = WordCloud(max_font_size=200,max_words=100,background_color="black",scale=10,width=800,height=800
                      ).generate(word_cloud_tweet)



plt.figure()
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#documentation says that it will auto convert it to tokens using regex
#https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# # Naive Bayes Model 2 after regex cleaning

# #### Splitting the data

# In[20]:


## create x and y variables for the training and split as the dataframe was overwritten in the above

x2 = tweets_subset.Tweet
y2 = tweets_subset.Sentiment

train_x, test_x, train_y, test_y = train_test_split(x2,y2,test_size=0.2,random_state=42)
#splitting the data by 80-20 ratio


# #### Vectorization

# In[21]:


## vectorizing the tweets
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials

vectorizer = TfidfVectorizer(max_features = 100, ngram_range = (1,2))
train_x_matrix = vectorizer.fit_transform(train_x)
test_x_matrix = vectorizer.transform(test_x)


# the python compiler suggest to convert back to array fom error


train_x_matrix = train_x_matrix.toarray()  # the python compiler suggest to convert back to array fom error
test_x_matrix = test_x_matrix.toarray()
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# #### Model Initiation

# In[22]:


from sklearn.naive_bayes import GaussianNB
NBC2 = GaussianNB()  #initiate model 

# Train the model:
NBmodel2 = NBC2.fit(train_x_matrix, train_y)
print("Accuracy Naive Baye's Model 2 :", NBC2.score(test_x_matrix,test_y)) # print out accuracy of the model


# #### Metrics

# In[23]:


# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report

predictions = NBC2.predict(test_x_matrix)
confusion_matrix(predictions,test_y) #print confusion matrix


#https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6


# In[24]:


print(classification_report(test_y,predictions))
confusion(test_y,predictions)

#gives a classification report of all the scores


# #### ROC value

# In[25]:


roc = roc_auc_score(test_y, NBC2.predict_proba(test_x_matrix)[:, 1])
print(f"ROC AUC: {roc:.4f}")


# ## Remove stop words to retrain a model

# #### Removing stop words as they dont contribute too much to sentiment, it is also possible to remove the most common words occuring, as part of top 5 percent of words etc. We see that the words "Freedom" and "Convoy" repeat alot from the word cloud. This is a potential set of words that can be removed. But it is not removed in this case

# In[26]:


#nltk.download('stopwords')  #commenting after downloading


# In[27]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
for i in range(len(tweets_subset)):
    tweet_temp=tweets_subset.loc[i, 'Tweet'] #extracting the tweet temporarily into a new variable
    tweet_tokens = nltk.word_tokenize(tweet_temp) #passing a list of tokens from the above temp variable to a new variable
    stop_word_tweet = [] #creating a variable to save the non stop words only from the tweet
    for word in tweet_tokens:
        if word.lower() not in stopwords:
            stop_word_tweet.append(word) #adding the non stop words into the list
    filtered_tweet = ' '.join(stop_word_tweet) #creating a summary of the tweet without the stop words
    tweets_subset.loc[i, 'Tweet'] = filtered_tweet #overwriting the tweet after removing the stop words

    
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# # Naive Bayes Model 3 after stop word removing

# #### Splitting the data

# In[28]:


## create x and y variables for the training and split


x3 = tweets_subset.Tweet
y3 = tweets_subset.Sentiment


train_x, test_x, train_y, test_y = train_test_split(x3,y3,test_size=0.2,random_state=42)
#splitting the data by 80-20 ratio


# #### Vectorization

# In[29]:


## vectorizing the tweets
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials

vectorizer = TfidfVectorizer(max_features = 100, ngram_range = (1,2))
train_x_matrix = vectorizer.fit_transform(train_x)
test_x_matrix = vectorizer.transform(test_x)


# the python compiler suggest to convert back to array fom error


train_x_matrix = train_x_matrix.toarray()  # the python compiler suggest to convert back to array fom error
test_x_matrix = test_x_matrix.toarray()
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# #### Model Initiation

# In[30]:


from sklearn.naive_bayes import GaussianNB
NBC3 = GaussianNB()  #initiate model 

# Train the model:
NBmodel3 = NBC3.fit(train_x_matrix, train_y)
print("Accuracy Naive Baye's Model 3 :", NBC3.score(test_x_matrix,test_y)) # print out accuracy of the model


# #### Metrics

# In[31]:


# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report

predictions = NBC2.predict(test_x_matrix)
confusion_matrix(predictions,test_y) #print confusion matrix


#https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6


# In[32]:


print(classification_report(test_y,predictions))  #gives a classification report of all the scores


# #### We notice the model is performing worse after the data cleaning

# #### ROC value

# In[33]:


roc = roc_auc_score(test_y, NBC3.predict_proba(test_x_matrix)[:, 1])
print(f"ROC AUC: {roc:.4f}")


# In[34]:


from sklearn.naive_bayes import MultinomialNB


# In[57]:


MNC = MultinomialNB()
MNC_model = MNC.fit(train_x_matrix,train_y)

print("Accuracy of the Model is: ",MNC_model.score(test_x_matrix,test_y))


# In[58]:


print(classification_report(test_y,predictions))  #gives a classification report of all the scores


# In[59]:


confusion(test_y,predictions)


# ### Deep learning model with TensorFlow

# In[41]:


## vectorizing the tweets
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials

vectorizer = TfidfVectorizer(max_features = 50, ngram_range = (1,7))
train_x_matrix = vectorizer.fit_transform(train_x)
test_x_matrix = vectorizer.transform(test_x)


# the python compiler suggest to convert back to array fom error


train_x_matrix = train_x_matrix.toarray()  # the python compiler suggest to convert back to array fom error
test_x_matrix = test_x_matrix.toarray()
#https://training.correlation-one.com/my-programs/COLM-DSCI-003/course-materials


# In[42]:


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM


# In[43]:


max_words = 500
embedding_dim = 50
max_len=50

dl_model = Sequential()
dl_model.add(Embedding(max_words,embedding_dim, input_length=max_len))
dl_model.add(LSTM(units=20))
dl_model.add(Dense(32,activation = 'relu'))
dl_model.add(Dense(1,activation = 'sigmoid'))


# In[44]:


dl_model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['acc'])
dl_model.summary()


# In[45]:


dlm = dl_model.fit(train_x_matrix,train_y,epochs = 5,batch_size=32,validation_split = 0.2)


# In[49]:


y_pred = dl_model.predict(test_x_matrix)
y_pred = (y_pred>0.5)


# In[52]:


from sklearn.metrics import accuracy_score
score=accuracy_score(test_y,y_pred)
print("Test Score:{:.2f}%".format(score*100))


# In[53]:


print(classification_report(test_y, y_pred))

cm = confusion_matrix(test_y,y_pred)
fig = plt.figure(figsize=(12,8))
sns.heatmap(
    cm,
    annot=True,
)
plt.title("Confusion Matrix")
cm


# In[ ]:




