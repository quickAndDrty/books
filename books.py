#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:14:51 2020

@author: andrac
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LinearRegression


books = pd.read_csv('book_data.csv')
books = books.drop('image_url', 1)
print(books)

#print(books["book_rating"])
#print(books.isnull().sum())
#books.hist(column="book_rating", range=[2.5, 5], bins=25)
'''
# remove rows with 0 reviews
books = books.loc[books['book_review_count'] != 0 ]
books.dropna(subset=['book_review_count'])

books = books.loc[books['book_rating_count'] != 0 ]
books.dropna(subset=['book_rating_count'])

books = books.loc[books['book_rating'] !=0]
books.dropna(subset=['book_rating'])

#checking of there are books with more reviews than ratings
count = 0       
for index, row in books.iterrows():
    if (row['book_review_count'] > row['book_rating_count']):
        count = count + 1
print("no of books with more reviews than ratings ", count)

books['percentage'] = (100 * books['book_review_count']) / books['book_rating_count']
books = books.loc[books['percentage'] < 100 ]

print(books['percentage'])
plt.xlabel('book rating')
plt.ylabel('feedback percentage')
plt.title('feedback percentage VS rating')
plt.scatter(books['book_rating'], books['percentage'],  alpha = 0.3, s = 7, linewidth = 0, c="#40A0C977")

count90p = 0
count4r = 0
count45r = 0
countS4r = 0
for index, row in books.iterrows():
    if (row['percentage'] > 90):
        count90p = count90p + 1
        if(row['book_rating'] >= 4):
            count4r = count4r + 1
        if (row['book_rating'] >= 4.5):
            count45r = count45r + 1
        if (row['book_rating'] < 4):
            countS4r = countS4r + 1
print("no of book with more than 90% ", count90p)
print("no of book with rating greater than 4 ", count4r)
print("no of book with rating greater than 4.5 ", count45r)
print("no of book with rating smaller than 4 ", countS4r)

count80p = 0
count4r2 = 0
count45r2 = 0
countS4r2 = 0
for index, row in books.iterrows():
    if (row['percentage'] > 80):
        count80p = count80p + 1
        if(row['book_rating'] >= 4):
            count4r2 = count4r2 + 1
        if (row['book_rating'] >= 4.5):
            count45r2 = count45r2 + 1
        if (row['book_rating'] < 4):
            countS4r2 = countS4r2 + 1
print("no of book with more than 80% ", count80p)
print("no of book with rating greater than 4 ", count4r2)
print("no of book with rating greater than 4.5 ", count45r2)
print("no of book with rating smaller than 4 ", countS4r2)

count75p = 0
count4r3 = 0
count45r3 = 0
countS4r3 = 0
for index, row in books.iterrows():
    if (row['percentage'] > 75):
        count75p = count75p + 1
        if(row['book_rating'] >= 4):
            count4r3 = count4r3 + 1
        if (row['book_rating'] >= 4.5):
            count45r3 = count45r3 + 1
        if (row['book_rating'] < 4):
            countS4r3 = countS4r3 + 1
print("no of book with more than 75% ", count75p)
print("no of book with rating greater than 4 ", count4r3)
print("no of book with rating greater than 4.5 ", count45r3)
print("no of book with rating smaller than 4 ", countS4r3)

count50p = 0
count4r4 = 0
count45r4 = 0
countS4r4 = 0
for index, row in books.iterrows():
    if (row['percentage'] > 50):
        count50p = count50p + 1
        if(row['book_rating'] >= 4):
            count4r4 = count4r4 + 1
        if (row['book_rating'] >= 4.5):
            count45r4 = count45r4 + 1
        if (row['book_rating'] < 4):
            countS4r4 = countS4r4 + 1
print("no of book with more than 50% ", count50p)
print("no of book with rating greater than 4 ", count4r4)
print("no of book with rating greater than 4.5 ", count45r4)
print("no of book with rating smaller than 4 ", countS4r4)

print("book review count 0 == ", books[books['book_review_count'] == 0].count() )
plt.xlabel('book_rating_count')
plt.ylabel('book_review_count')
plt.title('Review count VS rating count')

plt.scatter(books['book_rating_count'], books['book_review_count'],  linewidth = 0, c="#D06B36ff")


2.1 overview of genres and ratings
books["genres"]=books["genres"].str.split("|")
books = books.explode("genres").reset_index(drop=True)
#print(len(books["genres"].unique())) 867
#print(books.groupby(['genres']).count())
#print(books['genres'].value_counts().head(20))
mostPopularGenres = books['genres'].value_counts().head(20)
print("most popular genres ", mostPopularGenres)

genresAnalysis = ['Fiction', 'Fantasy', 'Romance', 'Young Adult', 'Historical',
                  'Paranormal', 'Mystery', 'Nonfiction', 'Science Fiction', 
                  'Historical Fiction', 'Classics', 'Contemporary', 'Childrens',
                  'Cultural', 'Literature', 'Sequential Art', 'Thriller', 'European Literature',
                  'Religion', 'History']
for genre in genresAnalysis:
    analysis = books.loc[books["genres"] == genre]
    analysis.hist(column="book_rating", range=[2.5, 5], bins=25)
    pl.title(genre)
    
    
#corr matrix    
print(books.isnull().sum())
print(books.info())

books = books.loc[books['genres'] != ' ' ]
books = books.dropna(subset=['genres', 'book_pages'])
pages = books['book_pages'].str.split(" ", n = 1, expand = True) 
books['noPages'] = pages[0].astype(int)
print(books.isnull().sum())
correlation_mat = books.corr()
sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of books data")
plt.xlabel("books features")
plt.ylabel("books features")
plt.show()

'''
#regression
# remove rows with 0 reviews
books = books.loc[books['book_review_count'] != 0 ]
books.dropna(subset=['book_review_count'])

books = books.loc[books['book_rating_count'] != 0 ]
books.dropna(subset=['book_rating_count'])
plt.scatter(books['book_rating_count'],books['book_review_count'], alpha=0.2, c="#00000022")
plt.show()

model = LinearRegression()

model.fit(books[['book_rating_count']],books[['book_review_count']])
print('intercept:', model.intercept_)
print('slope:', model.coef_)
r_sq=model.score(books[['book_rating_count']],books[['book_review_count']])
print("R2=",r_sq)


books['pred'] = model.predict(books[['book_rating_count']])

plt.scatter(books['book_rating_count'],books['book_review_count'], alpha=0.2,c="#77000022")
plt.scatter(books['book_rating_count'],books['pred'], alpha=0.2,c="#00770022")
plt.xlabel("rating count")
plt.ylabel("review count")
plt.show()

books['resid'] = books['pred']-books['book_review_count']
plt.boxplot(books['resid'])
plt.show()
plt.hist(books['resid'],bins=20)
plt.title("Distribution of the residuals")

plt.show()
books['resid'].describe()
print("Mean {:.12f}".format(books['resid'].mean()))#More readable number
print("Standard devation {:.12f}".format(books['resid'].std()))#More readable number
print("Mean review count {:.12f}".format(books['book_review_count'].mean()))
print("Median review count {:.12f}".format(books['book_review_count'].median()))

