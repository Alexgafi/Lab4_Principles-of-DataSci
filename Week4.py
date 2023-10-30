# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:51:41 2023

@author: adfw980
"""

import pandas as pd

import csv

f = open('C:/Users/adfw980/Downloads/censusCrimeClean.csv')

df = pd.DataFrame(csv.reader(f))

c1 = df[14]

med_income = c1

viol_crimes = df[101]

import scipy  

#skip the 1st row to avoid strings

med_income = med_income.iloc[1:]

viol_crimes = viol_crimes.iloc[1:]

#Check the distribution of these:
    
    import matplotlib.pyplot as plt
    
    plt.hist(med_income)
    
    plt.hist(viol_crimes)
    
    scipy.stats.pearsonr(med_income, viol_crimes)
    
    #Values are: PearsonRResult(statistic=-0.424220616726126, pvalue=6.334424808048533e-88)
    
    #The Data is Negatively Skewed, as shown by both histograms and QQ plots. We will try to nomralize it
    
    qplot = sm.qqplot(med_income, line='s', other=viol_crimes)
    
    import numpy as np

    trans_income = np.log(med_income)
    
    trans_crimes = np.log(viol_crimes)
    
    qqlot_log = sm.qqplot(trans_income, line='s', other=trans_crimes)
    
    ##QQplot seems to get closer to the normal distribution value
    
    ##The Spearman correlation result is significant: ignificanceResult(statistic=-0.482169950997556, pvalue=1.2978853095169542e-116)
    #plt.scatter(med_income, viol_crimes)
    
    scipy.stats.spearmanr(med_income, viol_crimes)
    
    plt.scatter(med_income, viol_crimes)
    
    plt.scatter(trans_income, trans_crimes)
    
    #Exercise2
    
    
f1 = open('C:/Users/adfw980/Downloads/heart.csv')

df1 = csv.reader(f)

df1 = csv.reader(f1)

df1 = pd.DataFrame(csv.reader(f1))

w_dis = df1[df1[13] == '0']

wo_dis = df1[df1[13] == '1']

mean_bp_w = w_dis[3].mean()

w_dis = w_dis.iloc[1:]


mean_bp_wo = wo_dis[3].mean()

std_bp_wo = wo_dis[3].std()

std_bp_w = w_dis[3].std()

plt.hist(w_dis[3])
plt.hist(wo_dis[3])

plt.plot(w_dis[3])
plt.plot(wo_dis[3])
plt.subplot(1,2,1)

from scipy.stats import ttest_ind

w_dis[3] = w_dis[3].astype(float)

wo_dis[3] = wo_dis[3].astype(float)

t_statistic, p_value = ttest_ind(w_dis[3], wo_dis[3])

#Cohen's D

pooled_SD = 1

cohens_d = (mean_bp_w - mean_bp_wo) / pooled_SD

import math 

cohens_d = math.sqrt( (137-1)*(std_bp_w**2) + (165-1) * (std_bp_wo**2)/ 137 + 165 -2 )

 # Exercise 3:
     #Count the number with the disease for each gender type
 hasDiseaseCount=df[df.hasDisease==True].groupby("gender").count().hasDisease

 #Count the number of gender type
 totalCount=df.groupby("gender").count()['hasDisease']

 #combine into a dataframe (both are indexed with gender, so will be matched) and specify the columns
 p=pd.concat([hasDiseaseCount, totalCount], axis=1)
 p.columns = ["heartDiseaseCount", "totalCount"]

 #create a new column and calculate the proportion
 p['propHeartDisease']=p["heartDiseaseCount"]/p["totalCount"]

 #print the results
 print(p.head())