# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:55:16 2019

@author: Harris Logic
"""

import pandas as pd 
#import os
#import csv
#
#SaveFile_Name = "allfeature.csv"
#work_dir = "A:\\cuckoo_json\\"
#
#readcsv1 = pd.read_csv(work_dir+'malfeature.csv')
##print(readcsv1)
#readcsv1.to_csv(work_dir+SaveFile_Name,index=False)


#readcsv1 = pd.read_csv(work_dir+'malfeature.csv')
#readcsv1.to_csv(work_dir+SaveFile_Name,index=False)
#
#readcsv2 = pd.read_csv(work_dir+'norfeature.csv')
#readcsv2.to_csv(work_dir+SaveFile_Name,index=False, header=False, mode='a+')

#with open(work_dir+'malfeature.csv',newline='') as csvopen:
#    reader1 = csv.reader(csvopen)
#    for row in reader1:
#        with open(work_dir+'norfeature.csv',"a",newline='') as csvopen2:
#            csvopen2.write(row)
            
#import pandas as pd
df1 = pd.read_csv("A:\\cuckoo_json\\malfeature.csv")
df2 = pd.read_csv("A:\\cuckoo_json\\norfeature.csv")
df = pd.concat([df1,df2],axis=0)
df = df.where(df.notnull(),0)
df.to_csv("A:\\cuckoo_json\\new.csv",index=None,mode="w")