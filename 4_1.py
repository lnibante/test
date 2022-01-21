import numpy as np
import pandas as pd
import csv

X1 =[]
X2 =[]
X3 =[]
X4 =[]
X5 =[]
X6 =[]
X7 =[]
X8 =[]
X9 =[]
X10 =[]

n = 0
while n <1000:
    x1 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x11 = np.dot(x1,x1)
    if x11 < 1:
        X11 = x11
        X1.append(X11)
        n = n + 1

n = 0
while n <1000:
    x2 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x22 = np.dot(x2,x2)
    if x22 < 1:
        X22 = x22
        X2.append(X22)
        n = n + 1

n = 0
while n <1000:
    x3 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x33 = np.dot(x3,x3)
    if x33 < 1:
        X33 = x33
        X3.append(X33)
        n = n + 1

n = 0
while n <1000:
    x4 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x44 = np.dot(x4,x4)
    if x44 < 1:
        X44 = x44
        X4.append(X44)
        n = n + 1

n = 0
while n <1000:
    x5 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x55 = np.dot(x5,x5)
    if x55 < 1:
        X55 = x55
        X5.append(X55)
        n = n + 1

n = 0
while n <1000:
    x6 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x66 = np.dot(x6,x6)
    if x66 < 1:
        X66 = x66
        X6.append(X66)
        n = n + 1

n = 0
while n <1000:
    x7 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x77 = np.dot(x7,x7)
    if x77 < 1:
        X77 = x77
        X7.append(X77)
        n = n + 1

n = 0
while n <1000:
    x8 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x88 = np.dot(x8,x8)
    if x88 < 1:
        X88 = x88
        X8.append(X88)
        n = n + 1

n = 0
while n <1000:
    x9 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x99 = np.dot(x9,x9)
    if x99 < 1:
        X99 = x99
        X9.append(X99)
        n = n + 1

n = 0
while n <1000:
    x10 = np.random.uniform(-0.5, 0.5, 10)
#    X1 = np.linalg.norm(x1, ord=2)
    x100 = np.dot(x10,x10)
    if x100 < 1:
        X100 = x100
        X10.append(X100)
        n = n + 1

'''
r1 = np.array([1000] * 1000)
X1 = X1 +r1
X2 = X2 +r1
X3 = X3 +r1
X4 = X4 +r1
X5 = X5 +r1
X6 = X6 +r1
X7 = X7 +r1
X8 = X8 +r1
X9 = X9 +r1
X10 = X10 +r1
'''

df2 = pd.DataFrame(data=[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10],index=['1','2','3','4','5','6','7','8','9','10'])
df =df2.transpose()

with open('test1.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(X1)
    writer.writerow(X2)
    writer.writerow(X3)
    writer.writerow(X4)
    writer.writerow(X5)
    writer.writerow(X6)
    writer.writerow(X7)
    writer.writerow(X8)
    writer.writerow(X9)
    writer.writerow(X10)
df = pd.read_csv('test1.csv')
print(df2)
print(df2.shape)


