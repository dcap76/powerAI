import pandas
import random

n = 1864 #number of records in file
s = 100 #desired sample size
filename = "result.csv"
skip = sorted(random.sample(xrange(n),n-s))
a = range(0, n)
b = list(set(a) - set(skip))
df = pandas.read_csv(filename, sep="|", skiprows=skip)
df2 = pandas.read_csv(filename, sep="|", skiprows=b)
df.to_csv(path_or_buf="1.csv", sep='|', encoding='utf-8')
df2.to_csv(path_or_buf="2.csv", sep='|', encoding='utf-8')