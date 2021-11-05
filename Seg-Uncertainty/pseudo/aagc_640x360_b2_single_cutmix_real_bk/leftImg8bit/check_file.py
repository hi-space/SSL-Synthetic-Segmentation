mypath = '/home/yoo/data/cityscapes/leftImg8bit/test'
# from os import listdir
# from os.path import isfile, join

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# print(onlyfiles)


import os
from glob import glob
result = [y for x in os.walk(mypath) for y in glob(os.path.join(x[0], '*.png'))]
print(result)

f = open('result.txt', 'w')
f.writelines(result)
f.close()
