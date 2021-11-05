mypath = '.'

import os
from glob import glob
result = [y for x in os.walk(mypath) for y in glob(os.path.join(x[0], '*.png'))]
print(result)

for f in result:
    newFile = f.replace('leftImg8bit.png', 'gtFine_labelIds.png')
    os.rename(f, newFile)