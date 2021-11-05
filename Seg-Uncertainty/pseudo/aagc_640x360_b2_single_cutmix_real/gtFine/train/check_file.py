mypath = '.'

import os
from glob import glob
result = [y for x in os.walk(mypath) for y in glob(os.path.join(x[0], '*leftImg8bit_color.png'))]
print(result)

for f in result:
    newFile = f.replace('leftImg8bit_color.png', 'gtFine_color.png')
    # newFile = f.replace('gtFine_labelIds.png', 'leftImg8bit.png')
    os.rename(f, newFile)