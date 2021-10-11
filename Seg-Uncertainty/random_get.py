import random

l = random.sample(range(0, 24966), 500)

result = []

with open('result.txt', 'w') as f:
    for i in l:
        f.write("{:05d}.png\n".format(i))






