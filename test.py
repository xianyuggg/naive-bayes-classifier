labels = [1,2,3,4,5]
dirs = [1,2,3,4,5]

import random
rand = random.randint(0, 100)
random.seed(rand)
random.shuffle(labels)
random.seed(rand)
random.shuffle(dirs)

print(labels, dirs)