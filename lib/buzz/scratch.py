import numpy as np
from main import Buzz
from random import randrange


one = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
two = np.array([1, 0, 0, 1, 0, 0])
output = np.zeros(len(one) - len(two), dtype=np.uint32)

for i in range(0, len(one) - len(two)):
    output[i] = sum(one[i:i + len(two)] * two)

correl = np.correlate(one, two, mode='valid')
print(f'Correlation: len={len(correl)}, {correl}')
print(f'Summation: len={len(output)}, {output}')
sample_rate = 3600

sample_data = np.zeros(sample_rate, dtype=np.uint32)
for i in range(0, sample_rate):
    sample_data[i] = randrange(0, 65536)

print(f'Input array (len={len(sample_data)}): {sample_data[0:sample_rate//120]}')

Buzz._calculate_pps_fit_array(sample_data, sample_rate)
