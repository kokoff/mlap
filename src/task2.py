import os
from HMM import HMM
from HMM import read_file


def task2(input_file):
    episodes = read_file(input_file)
    hmm = HMM(rand_init=False)
    hmm.baum_welch(episodes)
    print hmm

if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task2(input_file)
