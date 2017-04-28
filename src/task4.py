import os
from HMM import HMM
from HMM import read_file


def task3(input_file):
    print 'Task4'
    print 'Input fille is:', input_file, '\n'
    episodes = read_file(input_file)
    walls = [(1, 5), (2, 6), (4, 8), (6, 10), (7, 11), (13, 14)]
    hmm = HMM(rand_init=True, restricted_transitions=True, walls=walls)
    hmm.baum_welch(episodes, False)
    print hmm

if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task3(input_file)
