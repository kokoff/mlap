import os
from task2 import HMM
from task2 import read_file


def task3(input_file):
    print 'Task3'
    print 'Input fille is:', input_file
    episodes = read_file(input_file)
    print episodes
    return
    newlist = []
    for i in episodes:
         newlist.extend(i[1:])
    print newlist

    hmm = HMM(False)
    for i in range(1):
        hmm.baum_welch([newlist], True)
        print hmm
        print 'Log-likelihood:', hmm.likelihood
        print


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task3(input_file)
