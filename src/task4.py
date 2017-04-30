import os
import random

from HMM import HMM
from HMM import normalise_cond_prob
from HMM import read_file


def construct_restricted_transitions():
    # Only transitions to adjacent squares permitted
    # Walls constructed between squares - further restrict transitions
    walls = [((0, 1), (1, 1)),
             ((0, 2), (1, 2)),
             ((1, 0), (2, 0)),
             ((1, 2), (2, 2)),
             ((1, 3), (2, 3)),
             ((3, 1), (3, 2))]  # walls between squares

    transition = [[0.0 for i in range(16)] for j in range(16)]  # init empty transition matrix

    for y in range(4):
        for x in range(4):

            pos = x * 4 + y
            neighbours = [(x + 1, y),
                          (x - 1, y),
                          (x, y + 1),
                          (x, y - 1)]

            permitted_transitions = [(x, y)]  # Staying in same square is always permitted

            for n in neighbours:
                # Check if neighbour exists and weather it's separated by wall
                if -1 < n[0] < 4 and -1 < n[1] < 4 and \
                                ((n[0], n[1]), (x, y)) not in walls and \
                                ((x, y), (n[0], n[1])) not in walls:
                    permitted_transitions.append(n)

            # Set random probabilities for permitted transitions
            for n in permitted_transitions:
                transition[n[0] * 4 + n[1]][pos] = random.random()

    transition = normalise_cond_prob(transition)

    return transition


def task4(input_file):
    episodes = read_file(input_file)

    hmm = HMM(rand_init=True)
    hmm.transition = construct_restricted_transitions()

    hmm.baum_welch(episodes)
    print hmm


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task4(input_file)
