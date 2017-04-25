import os
import random
import re
import sys


def read_file(input_file):
    episodes = []
    episode = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        p = re.compile(r'(-?\d)')
        m = p.match(line)
        if m:
            reward = int(m.group(1)) + 1  # +1 so they represent indexes
            episode.append(reward)
        elif line == '\n':
            episodes.append(episode)
            episode = []
        else:
            print 'File not formatted properly.'
            sys.exit(0)
    return episodes


class HMM:
    initial = []
    transition = []
    emission = []

    def __init__(self):
        self.initial = [random.random() for i in range(16)]
        self.transition = [[random.random() for i in range(16)] for j in range(16)]
        self.emission = [[random.random() for i in range(16)] for j in range(3)]

    def _prob_repr(self, prob):
        return str(round(float(prob), 4))

    def probs_sum_to_1(self):
        prob_sum = 0
        for i in self.initial:
            prob_sum += i

        for i in self.transition:
            for j in i:
                prob_sum += j

        for i in self.emission:
            for j in i:
                prob_sum += j

        if int(prob_sum) != len(self.initial) * 2 + 1:
            print prob_sum, len(self.initial) * 2 + 1
            return False
        else:
            print prob_sum, float(len(self.initial)) * 2 + 1, prob_sum - float(len(self.initial) * 2 + 1)
            return True

    def __str__(self):
        rep = "Initial\n"
        rep += 'ht\\ht-1'

        for i in self.initial:
            rep += self._prob_repr(i) + '\t'

        rep += '\nTransition\n'
        for i in self.transition:
            for j in i:
                rep += self._prob_repr(j) + '\t'
            rep += '\n'

        rep += '\nEmission\n'
        for i in self.emission:
            for j in i:
                rep += self._prob_repr(j) + '\t'
            rep += '\n'
        return rep

    def forward(self, episodes):
        N = len(self.initial)  # Number of states
        alpha = []

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            for state in range(N):
                F[0][state] = self.emission[episode[0]][state] * self.initial[state]  # Calculate alpha(h1)

            for t in range(1, T):
                for state in range(N):
                    summation = 0
                    for lastState in range(N):
                        summation += self.transition[state][lastState] * F[t - 1][lastState]
                    F[t][state] = self.emission[episode[t]][state] * summation  # Calculate alpha(ht)
            alpha.append(F)

        return alpha

    def backward(self, episodes):
        N = len(self.initial)  # Number of states
        beta = []

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            for state in range(N):
                F[-1][state] = 1

            for t in reversed([i for i in range(1, T)]):
                for lastState in range(N):
                    summation = 0
                    for state in range(N):
                        summation += self.emission[episode[t]][state] * self.transition[state][lastState] * \
                                     F[t][state]

                    F[t - 1][lastState] = summation

            beta.append(F)
        return beta

    # p(h|v)
    def getGamma(self, alpha, beta):
        E = len(alpha)
        N = len(self.initial)
        gamma = []

        for episode in range(E):
            T = len(alpha[episode])  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            for t in range(T):
                summation = 0
                for state in range(N):
                    summation += alpha[episode][t][state] * beta[episode][t][state]
                for state in range(N):
                    try:
                        F[t][state] = alpha[episode][t][state] * beta[episode][t][state] / summation
                    except ZeroDivisionError:
                        F[t][state] = 0

            gamma.append(F)

        return gamma

    # p(ht,ht+1|v)
    def getXi(self, alpha, beta, episodes):
        E = len(episodes)
        N = len(self.initial)
        xi = []

        for episode in range(E):
            T = len(episodes[episode])  # Length of episode
            F = [[[0 for i in range(N)] for j in range(N)] for k in range(T - 1)]

            for t in range(T - 1):
                summation = 0
                for state in range(N):
                    for nextState in range(N):
                        summation += alpha[episode][t][state] * self.emission[episodes[episode][t + 1]][nextState] * \
                                     self.transition[nextState][state] * beta[episode][t + 1][nextState]
                for state in range(N):
                    for nextState in range(N):
                        try:
                            F[t][state][nextState] = alpha[episode][t][state] * self.emission[episodes[episode][t + 1]][
                                nextState] * self.transition[nextState][state] * beta[episode][t + 1][
                                                         nextState] / summation
                        except ZeroDivisionError:
                            F[t][state][nextState] = 0
            # printF(F)
            xi.append(F)
        return xi

    def updateInitial(self, gamma):
        E = len(gamma)  # Number of episodes
        S = len(self.initial)
        self.initial = [0 for i in range(16)]

        for episode in gamma:
            for state in range(S):
                self.initial[state] += episode[0][state] / E

    def updateTransition(self, gamma, xi):
        E = len(gamma)  # Number of episodes
        S = len(self.initial)
        self.transition = [[0 for i in range(16)] for j in range(16)]

        for ep_num in range(E):
            T = len(xi[ep_num])  # Timesteps in episode

            summations = [0 for i in range(16)]

            for t in range(T):
                for state1 in range(S):
                    summations[state1] += gamma[ep_num][t][state1]

            for t in range(T):
                for state1 in range(S):
                    for state2 in range(S):
                        try:
                            self.transition[state2][state1] += (
                                                                   (xi[ep_num][t][state1][state2] / summations[
                                                                       state1])) / E
                        except ZeroDivisionError:
                            self.transition[state2][state1] += 0
        return

    def updateEmission(self, gamma, episodes):
        E = len(gamma)  # Number of episodes
        S = len(self.initial)  # Number of episodes
        self.emission = [[0 for i in range(16)] for j in range(3)]

        for ep_num in range(E):
            T = len(episodes[ep_num])  # Timesteps in episode
            statesums = [0 for i in range(S)]
            newemmission = [[0 for i in range(16)] for j in range(3)]

            for t in range(T):
                for state in range(S):
                    statesums[state] += gamma[ep_num][t][state]

            for t in range(T):
                for state in range(S):
                    try:
                        newemmission[episodes[ep_num][t]][state] += gamma[ep_num][t][state] / statesums[state]
                    except ZeroDivisionError:
                        newemmission[episodes[ep_num][t]][state] = 0
            for i in range(len(newemmission)):
                for j in range(len(newemmission[i])):
                    self.emission[i][j] += newemmission[i][j] / E
        return

    def baum_welch(self, episodes):
        alpha = self.forward(episodes)
        beta = self.backward(episodes)
        gamma = self.getGamma(alpha, beta)
        xi = self.getXi(alpha, beta, episodes)
        self.updateInitial(gamma)
        self.updateEmission(gamma, episodes)
        self.updateTransition(gamma, xi)


def printF(F):
    print '\n\nF'
    for i in F:
        for j in i:
            print str(j) + '\t',
        print
    return

    def beta(self):
        pass


def task2(input_file):
    print "task2 " + input_file
    episodes = read_file(input_file)
    hmm = HMM()
    print hmm
    print hmm.probs_sum_to_1()
    for i in range(1000):
        hmm.baum_welch(episodes)
        print hmm
        print hmm.probs_sum_to_1()
        print
    return


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    print 'Input file is: ' + input_file
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task2(input_file)
1
