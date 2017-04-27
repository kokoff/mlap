import os
import random
import re
import sys
import math


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
    def __init__(self, rand_init=True, number_of_hidden_states=16, number_of_visible_states=3):
        self.hidden_states = number_of_hidden_states
        self.visible_states = number_of_visible_states

        if rand_init:
            self.initial = [random.random() for i in range(self.hidden_states)]
            self.transition = [[random.random() for i in range(self.hidden_states)] for j in range(self.hidden_states)]
            self.emission = [[random.random() for i in range(self.hidden_states)] for j in range(self.visible_states)]
        else:
            self.initial = [1.0 / self.hidden_states for i in range(self.hidden_states)]
            self.transition = [[1.0/8 for i in range(7)] + [1.0/72 for i in range(9)] for j in
                               range(self.hidden_states)]
            self.emission = [[1.0 / self.visible_states for i in range(self.hidden_states)] for j in
                             range(self.visible_states)]

        self.likelihood = 100

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
        DASHES = 135
        rep = ''
        rep += '-' * DASHES + '\n'
        rep += "Initial\n"
        rep += '-' * DASHES + '\n'
        rep += ''
        for i in range(self.hidden_states):
            rep += '(' + str(i / 4) + ',' + str(i % 4) + ')\t'
        rep += '\n\n'

        for i in self.initial:
            rep += self._prob_repr(i) + '\t'

        rep += '\n' + '-' * DASHES + '\n'
        rep += 'Transition\n'
        rep += '-' * DASHES + '\n'

        rep += ''
        for i in range(self.hidden_states):
            rep += '\t(' + str(i / 4) + ',' + str(i % 4) + ')'
        rep += '\n\n'

        count = 0
        for i in self.transition:
            rep += '(' + str(count / 4) + ',' + str(count % 4) + ')\t'
            count += 1
            for j in i:
                rep += self._prob_repr(j) + '\t'
            rep += '\n'

        rep += '-' * DASHES + '\n'
        rep += 'Emission\n'
        rep += '-' * DASHES + '\n'

        for i in range(self.hidden_states):
            rep += '\t(' + str(i / 4) + ',' + str(i % 4) + ')'
        rep += '\n\n'

        count = -1
        for i in self.emission:
            rep += '(' + str(count) + ')\t'
            count += 1
            for j in i:
                rep += self._prob_repr(j) + '\t'
            rep += '\n'
        rep += '-' * DASHES + '\n'
        return rep

    def getLikelihood(self, alpha):
        E = len(alpha)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        likelihood = 0

        for episode in alpha:
            T = len(episode)  # Length of episode
            summation = 0

            for state in range(N):
                episode[T-1][state]
                summation += episode[T-1][state]

            likelihood += math.log(summation)
        return likelihood

    def forward(self, episodes):
        N = self.hidden_states  # Number of hidden states
        alpha = []

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            # Calculate alpha(h1)
            for state in range(N):
                F[0][state] = self.emission[episode[0]][state] * self.initial[state]


            # Calculate alpha(ht)
            for t in range(1, T):
                for state in range(N):
                    summation = 0
                    for lastState in range(N):
                        summation += self.transition[state][lastState] * F[t - 1][lastState]

                    F[t][state] = self.emission[episode[t]][state] * summation

            alpha.append(F)

        return alpha

    def backward(self, episodes):
        N = self.hidden_states  # Number of states
        beta = []

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            # Calculate beta(h(T))
            for state in range(N):
                F[-1][state] = 1

            # Calculate beta(h(t-1))
            for t in reversed(range(1, T)):
                for previousState in range(N):
                    summation = 0
                    for state in range(N):
                        summation += self.emission[episode[t]][state] * self.transition[state][previousState] * \
                                     F[t][state]
                    F[t - 1][previousState] = summation

            beta.append(F)
        return beta

    def getGamma(self, alpha, beta):
        E = len(alpha)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        gamma = []

        for ep_num in range(E):
            T = len(alpha[ep_num])  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]

            for t in range(T):
                summation = 0

                # Calculate normalization factor
                for state in range(N):
                    summation += alpha[ep_num][t][state] * beta[ep_num][t][state]

                # Calculate p(h(t)|v(1:T))
                for state in range(N):
                    try:
                        F[t][state] = alpha[ep_num][t][state] * beta[ep_num][t][state] / summation
                    except ZeroDivisionError:
                        F[t][state] = 0

            gamma.append(F)
        return gamma

    def getXi(self, alpha, beta, episodes):
        E = len(episodes)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        xi = []

        for ep_num in range(E):
            T = len(episodes[ep_num])  # Length of episode
            F = [[[0 for i in range(N)] for j in range(N)] for k in range(T - 1)]

            for t in range(T - 1):
                summation = 0

                # Calculate normalization factor
                for state in range(N):
                    for nextState in range(N):
                        summation += alpha[ep_num][t][state] * self.emission[episodes[ep_num][t + 1]][nextState] * \
                                     self.transition[nextState][state] * beta[ep_num][t + 1][nextState]

                # Calculate p(h(t),h(t+1)|v(1:T))
                for state in range(N):
                    for nextState in range(N):
                        try:
                            F[t][state][nextState] = alpha[ep_num][t][state] * self.transition[nextState][state] * \
                                                     self.emission[episodes[ep_num][t + 1]][nextState] * \
                                                     beta[ep_num][t + 1][nextState] / summation
                        except ZeroDivisionError:
                            F[t][state][nextState] = 0

            xi.append(F)
        return xi

    def updateInitial(self, gamma):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        self.initial = [0 for i in range(N)]

        for episode in gamma:
            for state in range(N):
                self.initial[state] += episode[0][state] / E

    def updateTransition(self, gamma, xi):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        self.transition = [[0 for i in range(N)] for j in range(N)]

        for ep_num in range(E):
            T = len(xi[ep_num])  # Length of episode
            summations = [0 for i in range(N)]

            for t in range(T):
                for state in range(N):
                    summations[state] += gamma[ep_num][t][state]

            for t in range(T):
                for state in range(N):
                    for nextState in range(N):
                        try:
                            self.transition[nextState][state] += (xi[ep_num][t][state][nextState] / summations[
                                state]) / E
                        except ZeroDivisionError:
                            self.transition[nextState][state] += 0

    def updateEmission(self, gamma, episodes):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        self.emission = [[0 for i in range(N)] for j in range(3)]

        for ep_num in range(E):
            T = len(episodes[ep_num])  # Length of episode
            statesums = [[0 for i in range(T)] for j in range(N)]
            newemmission = [[0 for i in range(N)] for j in range(self.visible_states)]

            for t in range(T):
                for state in range(N):
                    statesums[state][t] += gamma[ep_num][t][state]

            for t in range(T):
                for state in range(N):
                    try:
                        newemmission[episodes[ep_num][t]][state] += gamma[ep_num][t][state]
                    except ZeroDivisionError:
                        newemmission[episodes[ep_num][t]][state] = 0

            for i in range(len(newemmission)):
                for j in range(len(newemmission[i])):
                    self.emission[i][j] += newemmission[i][j]

        for state in range(len(newemmission[i])):
            for v in range(len(newemmission)):
                self.emission[v][state] = self.emission[v][state] / (sum(statesums[state])) /E

    def baum_welch(self, episodes, verbouse=False):
        iteration = 0
        while True:
            iteration += 1
            if verbouse:
                print self.__str__()
                print 'Log-likelihood', self.likelihood, '\n'
                print 'Iteration:', iteration
            alpha = self.forward(episodes)
            beta = self.backward(episodes)
            gamma = self.getGamma(alpha, beta)
            xi = self.getXi(alpha, beta, episodes)
            self.updateInitial(gamma)
            self.updateEmission(gamma, episodes)
            self.updateTransition(gamma, xi)
            newLikelihood = self.getLikelihood(alpha)
            print 'Diferrence', self.likelihood - newLikelihood
            if abs(self.likelihood - newLikelihood) < 0.01:
                break

            self.likelihood = newLikelihood

        return iteration


def printF(F):
    print '\n\nF'
    for i in F:
        for j in i:
            print str(j) + '\t',
        print

    return


def task2(input_file):
    print 'Task2'
    print 'Input fille is:', input_file
    episodes = read_file(input_file)
    hmm = HMM(False)
    hmm.baum_welch(episodes,True)
    print hmm
    print 'Log-likelihood:', hmm.likelihood


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task2(input_file)
1
