import os
import random
import re
import sys
import math

coordinates = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
               4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3),
               8: (2, 0), 9: (2, 1), 10: (2, 2), 11: (2, 3),
               12: (3, 0), 13: (3, 1), 14: (3, 2), 15: (3, 3)}


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
    def __init__(self, rand_init=True, restricted_states=False, number_of_hidden_states=16, number_of_visible_states=3):
        self.hidden_states = number_of_hidden_states
        self.visible_states = number_of_visible_states
        self.likelihood = 0

        if rand_init:
            self.initial = [random.random() for i in range(self.hidden_states)]
            self.transition = [[random.random() for i in range(self.hidden_states)] for j in range(self.hidden_states)]
            self.emission = [[random.random() for i in range(self.hidden_states)] for j in range(self.visible_states)]

        else:
            self.initial = [1.0 / self.hidden_states for i in range(self.hidden_states)]
            self.transition = [[1.0 / self.hidden_states for i in range(self.hidden_states)] for j in
                               range(self.hidden_states)]
            self.emission = [[1.0 / self.visible_states for i in range(self.hidden_states)] for j in
                             range(self.visible_states)]

        if restricted_states:
            self.transition = [[0.0 for i in range(self.hidden_states)] for j in
                               range(self.hidden_states)]

            for y in range(4):
                for x in range(4):

                    pos = x * 4 + y
                    neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    forbidden = [(1, 5), (2, 6), (4, 8), (6, 10), (7, 11), (13, 14)]
                    neighbour_ok = []

                    for n in neighbours:
                        if -1 < n[0] < 4 and -1 < n[1] < 4 and (n[0] * 4 + n[1], pos) not in forbidden and \
                                        (pos, n[0] * 4 + n[1]) not in forbidden:
                            neighbour_ok.append(n)

                    norm = len(neighbour_ok) + 1

                    for n in neighbour_ok:
                        prob = 1.0/norm if not rand_init else random.random()
                        self.transition[n[0] * 4 + n[1]][pos] = 1.0 / norm

                    self.transition[pos][pos] = 1.0 / norm

    def _prob_repr(self, prob):
        return str(round(float(prob), 4))

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
            rep += self._prob_repr(i) + ' \t'

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
                rep += self._prob_repr(j) + ' \t'
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
                rep += self._prob_repr(j) + ' \t'
            rep += '\n'
        rep += '-' * DASHES
        return rep

    def forward(self, episodes):
        N = self.hidden_states  # Number of hidden states
        alpha = []
        loglik = 0  # Log-Likelihood

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]
            sumAlpha = 0

            # Calculate alpha(h1)
            for state in range(N):
                F[0][state] = self.emission[episode[0]][state] * self.initial[state]

            try:
                sumAlpha += math.log(sum(F[0]))
            except ValueError:
                sumAlpha += 0

            # Normalisation to avoid underflow
            try:
                F[0] = [i / sum(F[0]) for i in F[0]]
            except ZeroDivisionError:
                F[0] = [0 for i in range(N)]

            # Calculate alpha(ht)
            for t in range(1, T):
                for state in range(N):
                    summation = 0
                    for lastState in range(N):
                        summation += self.transition[state][lastState] * F[t - 1][lastState]

                    F[t][state] = self.emission[episode[t]][state] * summation

                try:
                    sumAlpha += math.log(sum(F[t]))
                except ValueError:
                    print 'Value Error'
                    sumAlpha += 0

                # Normalisation to avoid underflow
                try:
                    F[t] = [i / sum(F[t]) for i in F[t]]
                except ZeroDivisionError:
                    F[t] = [0 for i in range(N)]

            alpha.append(F)
            loglik = sumAlpha

        return alpha, loglik

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

                # Normalisation to avoid underflow
                try:
                    F[t - 1] = [i / sum(F[t - 1]) for i in F[t - 1]]
                except ZeroDivisionError:
                    F[t-1] = [0 for i in range(N)]

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
                norm = 0  # Normalisation factor

                # Calculate normalization factor
                for state in range(N):
                    norm += alpha[ep_num][t][state] * beta[ep_num][t][state]

                # Calculate p(h(t)|v(1:T))
                for state in range(N):
                    try:
                        F[t][state] = alpha[ep_num][t][state] * beta[ep_num][t][state] / norm
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
                norm = 0  # Normalisation factor

                # Calculate normalization factor
                for state in range(N):
                    for nextState in range(N):
                        norm += alpha[ep_num][t][state] * self.emission[episodes[ep_num][t + 1]][nextState] * \
                                self.transition[nextState][state] * beta[ep_num][t + 1][nextState]

                # Calculate p(h(t),h(t+1)|v(1:T))
                for state in range(N):
                    for nextState in range(N):
                        try:
                            F[t][state][nextState] = alpha[ep_num][t][state] * self.transition[nextState][state] * \
                                                     self.emission[episodes[ep_num][t + 1]][nextState] * \
                                                     beta[ep_num][t + 1][nextState] / norm
                        except ZeroDivisionError:
                            F[t][state][nextState] = 0

            xi.append(F)
        return xi

    def updateInitial(self, gamma):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        self.initial = [0 for i in range(N)]

        # Compute new initial (normalised)
        for episode in gamma:
            for state in range(N):
                self.initial[state] += episode[0][state] / E

    def updateTransition(self, gamma, xi):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        self.transition = [[0 for i in range(N)] for j in range(N)]
        norm = [0 for i in range(N)]  # Normalisation factor

        for ep_num in range(E):
            T = len(xi[ep_num])  # Length of episode

            # Compute new transition
            for t in range(T):
                for nextState in range(N):
                    for state in range(N):
                        self.transition[nextState][state] += xi[ep_num][t][state][nextState]
                        norm[state] += xi[ep_num][t][state][nextState]

        # Normalisation
        for row in range(N):
            for col in range(N):
                try:
                    self.transition[row][col] /= norm[col]
                except ZeroDivisionError:
                    self.transition[row][col] = 0.0

    def updateEmission(self, gamma, episodes):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        V = self.visible_states  # Number of visible states
        self.emission = [[0 for i in range(N)] for j in range(V)]
        norm = [0 for i in range(N)]  # Normalisation factor

        for ep_num in range(E):
            T = len(episodes[ep_num])  # Length of episode

            # Compute new emission
            for t in range(T):
                for state in range(N):
                    self.emission[episodes[ep_num][t]][state] += gamma[ep_num][t][state]
                    norm[state] += gamma[ep_num][t][state]

        # Normalisation
        for row in range(V):
            for col in range(N):
                try:
                    self.emission[row][col] /= norm[col]
                except ZeroDivisionError:
                    self.emission[row][col] = 0.0

    def baum_welch(self, episodes, verbouse=False, tolerance=0.01):
        iteration = 0
        alpha, self.likelihood = self.forward(episodes)
        oldLikelihood = self.likelihood + 2*tolerance

        if verbouse:
            print 'Initial'
            print self.__str__()
            print 'Log-Likelihood', self.likelihood
            print '-' * 135
        else:
            print 'Initial', '\t\t', 'Log-Likelihood', self.likelihood

        while abs(self.likelihood - oldLikelihood) > tolerance:
            if verbouse:
                print

            iteration += 1
            oldLikelihood = self.likelihood
            beta = self.backward(episodes)
            gamma = self.getGamma(alpha, beta)
            xi = self.getXi(alpha, beta, episodes)
            self.updateInitial(gamma)
            self.updateEmission(gamma, episodes)
            self.updateTransition(gamma, xi)
            alpha, self.likelihood = self.forward(episodes)

            if verbouse:
                print self.__str__()
                print 'Iteration:', iteration,'\n', 'Log-Likelihood', self.likelihood
                print '-' * 135
            else:
                print 'Iteration:', iteration, '\t\t', 'Log-Likelihood', self.likelihood

        return iteration


def main(input_file):
    print 'Task2'
    print 'Input fille is:', input_file
    episodes = read_file(input_file)
    hmm = HMM(rand_init=True, restricted_states=False)
    iterations = hmm.baum_welch(episodes, True)
    #print hmm
    #print 'Total number of iterations:', iterations,'\t\t\t', 'Final Log-Likelihood:', hmm.likelihood


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        main(input_file)
1
