import os
import random
import re
import sys
import math
from fractions import Fraction


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


def normalise_cond_prob(cond_prob, norm=None):
    rows = len(cond_prob)
    cols = len(cond_prob[0])

    if not norm:
        # Compute normalisation factor
        norm = [0 for i in range(cols)]
        for i in range(rows):
            for j in range(cols):
                norm[j] += cond_prob[i][j]

    # Compute probabilities
    for i in range(rows):
        for j in range(cols):
            if norm[j] != 0:
                cond_prob[i][j] /= norm[j]
            else:
                cond_prob[i][j] = 0.0

    return cond_prob


def normalise_marginal_prob(prob, norm=None):
    if not norm:
        # Compute normalisation factor
        norm = sum(prob)

    # Compute probabilities
    if norm != 0:
        prob = [i / norm for i in prob]
    else:
        prob = [0.0 for i in prob]

    return prob


def normalise_joint_prob(prob, norm=None):
    rows = len(prob)
    cols = len(prob[0])

    if not norm:
        # Compute normalisation factor
        norm = sum([prob[i][j] for i in range(rows) for j in range(cols)])

    # Compute probabilities
    for i in range(rows):
        for j in range(cols):
            if norm != 0:
                prob[i][j] = prob[i][j] / norm
            else:
                prob[i][j] = 0.0

    return prob


class HMM:
    def __init__(self, rand_init=True, restricted_transitions=False, walls=[], number_of_hidden_states=16,
                 number_of_visible_states=3):
        self.hidden_states = number_of_hidden_states
        self.visible_states = number_of_visible_states
        self.likelihood = 0

        if rand_init:
            self.initial = [random.random() for i in range(self.hidden_states)]
            self.initial = normalise_marginal_prob(self.initial)

            self.transition = [[random.random() for i in range(self.hidden_states)] for j in range(self.hidden_states)]
            self.transition = normalise_cond_prob(self.transition)

            self.emission = [[random.random() for i in range(self.hidden_states)] for j in range(self.visible_states)]
            self.emission = normalise_cond_prob(self.emission)
        else:
            self.initial = [1.0 / self.hidden_states for i in range(self.hidden_states)]
            self.transition = [[1.0 / self.hidden_states for i in range(self.hidden_states)] for j in
                               range(self.hidden_states)]
            self.emission = [[1.0 / self.visible_states for i in range(self.hidden_states)] for j in
                             range(self.visible_states)]

        if restricted_transitions:
            self.transition = [[0.0 for i in range(self.hidden_states)] for j in
                               range(self.hidden_states)]

            for y in range(4):
                for x in range(4):

                    pos = x * 4 + y
                    neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    forbidden = walls
                    neighbour_ok = []

                    for n in neighbours:
                        if -1 < n[0] < 4 and -1 < n[1] < 4 and (n[0] * 4 + n[1], pos) not in forbidden and \
                                        (pos, n[0] * 4 + n[1]) not in forbidden:
                            neighbour_ok.append(n)

                    norm = len(neighbour_ok) + 1

                    for n in neighbour_ok:
                        prob = 1.0 / norm if not rand_init else random.random()
                        self.transition[n[0] * 4 + n[1]][pos] = prob

            self.transition = normalise_cond_prob(self.transition)

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
        E = len(episodes)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        alpha = []
        loglik = 0  # Average Log-Likelihood for each episode

        for episode in episodes:
            T = len(episode)  # Length of episode
            F = [[0 for i in range(N)] for j in range(T)]
            norm = 0  # Normalisations factor

            # Calculate alpha(h1)
            for state in range(N):
                F[0][state] = self.emission[episode[0]][state] * self.initial[state]
                norm += self.emission[episode[0]][state] * self.initial[state]

            # Normalisation to avoid underflow
            F[0] = normalise_marginal_prob(F[0], norm)

            # Calculate log-likelihood
            loglik += math.log(norm) if norm != 0.0 else 0.0

            # Calculate alpha(ht)
            for t in range(1, T):
                norm = 0  # Normalisations factor

                for state in range(N):
                    summation = 0
                    for lastState in range(N):
                        summation += self.transition[state][lastState] * F[t - 1][lastState]

                    F[t][state] = self.emission[episode[t]][state] * summation
                    norm += self.emission[episode[t]][state] * summation

                # Normalisation to avoid underflow
                F[t] = normalise_marginal_prob(F[t], norm)

                # Calculate log-likelihood
                loglik += math.log(norm) if norm != 0.0 else 0.0

            alpha.append(F)
        loglik /= E  # Average log-likelihood

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
                norm = 0  # Normalisation factor

                for previousState in range(N):
                    summation = 0
                    for state in range(N):
                        summation += self.emission[episode[t]][state] * self.transition[state][previousState] * \
                                     F[t][state]
                    F[t - 1][previousState] = summation

                    # Calculate normalisation factor
                    norm += summation

                # Normalisation to avoid underflow
                F[t - 1] = normalise_marginal_prob(F[t - 1], norm)

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

                for state in range(N):
                    # Calculate p(h(t)|v(1:T))
                    F[t][state] = alpha[ep_num][t][state] * beta[ep_num][t][state]

                    # Calculate normalisation factor
                    norm += alpha[ep_num][t][state] * beta[ep_num][t][state]

                # Normalisation
                F[t] = normalise_marginal_prob(F[t], norm)

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

                for state in range(N):
                    for nextState in range(N):
                        # Calculate p(h(t),h(t+1)|v(1:T))
                        F[t][state][nextState] = alpha[ep_num][t][state] * self.transition[nextState][state] * \
                                                 self.emission[episodes[ep_num][t + 1]][nextState] * \
                                                 beta[ep_num][t + 1][nextState]

                        # Calculate normalization factor
                        norm += alpha[ep_num][t][state] * self.transition[nextState][state] * \
                                self.emission[episodes[ep_num][t + 1]][nextState] * \
                                beta[ep_num][t + 1][nextState]

                # Normalisation
                F[t] = normalise_joint_prob(F[t], norm)

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
        norm = [0 for i in range(N)]  # Normalisation factors

        for ep_num in range(E):
            T = len(xi[ep_num])  # Length of episode

            # Compute new transition
            for t in range(T):
                for nextState in range(N):
                    for state in range(N):
                        self.transition[nextState][state] += xi[ep_num][t][state][nextState]
                        norm[state] += xi[ep_num][t][state][nextState]

        # Normalisation
        self.transition = normalise_cond_prob(self.transition, norm)

    def updateEmission(self, gamma, episodes):
        E = len(gamma)  # Number of episodes
        N = self.hidden_states  # Number of hidden states
        V = self.visible_states  # Number of visible states
        self.emission = [[0 for i in range(N)] for j in range(V)]
        norm = [0 for i in range(N)]  # Normalisation factors

        for ep_num in range(E):
            T = len(episodes[ep_num])  # Length of episode

            # Compute new emission
            for t in range(T):
                for state in range(N):
                    self.emission[episodes[ep_num][t]][state] += gamma[ep_num][t][state]
                    norm[state] += gamma[ep_num][t][state]

        # Normalisation
        self.emission = normalise_cond_prob(self.emission, norm)

    def baum_welch(self, episodes, tolerance=0.01):
        iteration = 0

        # first E-step
        alpha, self.likelihood = self.forward(episodes)
        beta = self.backward(episodes)
        gamma = self.getGamma(alpha, beta)
        xi = self.getXi(alpha, beta, episodes)

        oldLikelihood = self.likelihood + 2 * tolerance
        print 'Initial:', '\t\t', 'Log-Likelihood', self.likelihood

        while abs(self.likelihood - oldLikelihood) >= tolerance:
            iteration += 1
            oldLikelihood = self.likelihood

            # M-Step
            self.updateInitial(gamma)
            self.updateEmission(gamma, episodes)
            self.updateTransition(gamma, xi)

            # E-step
            alpha, self.likelihood = self.forward(episodes)
            beta = self.backward(episodes)
            gamma = self.getGamma(alpha, beta)
            xi = self.getXi(alpha, beta, episodes)

            print 'Iteration:', iteration, '\t\t', 'Log-Likelihood', self.likelihood

        return iteration, self.likelihood
