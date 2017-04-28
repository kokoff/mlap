import os
import re

from HMM import HMM


def read_input_file(input_file):
    episodes = []
    episode = []
    state_visit_count = [0 for i in range(16)]
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        p = re.compile(r'\((\d),(\d)\) (-?\d)')
        m = p.match(line)
        if m:
            x = int(m.group(1))
            y = int(m.group(2))
            reward = int(m.group(3))
            episode.append((x * 4 + y, reward))
            state_visit_count[x * 4 + y] += 1
        elif line == '\n':
            episodes.append(episode)
            episode = []
        else:
            print 'File not formatted properly.'
            return
    return episodes, state_visit_count


def task1(input_file):
    episodes, state_visit_count = read_input_file(input_file)
    hmm = HMM(use_fractions=True)
    E = len(episodes)
    N = hmm.hidden_states
    V = hmm.visible_states

    # Compute initial probabilities
    hmm.initial = [0 for i in range(N)]
    for episode in episodes:
        hmm.initial[episode[0][0]] += 1.0 / E

    # Compute transition probabilities
    hmm.transition = [[0 for i in range(N)] for j in range(N)]

    norm = [0 for i in range(N)]
    for episode in episodes:

        for t in range(len(episode) - 1):
            state = episode[t][0]
            nextState = episode[t + 1][0]
            hmm.transition[nextState][state] += 1.0
            norm[state] += 1

    for nextState in range(N):
        for state in range(N):
            try:
                hmm.transition[nextState][state] /= norm[state]
            except ZeroDivisionError:
                continue

    # Compute emission probabilities
    hmm.emission = [[0 for i in range(N)] for j in range(V)]
    norm = [0 for i in range(N)]

    for episode in episodes:
        for timestep in episode:
            reward = timestep[1]
            state = timestep[0]
            hmm.emission[reward][state] += 1.0
            norm[state] += 1

    for reward in range(V):
        for state in range(N):
            try:
                hmm.emission[reward][state] /= norm[state]
            except ZeroDivisionError:
                continue

    print hmm
    return


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task1.dat')
    print 'Task1'
    print 'Input file is: ' + input_file
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task1(input_file)
