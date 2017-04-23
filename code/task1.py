import os
import re


def task1(input_file):
    episodes = []
    episode = []
    state_visit_count = [[0 for i in range(4)] for j in range(4)]

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        p = re.compile(r'\((\d),(\d)\) (-?\d)')
        m = p.match(line)
        if m:
            x = int(m.group(1))
            y = int(m.group(2))
            reward = int(m.group(3))
            episode.append((x, y, reward))
            state_visit_count[x][y] += 1
        elif line == '\n':
            episodes.append(episode)
            episode = []
        else:
            print 'File not formatted properly.'
            return

    # Compute initial probabilities
    initial = [[[0, 0] for i in range(4)] for j in range(4)]
    for ep in episodes:
        initial[ep[0][0]][ep[0][1]][0] += 1
        initial[ep[0][0]][ep[0][1]][1] = len(episodes)

    print '\nInitial'
    for i in range(len(initial)):
        for j in range(len(initial[i])):
            print '(' + str(i) + ',' + str(j) + ')\t\t',
            if initial[i][j][1] != initial[i][j][0]:
                print str(initial[i][j][0]) + '\\' + str(initial[i][j][1]) + '\t'
            elif initial[i][j][0] == 0:
                print 0
            else:
                print 1

    transition_from_count = [[i for i in pizza] for pizza in state_visit_count]
    for ep in episodes:
        transition_from_count[ep[-1][0]][ep[-1][1]] -= 1

    # Compute transition probabilities
    transition = [[[0, 0] for i in range(16)] for j in range(16)]

    for ep in episodes:
        last = None
        for t in ep:
            if not last:
                last = t
            else:
                state = last[0] * 4 + last[1]
                transtate = t[0] * 4 + t[1]
                transition[state][transtate][0] += 1
                transition[state][transtate][1] = transition_from_count[last[0]][last[1]]
                last = t

    for row in range(len(transition)):
        all_zero = True
        for col in range(len(transition[row])):
            if transition[row][col][0] != 0:
                all_zero = False
                break
        if all_zero:
            transition[row] = [[1, 16] for i in range(len(transition))]

    print '\n\nTransition'
    print 'From/To   \t',
    for i in range(16):
        print '(' + str(i / 4) + ',' + str(i % 4) + ')\t',
    print
    print
    for i in range(len(transition)):
        print '(' + str(i / 4) + ',' + str(i % 4) + ')\t\t',
        for j in range(len(transition[i])):
            if transition[i][j][0] != 0 and transition[i][j][1] != transition[i][j][0]:
                print str(transition[i][j][0]) + '\\' + str(transition[i][j][1]) + '\t',
            elif transition[i][j][1] == 0:
                print '0\t',
            elif transition[i][j][1] == transition[i][j][0]:
                print '1\t',
        print

    # Compute emission probabilities
    emission = [[[0, 0] for i in range(3)] for j in range(16)]

    for ep in episodes:
        for t in ep:
            x = t[0] * 4 + t[1]
            r = t[2]
            emission[x][r + 1][0] += 1
            emission[x][r + 1][1] = state_visit_count[t[0]][t[1]]

    for row in range(len(emission)):
        all_zero = True
        for col in range(len(emission[row])):
            if emission[row][col][0] != 0:
                all_zero = False
                break
        if all_zero:
            emission[row] = [[1, 3] for i in range(len(emission[row]))]

    print "\n\nEmission"
    print 'State/Reward   ' + 1 * ' ' + '-1\t 0\t 1'
    print
    for i in range(len(emission)):
        print '(' + str(i / 4) + ',' + str(i % 4) + ')\t\t',
        for j in range(len(emission[i])):
            if emission[i][j][1] != 0 and emission[i][j][1] != emission[i][j][0]:
                print str(emission[i][j][0]) + '\\' + str(emission[i][j][1]) + '\t',
            elif emission[i][j][1] == 0:
                print '0\t',
            elif emission[i][j][1] == emission[i][j][0]:
                print '1\t',
        print


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task1.dat')
    print 'Input file is: ' + input_file
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task1(input_file)
