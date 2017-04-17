import os


def task2(input_file):
    print "task2 " + input_file
    return


if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'task2.dat')
    print 'Input file is: ' + input_file
    if not os.path.exists(input_file):
        print("Input file doesn't exist!")
    else:
        task2(input_file)
