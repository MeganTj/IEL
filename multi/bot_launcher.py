import argparse
from subprocess import Popen
import signal
import sys

CONFIG_FILE = "/Users/MeganT/Documents/mytest/IEL_config2.xlsx"

def signal_handler(sig, frame):
    terminate_processes()
    sys.exit()

def terminate_processes():
    for proc in processes:
        proc.kill()
        print("Process killed")

if __name__ == "__main__":
    # Parse command line arguments, which must  include
    # the name of the bots' python scripts and the number of them
    # to run
    parser = argparse.ArgumentParser(description='Run bots.')
    parser.add_argument('fname', type=str,
                        help='The name of the bot file without the number, i.e. '
                             'if the filename is IELv3_b1.py, enter IELv3_b')
    parser.add_argument('num_bots', metavar='N', type=int,
                        help='The number of bot scripts to run')
    parser.add_argument('-m', nargs=1, default=None,
                        help='The name of the manager script without the .py')
    parser.add_argument('-r', nargs=1, default=[1], type=int,
                        help='The number of runs to repeat the simulation for')
    args = parser.parse_args()
    print(args)
    global processes
    processes = []
    signal.signal(signal.SIGTSTP, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    for i in range(args.r[0]):
        if args.m is not None:
            processes.append(Popen(['python', str(args.m[0]) + '.py', CONFIG_FILE]))
        for i in range(1, args.num_bots + 1):
            processes.append(Popen(['python', args.fname + str(i) + '.py', CONFIG_FILE, str(i)]))
        if len(processes) > 0:
            processes[0].wait()
        terminate_processes()
        processes = []
