import sys

def R(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print('\033[31;1m', end = '')
    print(*objects, '\033[0m')
    
def B(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print('\033[34;1m', end = '')
    print(*objects, '\033[0m')
    
def Y(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print('\033[33;1m', end = '')
    print(*objects, '\033[0m')

if __name__ == '__main__':
    R('red', 'RED')
    B('%s %d %.3f %d' % ('blue', 1314, 1314.520, 520))
    Y('cute')
