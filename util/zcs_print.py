import sys

def R(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print("\033[31;1m", end = "")
    print(*objects, "\033[0m")
    # print("\033[0m", end = "")
    
def B(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print("\033[34;1m", end = "")
    print(*objects)
    print("\033[0m", end = "")
    
def Y(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    print("\033[33;1m", end = "")
    print(*objects)
    print("\033[0m", end = "")

if __name__ == "__main__":
    R("red", "RED")
    B("%s %d %.3f %d" % ("blue", 1314, 1314.520, 520))
    Y("cute")
