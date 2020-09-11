class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n))
        self.avg = self.avg + (self.val * n / (self.count + n))
        self.count = self.count + n