import pypesq

def compute_pesq(ref, deg, fs=16000):
    return pypesq.pesq(ref, deg, fs)