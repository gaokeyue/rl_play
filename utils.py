import numpy as np
import time
from functools import wraps

def convex_comb(x, y, eta):
    """convex combination between x and y, eta in [0, 1]"""
    return x + eta * (y - x)

def expectation(prob_dict, f=None):
    if f is None:  # just the expectation of the probability distribution
        return sum(x * p for x, p in prob_dict.items())
    # if f is given, then E[f(X)]
    return sum(f[x] * p for x, p in prob_dict.items())


def compare(x, y, thresh=10 ** -8):
    d = x - y
    if d > thresh:
        return 1
    if d < -thresh:
        return -1
    return 0


def allmax(input_dict, thresh):
    best_score = -float('inf')
    best_keys = []
    for k, v in input_dict.items():
        flag = compare(v, best_score, thresh)
        if flag is 1:
            best_score = v
            best_keys = [k]
        elif flag is 0:
            best_keys.append(k)
    return best_keys, best_score


class Averager:

    def __init__(self):
        self.n = 0
        self.total = 0

    def __call__(self, x):
        self.add_new(x)

    def add_new(self, x):
        self.total += x
        self.n += 1

    @property
    def average(self):
        return self.total / self.n

    def __repr__(self):
        return repr(self.average)


def deco_timer(orig_fun):
    @wraps(orig_fun)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_fun(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{orig_fun.__name__} ran in {(t1 - t0):.6f} seconds")
        return result

    return wrapper


def stop_condition_inf(v1, v2, thresh):
    print(np.linalg.norm(v1 - v2))
    return all(abs(a - b) <= thresh for a, b in zip(v1, v2))


def stop_condition_real(x1, x2, thresh):
    return abs(x1 - x2) <= thresh


def deco_while(max_iter, print_iter, thresh,
               stop_condition=stop_condition_inf):
    def decorator(step):

        @wraps(step)
        def wrapper(x0, *args, **kwargs):
            n_iter = 0
            while n_iter < max_iter:
                n_iter += 1
                if n_iter % print_iter == 0:
                    print(f"Running iteration {n_iter}")
                x1 = step(x0, *args, **kwargs)
                to_exit = stop_condition(x1, x0, thresh)
                x0 = x1
                if to_exit:
                    print(f"Convergence achieved at iteration {n_iter}")
                    return x0
            else:
                print(f"{step.__name__} exceeded maximum iterations {max_iter}")
                return x0

        return wrapper

    return decorator


@deco_while(max_iter=10, thresh=10 ** -4, print_iter=1,
            stop_condition=stop_condition_real)  # shrink = deco_while(args)(shrink)
def shrink(x, alpha=.5):
    return alpha * x


@deco_timer  # wrapper = deco_timer(test_haha)
def test_haha():
    time.sleep(1)
    print('haha')
    return 'haha'


if __name__ == '__main__':
    print('haha')
    aa = Averager()
