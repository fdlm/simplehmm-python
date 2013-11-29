from math import log
from itertools import izip


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B


def normalise(l):
    norm_const = sum(l)
    return map(lambda x: x / norm_const, l), norm_const


def find_max(l):
    m = max(l)
    i = l.index(m)
    return m, i


def forward(model, observations):
    state_idxs = range(len(model.pi))
    log_prob = 0.

    alphas = [[model.pi[i] * model.B[i][observations[0]] for i in state_idxs]]
    alphas[0], nc = normalise(alphas[0])
    log_prob += log(nc)

    for obs in observations[1:]:
        alphas += [[sum([alphas[-1][j] * model.A[j][i] for j in state_idxs]) * model.B[i][obs] for i in state_idxs]]
        alphas[-1], nc = normalise(alphas[-1])
        log_prob += log(nc)

    return alphas, log_prob


def backward(model, observations):
    state_idxs = range(len(model.pi))
    betas = [[1] * len(model.pi)]

    for obs in observations[::-1]:
        beta = [sum([betas[0][j] * model.B[j][obs] * model.A[i][j] for j in state_idxs]) for i in state_idxs]
        betas.insert(0, normalise(beta)[0])

    return betas


def forward_backward(model, observations):
    alphas, logprob = forward(model, observations)
    betas = backward(model, observations)

    gammas = [normalise([a * b for a, b in izip(alpha, beta)])[0] for alpha, beta in izip(alphas, betas[1:])]
    return gammas, logprob


def viterbi(model, observations):
    state_idxs = range(len(model.pi))
    deltas = [[]]
    psis = []

    deltas[0] = forward(model, observations[:1])[0][0]

    for obs in observations[1:]:
        trans, from_state = izip(*[find_max([deltas[-1][j] * model.A[j][i] for j in state_idxs]) for i in state_idxs])
        deltas += [normalise([trans[i] * model.B[i][obs] for i in state_idxs])[0]]
        psis += [from_state]

    path = [find_max(deltas[-1])[1]]
    for psi in psis[::-1]:
        path.insert(0, psi[path[0]])

    return path


if __name__ == "__main__":
    A = [[0.5, 0.3, 0.2],[0.2, 0.6, 0.2],[0.1, 0.2, 0.7]]
    B = [[0.1, 0.9], [0.4, 0.6], [0.9, 0.1]]
    pi = [0.3, 0.5, 0.2]

    obs = [1, 0, 0, 0, 1, 1, 0, 1, 1]

    model = HMM(pi, A, B)
    print forward(model, obs)
    print forward_backward(model, obs)
    print viterbi(model, obs)
