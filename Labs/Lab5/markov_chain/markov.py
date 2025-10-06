import numpy as np

def markov(rho, A, nmax, rng = np.random.default_rng()):

    assert all(rho >= 0)
    assert np.isclose(np.sum(rho), 1.0)
    assert np.allclose(np.sum(A, axis=1), 1.0)
    assert np.all(A>=0)
    N = len(rho)
    etat = rho
    X = np.zeros(nmax)
    X[0] = rng.choice(N, p=rho)

    for n in range(1,nmax):
        etat = etat@A #note that tere is a pb of notation A or T.A
        X[n] = rng.choice(N, p=etat)
    return X