import random


def estimate_pi(n, seed):
    """Estimate pi with n reproducible Monte Carlo samples."""
    rng = random.Random(seed)
    inside_quarter_circle = 0

    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1:
            inside_quarter_circle += 1

    return 4 * inside_quarter_circle / n
