from math import sqrt, ceil

def is_prime(n):
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for d in range(5, ceil(sqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True

def get_non_1_prime_factors(n):
    factor_list = []

    if n >= 2:
        for d in [2] + list(range(3, ceil(sqrt(n))+2, 2)):
            while n % d == 0:
                factor_list.append(d)
                n = n // d

    if n != 1:
        factor_list.append(n)

    return factor_list

