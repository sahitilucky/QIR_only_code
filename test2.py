
from fractions import Fraction, gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def common_integer(numbers):
    fractions = [Fraction(n).limit_denominator() for n in numbers]
    multiple  = reduce(lcm, [f.denominator for f in fractions])
    ints      = [f * multiple for f in fractions]
    divisor   = reduce(gcd, ints)
    return [int(n / divisor) for n in ints]
n2 = [1.4, 1.7]
print common_integer(n2)






