import sympy
from sympy.stats import Bernoulli as spB
from sympy import oo, S
import abc


class Sdistribution(metaclass=abc.ABCMeta):
    def __init__(self):
        self.dist = None
        return

    def sample(self):
        return sympy.stats.sample(self.dist)

    def __str__(self):
        return str(type(self))

    def __eq__(self, other):
        pass


class DiracDelta(Sdistribution):
    def __init__(self, expr=S(0)):
        super().__init__()
        self.prob = expr

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, expr):
        assert isinstance(expr, sympy.Basic) or expr == oo or expr == -oo, "prob should be a Sympy object for Sdistibution"
        self._prob = expr

    def __eq__(self, other):
        if isinstance(other, DiracDelta):
            return other.prob == self.prob
        else:
            return False

    def sample(self):
        return self._prob

    def __str__(self):
        return "DiracDelta("+str(self._prob)+")"


class Bernoulli(Sdistribution):
    def __init__(self, prob=S(0.5)):
        super().__init__()
        self.prob = prob
        self.dist = sympy.stats.Bernoulli('dist', prob)

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, expr):
        assert isinstance(expr,
                          sympy.Basic) or expr == oo or expr == -oo, "prob should be a Sympy object for Sdistibution"
        self._prob = expr

    def __eq__(self, other):
        if isinstance(other, Bernoulli):
            return other.prob == self.prob
        else:
            return False

    def __str__(self):
        return "Bernoulli("+str(self._prob)+")"


class Uniform(Sdistribution):
    def __init__(self, a=S(0), b=S(1)):
        super().__init__()
        self.low = a
        self.high = b
        self.dist = sympy.stats.Uniform("dist", a, b)

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, a):
        assert isinstance(a,
                          sympy.Basic) or a == oo or a == -oo, "bound should be a Sympy object for Sdistibution"
        self._low = a

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, b):
        assert isinstance(b,
                          sympy.Basic) or b == oo or b == -oo, "bound should be a Sympy object for Sdistibution"
        self._high = b

    def __str__(self):
        return "Uniform("+str(self._low)+","+str(self._high)+")"

    def __eq__(self, other):
        if isinstance(other, Uniform):
            return (other.low == self.low) and (other.high == self.high)
        else:
            return False


class Gaussian(Sdistribution):
    def __init__(self, mu=S(0), sigma=S(1)):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.dist = sympy.stats.Normal("dist", mu, sigma)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        assert isinstance(mu,
                          sympy.Basic) or mu == oo or mu == -oo, "mean should be a Sympy object for Sdistibution"
        self._mu = mu

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        assert isinstance(sigma,
                          sympy.Basic) or sigma == oo or sigma == -oo, "variance should be a Sympy object for Sdistibution"
        self._sigma = sigma

    def __str__(self):
        return "Normal("+str(self.mu)+","+str(self.sigma)+")"

    def __eq__(self, other):
        if isinstance(other, Gaussian):
            return (other.mu == self.mu) and (other.sigma == self.sigma)
        else:
            return False


def main():
    x, y = S('x'), S('y')
    B1 = Bernoulli(x)
    B2 = Bernoulli(y)
#     c = S(0.4)
    # B1 = Gaussian(2*c,c/10)
    # B2 = DiracDelta()
    print(y.subs(y, 0.5))
    # bval1 = B1.sample()
    # bval2 = B2.sample()
    # print(type(bval1), bval1)
    # print(type(bval2), bval2)
    print(B2)
    # print(B2==B1)



if __name__ == "__main__":
    main()