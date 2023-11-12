import numpy as np
from numpy.random import uniform


class HyperParameters:
    def __init__(self,
                 num_params: int,
                 limits: list[tuple[float, float]] = None,
                 parameters: np.ndarray[float] = None):
        if not isinstance(num_params, int):
            raise TypeError('num_params must be an integer.')
        if num_params <= 0:
            raise ValueError('num_params must be greater than 0.')
        if limits is not None:
            if len(limits) != num_params:
                raise ValueError('list of limits should be of length "num_params". ')
        if parameters is not None:
            if len(parameters) != num_params:
                raise ValueError('When explicitly provided, the number of parameters should be equal to "num_params". ')
        self.num_params = num_params
        self.limits = limits
        self.parameters = parameters

    def get_parameters(self):
        if self.parameters is None:
            raise RuntimeError(f' "parameters" attribute has not been set. ')
        return self.parameters

    def set_parameters(self, parameters: np.ndarray[float]):
        if len(parameters) != self.num_params:
            raise ValueError('When explicitly provided, the number of parameters should be equal to "num_params". ')
        self.parameters = parameters

    def set_random_parameters(self):
        self.parameters = self.get_random_parameters()

    def get_random_parameters(self, distribution: str = 'uniform'):
        if distribution == 'uniform':
            if self.limits is not None:
                self.parameters = np.array([uniform(low=limit[0], high=limit[1]) for limit in self.limits])
            else:
                self.parameters = np.array([uniform(low=-np.inf, high=np.inf) for _ in range(self.num_params)])
            return self.get_parameters()