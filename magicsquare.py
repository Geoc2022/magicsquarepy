# import modules
import numpy as np
import math


# class which contains the operations for magic squares
class magicsquare:
    def __init__(self, name, matrix=None, patterntype=None):
        self.name = name
        if np.any(matrix):
            if matrix.shape[0] == matrix.shape[1]:
                self.matrix = matrix
                self.dim = matrix.shape[0]
        else:
            self.matrix = np.array([])
            self.dim = 0
        self.constant = 0 if self.dim == 0 else np.sum(self.dim) / self.dim
        self.patterntype = patterntype or 'Mid'

    def __repr__(self):
        matrix_repr = np.array2string(self.matrix, separator=',')
        return f"magicsquare('{self.name}', {matrix_repr}, {self.patterntype})"

    def __str__(self):
        return f"{self.name}:\n {self.matrix}"

    # numeric operations

    def __neg__(self):
        return magicsquare(f'-{self.name}', np.negative(self.matrix), self.patterntype)

    def __add__(self, other):
        if isinstance(other, magicsquare):
            return magicsquare(f'({self.name} + {other.name})', np.add(self.matrix, other.matrix), self.patterntype)
        elif isinstance(other, int) or isinstance(other, float):
            return magicsquare(f'({self.name} + {str(other)})', np.add(self.matrix, np.full_like(self.matrix, other)),
                               self.patterntype)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, magicsquare):
            return magicsquare(f'({self.name} - {other.name})', np.subtract(self.matrix, other.matrix),
                               self.patterntype)
        elif isinstance(other, int) or isinstance(other, float):
            return magicsquare(f'({self.name} - {str(other)})',
                               np.subtract(self.matrix, np.full_like(self.matrix, other)), self.patterntype)
        return NotImplemented

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            if isinstance(other, int) or isinstance(other, float):
                return other + (-self)
        return NotImplemented

    def __mul__(self, other, others=None):
        if isinstance(other, magicsquare) and (self.matrix.shape == other.matrix.shape):
            return magicsquare(f'({self.name} * {other.name})', np.matmul(self.matrix, other.matrix), self.patterntype)
        elif isinstance(other, int) or isinstance(other, float):
            return magicsquare(f'({self.name} * {str(other)})', other * self.matrix, self.patterntype)
        return NotImplemented

    def __rmul__(self, other):
        if other == 0:
            return self
        else:
            return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, magicsquare) and (self.matrix.shape == other.matrix.shape):
            return magicsquare(f'({self.name} / {other.name})', np.matmul(self.matrix, np.linalg.inv(other.matrix)),
                               self.patterntype)
        elif isinstance(other, int) or isinstance(other, float):
            return magicsquare(f'({self.name} / {str(other)})', self.matrix / other, self.patterntype)
        return NotImplemented

    def __rtruediv__(self, other):
        if other == 0:
            return self
        else:
            if isinstance(other, int) or isinstance(other, float):
                return magicsquare(f'({str(other)} / {self.name})', other / self.matrix, self.patterntype)
        return NotImplemented

    def __invert__(self):
        if np.linalg.det(self.matrix) != 0:
            return magicsquare(f'(~{self.name})', np.linalg.inv(self.matrix), self.patterntype)
        else:
            return None

    def __rshift__(self, other):
        if isinstance(other, int):
            return magicsquare(f'({self.name} >> {str(other)})', np.roll(self.matrix, other, axis=1), self.patterntype)
        return NotImplemented

    def __rrshift__(self, other):
        if other == 0:
            return self
        else:
            return self.__rshift__(other)

    def __lshift__(self, other):
        if isinstance(other, int):
            return magicsquare(f'({self.name} << {str(other)})', np.roll(self.matrix, other, axis=0), self.patterntype)
        return NotImplemented

    def __rlshift__(self, other):
        if other == 0:
            return self
        else:
            return self.__lshift__(other)

    # nonnumeric operations

    def list_square(self, input_list):
        if isinstance(input_list, list):
            if (math.sqrt(len(input_list))).is_integer():
                self.dim = int(math.sqrt(len(input_list)))
                self.matrix = (np.array(input_list)).reshape((self.dim, self.dim))
            return self.matrix

    def patternize(self, patterntype=None):
        self.patterntype = patterntype or 'med'
        if (self.patterntype == 'med') or (self.patterntype == 'median'):
            return np.subtract(self.matrix, np.full((self.dim, self.dim), int(np.median(self.matrix.tolist()))))
        else:
            return np.subtract(self.matrix, np.full((self.dim, self.dim), int(np.amin(self.matrix.tolist()))))

    def calc_dim(self):
        if self.matrix.shape[0] == self.matrix.shape[1]:
            self.dim = self.matrix.shape[0]
            return self.dim
        return None

    def calc_constant(self):
        if self.calc_dim():
            self.constant = np.sum(self.matrix) / self.dim
            return self.constant
        return None

    def check_ms(self):
        self.calc_constant()
        if (np.sum(self.matrix, axis=0) == np.full((1, self.dim), self.constant)).all() and (
                np.full((1, self.dim), self.constant) == np.sum(self.matrix, axis=1).T).all():
            if np.trace(self.matrix) == np.trace(np.flip(self.matrix, axis=1)) == self.constant:
                return True
        return False

    def generate_odd_square(self, gen_dim):
        if (gen_dim % 2) == 0:
            odd_dim = gen_dim + 1
        else:
            odd_dim = gen_dim
        odd_magic_square = np.zeros((odd_dim, odd_dim))
        half_dim = odd_dim / 2
        sub_dim = odd_dim - 1
        num = 1
        while num <= (odd_dim * odd_dim):
            if half_dim == -1 and sub_dim == odd_dim:
                sub_dim = odd_dim - 2
                half_dim = 0
            else:
                if sub_dim == odd_dim:
                    sub_dim = 0
                if half_dim < 0:
                    half_dim = odd_dim - 1
            if odd_magic_square[int(half_dim)][int(sub_dim)]:
                sub_dim = sub_dim - 2
                half_dim = half_dim + 1
                continue
            else:
                odd_magic_square[int(half_dim)][int(sub_dim)] = num
                num += 1
            sub_dim = sub_dim + 1
            half_dim = half_dim - 1
        self.matrix = odd_magic_square
        if self.check_ms():
            self.calc_constant()
            return self.matrix
        self.matrix = np.array([])
        return None


# list of example magic squares from Wikipedia
ms_list = [[8, 1, 6, 3, 5, 7, 4, 9, 2],
           [16, 14, 7, 30, 23, 24, 17, 10, 8, 31, 32, 25, 18, 11, 4, 5, 28, 26, 19, 12, 13, 6, 29, 22, 20],
           [1, 35, 4, 33, 32, 6, 25, 11, 9, 28, 8, 30, 24, 14, 18, 16, 17, 22, 13, 23, 19, 21, 20, 15, 12, 26, 27, 10,
            29, 7, 36, 2, 34, 3, 5, 31],
           [35, 26, 17, 1, 62, 53, 44, 46, 37, 21, 12, 3, 64, 55, 57, 41, 32, 23, 14, 5, 66, 61, 52, 43, 34, 25, 16, 7,
            2, 63, 54, 45, 36, 27, 11, 13, 4, 65, 56, 47, 31, 22, 24, 15, 6, 67, 51, 42, 33],
           [60, 53, 44, 37, 4, 13, 20, 29, 3, 14, 19, 30, 59, 54, 43, 38, 58, 55, 42, 39, 2, 15, 18, 31, 1, 16, 17, 32,
            57, 56, 41, 40, 61, 52, 45, 36, 5, 12, 21, 28, 6, 11, 22, 27, 62, 51, 46, 35, 63, 50, 47, 34, 7, 10, 23, 26,
            8, 9, 24, 25, 64, 49, 48, 33], [2, 7, 6, 9, 5, 1, 4, 3, 8],
           [12, 3, 13, 6, 2, 7, 9, 16, 15, 10, 8, 1, 5, 4, 14, 11]]

# checking the magic squares in the the example list
for ms in ms_list:
    testing_ms = magicsquare('ms')
    testing_ms.list_square(ms)
    print(str(testing_ms))
    print(testing_ms.check_ms())

# checking how rolls effect whether a square is magic or not
testing_list = []
testing_num = 5
testing_ms = magicsquare('ms')
testing_ms.list_square(ms_list[6])
for i in range(testing_num):
    for j in range(testing_num):
        ms = testing_ms
        ms = ms << i
        ms = ms >> j
        print(ms)
        testing_list.append('██' if ms.check_ms() is True else '  ')
        print(ms.check_ms())
print('\nrolls rightward ( x axis → )\nrolls downward ( y axis ↓ )')
for i in range(testing_num):
    print(testing_list[testing_num * i: testing_num * (i + 1)])
print('██ means altered magic square is a magic square')

# generating some odd magic squares
ms = magicsquare('ms')
for i in range(3, 7, 2):
    print(f'{i}:\n {ms.generate_odd_square(i)}')
