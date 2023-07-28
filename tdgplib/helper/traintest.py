import collections
from typing import Mapping

__all__ = ['TrainTestSplit']


class TrainTestSplit(collections.namedtuple("TrainTestSplit", ["train", "test"]), Mapping):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def sapply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(*self.train, *args, **kwargs), f(*self.test, *args, **kwargs)
        )

    def apply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(self.train, *args, **kwargs), f(self.test, *args, **kwargs)
        )

    def unzip(*args):
        return TrainTestSplit([a.train for a in args], [a.test for a in args])

    def zip(self):
        return (TrainTestSplit(*x) for x in zip(self.train, self.test))

    @staticmethod
    def from_sklearn(sklearn_split_ret):
        return [
            TrainTestSplit(sklearn_split_ret[i], sklearn_split_ret[i + 1])
            for i in range(0, len(sklearn_split_ret), 2)
        ]

    def __getitem__(self, k):
        if k == 'train':
            return self.train
        elif k == 'test':
            return self.test
        else:
            raise IndexError(f'k={k!r} must be "train" or "test"')

    def keys(self):
        return ['train', 'test']

    def items(self):
        return [('train', self.train), ('test', self.test)]

    def values(self):
        return [self.train, self.test]

    def __len__(self):
        return 2
