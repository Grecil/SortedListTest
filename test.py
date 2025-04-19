import math
import random
import time
from sortedcontainers import SortedList
from typing import Generic, Iterable, Iterator, List, Tuple, TypeVar, Optional
from bisect import bisect_left, bisect_right

T = TypeVar("T")


class BucketSortedList(Generic[T]):
    BUCKET_RATIO = 16
    SPLIT_RATIO = 24

    def __init__(self, a: Iterable[T] = []) -> None:
        a = list(a)
        n = self.size = len(a)
        if any(a[i] > a[i + 1] for i in range(n - 1)):
            a.sort()
        num_bucket = int(math.ceil(math.sqrt(n / self.BUCKET_RATIO)))
        self.a = [
            a[n * i // num_bucket : n * (i + 1) // num_bucket]
            for i in range(num_bucket)
        ]

    def __iter__(self) -> Iterator[T]:
        for i in self.a:
            for j in i:
                yield j

    def __reversed__(self) -> Iterator[T]:
        for i in reversed(self.a):
            for j in reversed(i):
                yield j

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "SortedList" + str(self.a)

    def __str__(self) -> str:
        s = str(list(self))
        return "{" + s[1 : len(s) - 1] + "}"

    def _position(self, x: T) -> Tuple[List[T], int, int]:
        for i, a in enumerate(self.a):
            if x <= a[-1]:
                break
        return (a, i, bisect_left(a, x))

    def __contains__(self, x: T) -> bool:
        if self.size == 0:
            return False
        a, _, i = self._position(x)
        return i != len(a) and a[i] == x

    def count(self, x: T) -> int:
        return self.index_right(x) - self.index(x)

    def add(self, x: T) -> None:
        if self.size == 0:
            self.a = [[x]]
            self.size = 1
            return
        a, b, i = self._position(x)
        a.insert(i, x)
        self.size += 1
        if len(a) > len(self.a) * self.SPLIT_RATIO:
            mid = len(a) >> 1
            self.a[b : b + 1] = [a[:mid], a[mid:]]

    def _pop(self, a: List[T], b: int, i: int) -> T:
        ans = a.pop(i)
        self.size -= 1
        if not a:
            del self.a[b]
        return ans

    def remove(self, x: T) -> bool:
        if self.size == 0:
            return False
        a, b, i = self._position(x)
        if i == len(a) or a[i] != x:
            return False
        self._pop(a, b, i)
        return True

    def lt(self, x: T) -> Optional[T]:
        for a in reversed(self.a):
            if a[0] < x:
                return a[bisect_left(a, x) - 1]

    def le(self, x: T) -> Optional[T]:
        for a in reversed(self.a):
            if a[0] <= x:
                return a[bisect_right(a, x) - 1]

    def gt(self, x: T) -> Optional[T]:
        for a in self.a:
            if a[-1] > x:
                return a[bisect_right(a, x)]

    def ge(self, x: T) -> Optional[T]:
        for a in self.a:
            if a[-1] >= x:
                return a[bisect_left(a, x)]

    def __getitem__(self, i: int) -> T:
        if i < 0:
            for a in reversed(self.a):
                i += len(a)
                if i >= 0:
                    return a[i]
        else:
            for a in self.a:
                if i < len(a):
                    return a[i]
                i -= len(a)
        raise IndexError

    def pop(self, i: int = -1) -> T:
        if i < 0:
            for b, a in enumerate(reversed(self.a)):
                i += len(a)
                if i >= 0:
                    return self._pop(a, ~b, i)
        else:
            for b, a in enumerate(self.a):
                if i < len(a):
                    return self._pop(a, b, i)
                i -= len(a)
        raise IndexError

    def index(self, x: T) -> int:
        ans = 0
        for a in self.a:
            if a[-1] >= x:
                return ans + bisect_left(a, x)
            ans += len(a)
        return ans

    def index_right(self, x: T) -> int:
        ans = 0
        for a in self.a:
            if a[-1] > x:
                return ans + bisect_right(a, x)
            ans += len(a)
        return ans

    def find_closest(self, k: T) -> Optional[T]:
        if self.size == 0:
            return None
        ltk = self.le(k)
        gtk = self.ge(k)
        if ltk is None:
            return gtk
        if gtk is None:
            return ltk
        if abs(k - ltk) <= abs(k - gtk):
            return ltk
        else:
            return gtk


class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]

    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1

    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k


class FenwickSortedList:
    block_size = 700

    def __init__(self, iterable=()):
        iterable = sorted(iterable)
        self.micros = [
            iterable[i : i + self.block_size - 1]
            for i in range(0, len(iterable), self.block_size - 1)
        ] or [[]]
        self.macro = [i[0] for i in self.micros[1:]]
        self.micro_size = [len(i) for i in self.micros]
        self.fenwick = FenwickTree(self.micro_size)
        self.size = len(iterable)

    def add(self, x):
        i = bisect_left(self.macro, x)
        j = bisect_right(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i : i + 1] = (
                self.micros[i][: self.block_size >> 1],
                self.micros[i][self.block_size >> 1 :],
            )
            self.micro_size[i : i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def lower_bound(self, x):
        i = bisect_left(self.macro, x)
        return self.fenwick(i) + bisect_left(self.micros[i], x)

    def upper_bound(self, x):
        i = bisect_right(self.macro, x)
        return self.fenwick(i) + bisect_right(self.micros[i], x)

    def remove(self, x):
        self.pop(self.lower_bound(x))

    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))


def benchmark(cls, data, name):
    sl = cls()
    print(f"\n--- {name} SortedList ---")
    # Add
    start = time.perf_counter()
    for x in data:
        sl.add(x)
    d_add = time.perf_counter() - start
    print(f"Add   : {d_add:.4f} sec")
    # Find
    start = time.perf_counter()
    found = sum(1 for x in data if x in sl)
    d_find = time.perf_counter() - start
    print(f"Find  : {d_find:.4f} sec (found {found})")
    # Remove
    start = time.perf_counter()
    removed = 0
    for x in data:
        try:
            sl.remove(x)
            removed += 1
        except ValueError:
            pass
    d_rem = time.perf_counter() - start
    print(f"Remove: {d_rem:.4f} sec (removed {removed})")
    print(f"Final size: {len(sl)}")
    return d_add, d_find, d_rem


if __name__ == "__main__":
    N = 10**6
    data = list(range(N))
    random.shuffle(data)

    for cls, name in [
        (BucketSortedList, "Bucket"),
        (FenwickSortedList, "Fenwick"),
        (SortedList, "sortedcontainers"),
    ]:
        benchmark(cls, data, name)
