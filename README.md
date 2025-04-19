# SortedListTest
Testing the performance of different SortedList implementations on CPython and PyPy

CPU: Intel(R) Core(TM) i5-10500H CPU @ 2.496GHz

RAM: 8 GB DDR4 2933 MHz

## CPython 3.12.6
```
--- Bucket SortedList ---
Add   : 5.4913 sec
Find  : 6.4065 sec (found 1000000)
Remove: 6.7530 sec (removed 1000000)
Final size: 0

--- Fenwick SortedList ---
Add   : 2.1120 sec
Find  : 2.6750 sec (found 1000000)
Remove: 4.2541 sec (removed 1000000)
Final size: 0

--- sortedcontainers SortedList ---
Add   : 1.6163 sec
Find  : 1.1453 sec (found 1000000)
Remove: 2.0342 sec (removed 1000000)
Final size: 0
```

## PyPy 7.3.17 (based on Python 3.10.14)
```
--- Bucket SortedList ---
Add   : 0.8771 sec
Find  : 0.5591 sec (found 1000000)
Remove: 0.7941 sec (removed 1000000)
Final size: 0

--- Fenwick SortedList ---
Add   : 0.6002 sec
Find  : 0.7668 sec (found 1000000)
Remove: 0.8138 sec (removed 1000000)
Final size: 0

--- sortedcontainers SortedList ---
Add   : 0.5811 sec
Find  : 0.3820 sec (found 1000000)
Remove: 0.4963 sec (removed 1000000)
Final size: 0
```
