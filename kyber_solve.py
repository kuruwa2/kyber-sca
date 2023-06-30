from sage.all import *
from fpylll import BKZ
from fpylll.fplll.integer_matrix import IntegerMatrix
import random
import time

q = 3329
g = GF(q)(17)
ans = vector([0, -2, 1, -1, 2, 1, 1, 0, -1, 0, 1, -1, 1, 1, 1, 0, 1, -1, 0, -2, 0, 1, 2, 1, 1, -1, 0, 0, 1, -1, 1, 0, 0, 2, 1, -1, 2, 1, 2, -1, 1, 1, -2, 0, 1, 1, 1, -1, 1, 1, 0, -1, 2, -1, -2, -2, 0, -1, 1, 1, 0, 1, -2, 0, -1, 0, -1, 1, 0, 1, 0, 1, 0, -1, -1, 1, 1, 1, 1, 0, 1, -2, -1, 0, 0, 0, 0, -1, -1, 1, 1, -1, 0, -1, 0, 0, 1, 0, 1, 0, -1, 1, 1, -1, 1, -1, -1, 1, -1, 0, 0, 2, -1, 0, -1, -1, 1, 2, -1, 0, -1, -1, 1, -2, -1, 1, -1, 0, 0, 0, -2, 0, -1, 1, -1, -1, -1, 0, 1, 0, 0, -1, -1, 2, 1, 0, -1, -1, 0, -2, 0, 1, -1, 0, 0, -1, 0, 1, 1, 0, 1, -1, -2, -1, -2, 0, 0, -1, -2, -1, 0, -1, 0, 2, -2, -1, 0, 0, 0, -1, 0, -1, -1, 2, 1, 0, -1, 1, 0, 0, 0, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, -1, -2, 0, 1, -1, 1, -1, -1, -2, 2, -1, 1, -1, -1, -1, 1, 0, -1, -1, -2, 0, 1, 2, 0, -2, -1, 1, 0, 2, 2, -1, 0, 1, -1, 0, 0, 0, -1, -1, -1, 0, 2, 0, 1, 0, -1, 0, 0, 1, 1, 2, -1, 1, 0, -1])

def bit_reverse(n):
    return int('{:08b}'.format(n)[::-1], 2)

def gen_M(): # Matrix for NTT
    M = []
    for i in range(128):
        gg = g**(bit_reverse(i) + 1)
        m = [int(gg**i) for i in range(128)]
        M.append(m)
    return matrix(GF(q), M)

M = gen_M()
MI = M.inverse().T
test = [1170, 2527, 564, 126, 2080, 3106, 1114, 1010, 825, 2634, 1267, 823, 344, 3108, 2320, 1608, 3241, 3136, 843, 549, 2707, 1566, 2373, 1406, 1329, 820, 380, 766, 1780, 2625, 1398, 1512, 2760, 1749, 1265, 2383, 3269, 364, 1155, 2881, 2050, 278, 219, 3251, 2934, 842, 837, 30, 2133, 328, 3003, 2262, 1285, 155, 827, 942, 847, 1910, 349, 1830, 1901, 660, 1262, 2684, 2194, 469, 1908, 3015, 2731, 1782, 1264, 1443, 2723, 3065, 2416, 2960, 1151, 2305, 603, 1958, 3115, 1470, 1232, 1131, 2601, 2921, 1866, 76, 3309, 1439, 875, 2392, 92, 1880, 2686, 22, 2152, 1347, 2071, 3314, 1935, 753, 2371, 163, 3284, 2997, 2279, 531, 2877, 389, 652, 28, 1724, 804, 1012, 273, 2295, 1024, 1254, 2551, 548, 1005, 86, 2013, 1438, 1494, 3282, 2857, 593, 3327, 3004, 2989, 1999, 1827, 2331, 2859, 601, 1870, 2944, 2176, 2299, 1038, 73, 268, 2517, 1338, 1787, 2171, 1339, 880, 2232, 2896, 2192, 2197, 3244, 857, 1007, 2616, 528, 1468, 2857, 1509, 336, 1828, 1445, 984, 2091, 131, 480, 13, 778, 1715, 1934, 2527, 805, 1043, 2985, 1202, 1698, 2639, 304, 3228, 2852, 2912, 904, 1337, 833, 1617, 349, 649, 3204, 2418, 2109, 3272, 1361, 1810, 988, 590, 355, 607, 2020, 1592, 2795, 289, 1283, 359, 921, 2337, 1724, 1049, 2260, 1817, 2323, 1516, 1473, 2378, 1908, 1980, 430, 149, 2292, 836, 1569, 1484, 207, 2042, 531, 1748, 873, 649, 2725, 3268, 2487, 679, 3012, 2606, 827, 14, 1606, 1768, 2237, 1941, 2115, 1714, 37, 2990, 1920, 173, 3246, 350, 1440, 1194, 1395, 1988, 185, 1353]

flags = BKZ.GH_BND|BKZ.AUTO_ABORT#|BKZ.VERBOSE#|BKZ.MAX_LOOPS
par = BKZ.Param(50, strategies = BKZ.DEFAULT_STRATEGY, max_loops = 0, flags = flags)

def timing_test(zeros):
    print('[o]', zeros)
    zero_index = random.sample(list(range(128)), zeros)

    v = vector(GF(q), test)
    for i in zero_index:
        v[2*i] = 0
        v[2*i+1] = 0

    c1 = v[0::2] * MI
    c2 = v[1::2] * MI

    a1 = []
    a2 = []
    print('[x] building matrix...')
    for i in zero_index:
        a1.append(MI[i])
        a2.append(MI[i])

    U = []
    while len(U) != 128:
        ind = random.sample(list(range(128)), 2)
        u = [0] * 128
        u[ind[0]] = 1
        u[ind[1]] = 1
        U.append(u)
        if matrix(GF(q), U).rank() != len(U):
            U = U[:-1]

    while len(U) != zeros*2:
        ind = random.sample(list(range(128)), 2)
        u = [0] * 128
        u[ind[0]] = 1
        u[ind[1]] = 1
        if u not in U:
            U.append(u)

    U = matrix(GF(q), U)
    a1 = (U * Matrix(GF(q), a1).T).T.echelon_form().change_ring(ZZ)
    a2 = (U * Matrix(GF(q), a2).T).T.echelon_form().change_ring(ZZ)

    b1 = []
    b2 = []
    for i in range(zeros):
        b1.append(list(a1[i]) + [0])
        b2.append(list(a2[i]) + [0])

    for i in range(zeros, 2*zeros):
        b1.append([0] * i + [q] + [0] * (zeros*2-i))
        b2.append([0] * i + [q] + [0] * (zeros*2-i))

    b1.append(list((U*c1).change_ring(ZZ)) + [1])
    b2.append(list((U*c2).change_ring(ZZ)) + [1])
    
    #B1 = matrix(ZZ, b1)
    #B2 = matrix(ZZ, b2)
    B1 = IntegerMatrix.from_matrix(b1)
    B2 = IntegerMatrix.from_matrix(b2)

    print('[x] LLL...')
    start = time.time()
    #test1 = B1.BKZ(block_size=20)
    #test2 = B2.BKZ(block_size=20)
    BKZ.reduction(B1, par)
    BKZ.reduction(B2, par)
    last = time.time() - start
    
    B1.to_matrix(b1)
    B2.to_matrix(b2)

    U = U.change_ring(ZZ)
    
    test1 = U.solve_right(vector(b1[0][:-1]))
    test2 = U.solve_right(vector(b2[0][:-1]))
    print(test1, test2)
    if (ans[::2] == test1 or -ans[::2] == test1) and (ans[1::2] == test2 or -ans[1::2] == test2):
        print('[!] success! in', last, 'sec')
        return last

    return 0

times = []
#for i in [10,20,30,40,50,60,70,80,85,86,87,88,89,90,91,92]:
for i in range(85, 91):
    avg = 0
    sus = 0
    for _ in range(10):
        t = timing_test(i)
        if t != 0:
            sus += 1
            avg += t
    if sus != 0:
        times.append(avg / sus)
    print(i, sus)

    print(times)

#85~91
#[10, 10, 9, 9, 7]
#[425.67085916996, 367.52429430484773, 393.1317620012495, 456.2010177241431, 476.9453225476401]
