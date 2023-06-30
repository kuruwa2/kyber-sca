#!/usr/bin/env python3

import collections
import bisect
import binascii
import ctypes
import json
import multiprocessing
import random
import sys
import string
import numpy as np
from collections import namedtuple
from tqdm import tqdm

from scipy import stats

N = 256
K = 2
Q = 3329
MIN_PCC_SCORE = 0.6
NUM_A1_CANDIDATES = 3
db = False
Candidate = namedtuple('Candidate', ('score', 'value', 'timestamp'))
start_coeff = 0
end_coeff = 256
window_start = 0
window_size = 20
trace_num = 100
num_proc = 2

zetas = [
    -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
        -171,   622,  1577,   182,   962, -1202, -1474,  1468,
        573, -1325,   264,   383,  -829,  1458, -1602,  -130,
        -681,  1017,   732,   608, -1542,   411,  -205, -1571,
        1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
        516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
        -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
        -398,   961, -1508,  -725,   448, -1065,   677, -1275,
        -1103,   430,   555,   843, -1251,   871,  1550,   105,
        422,   587,   177,  -235,  -291,  -460,  1574,  1653,
        -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
        -1590,   644,  -872,   349,   418,   329,  -156,   -75,
        817,  1097,   603,   610,  1322, -1285, -1465,   384,
        -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
        -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
        -108,  -308,   996,   991,   958, -1460,  1522,  1628
]

# Hamming Weight Table.
HW = {}
for i in range(-65536, 65537):
  HW[i] = bin(ctypes.c_uint16(i).value).count('1')
  
def poly_frombytes(a):
    r = [0] * 256
    for i in range(128):
        r[2 * i]   = ((a[3 * i + 0] >> 0) | (a[3 * i + 1] << 8)) & 0xFFF;
        r[2*i+1] = ((a[3 * i + 1] >> 4) | (a[3 * i + 2] << 4)) & 0xFFF;
    return r

def unpack_sk(a):
    r = [[0] * 256 for _ in range(2)]
    for i in range(2):
        r[i] = poly_frombytes(a[384*i:384+384])
    return r

def tQ(a):
    if a < 0:
        a += 2**32
    t = (a & (2**16-1)) * (2**16 - 3327) % 2**16
    if t & 2**15:
        t += 2**32 - 2**16
    tQ = t * 3329 & (2**32 - 1)
    return tQ

def montgomery_reduce(a):
    if a < 0:
        a += 2**32
    t = (a & (2**16-1)) * (2**16 - 3327) % 2**16
    if t & 2**15:
        t += 2**32 - 2**16
    tQ = t * 3329 & (2**32 - 1)
    t = (a - tQ) & (2**32 - 1)
    t = t >> 16
    if t & 2**15:
        t -= 2**16
    return t

def fqmul(a, b):
    return montgomery_reduce(a*b)

def LeakModelForA1(a1_guess, a1_coeff, bs):
  # SK is a vector of two polynomials, each with 256 coefficients.
  vec_idx, a1_coeff = a1_coeff // N, a1_coeff % N
  assert (a1_coeff % 2 == 1)
  b1_idx = a1_coeff
  hw = np.zeros(len(bs), dtype=np.float64)
  for i, b in enumerate(bs):
    r0 = fqmul(a1_guess, b[vec_idx][b1_idx])
    hw[i] = np.float64(HW[r0])
  return hw

def LeakModelForA0(a0_guess, a0_coeff, a1, bs):
  # SK is a vector of two polynomials, each with 256 coefficients.
  vec_idx, a0_coeff = a0_coeff // N, a0_coeff % N
  assert (a0_coeff % 2 == 0)
  b0_coeff = a0_coeff
  b1_coeff = a0_coeff + 1
  hw = np.zeros(len(bs), dtype=np.float64)
  for i, b in enumerate(bs):
    r1 = fqmul(a0_guess, b[vec_idx][b1_coeff])
    r1 += fqmul(a1, b[vec_idx][b0_coeff])
    hw[i] = np.float64(HW[r1])
  return hw

def poly_decompress(a):
    r = [0] * 256
    for i in range(128):
        r[2*i] =(((a[i] & 15) * 3329) + 8) >> 4;
        r[2*i+1] = (((a[i] >> 4) * 3329) + 8) >> 4;
    return r

def polyvec_decompress(a):
    r = [[0] * 256 for _ in range(2)]
    t = [0] * 4
    for i in range(2):
        for j in range(64):
            t[0] = (a[0] >> 0) | (a[1] << 8)
            t[1] = (a[1] >> 2) | (a[2] << 6)
            t[2] = (a[2] >> 4) | (a[3] << 4)
            t[3] = (a[3] >> 6) | (a[4] << 2)
            a = a[5:]
            for k in range(4):
                r[i][j*4+k] = ((t[k] & 0x3FF) * 3329 + 512) >> 10;
    return r

def poly_ntt(a):
    k = 1
    l = 128
    while l >= 2:
        start = 0
        while start < 256:
            zeta = zetas[k]
            k += 1
            for j in range(start, start+l):
                t = fqmul(zeta, a[j+l])
                a[j+l] = a[j] - t
                a[j] = a[j] + t
            start += 2*l
        l >>= 1

def poly_reduce(a):
    for i in range(256):
        v = ((1 << 26) + 1664) // 3329
        t = (v * a[i] + (1 << 25)) >> 26
        t *= 3329
        a[i] = a[i] - t

def polyvec_ntt(a):
    for i in range(2):
        poly_ntt(a[i])
        
# Decompress ciphertext vector and transform it to NTT domain.
def UnpackCiphertext(packed):
  '''ct = kyber.ffi.new('uint8_t [{}]'.format(len(packed)))
  kyber.ffi.buffer(ct)[:] = bytes(packed)
  b = kyber.ffi.new('polyvec*')
  kyber.lib.PQCLEAN_KYBER51290S_CLEAN_polyvec_decompress(b, ct)
  kyber.lib.PQCLEAN_KYBER51290S_CLEAN_polyvec_ntt(b)'''
  b = polyvec_decompress(packed[:640])
  polyvec_ntt(b)
  return b


# Maintain a sorted list of items with a maximum length.
# https://stackoverflow.com/questions/30443150
def InsertQueue(h, item):
  if len(h) < h.maxlen or item < h[-1]:
    if len(h) == h.maxlen:
      h.pop()
    bisect.insort_left(h, item)


class Cracker(object):

  def __init__(self, capture, trace_num):
    self.capture = capture
    self.traces = random.sample(self.capture['sessions'], trace_num)

    print('Unpacking {0} ciphertexts'.format(trace_num))
    self.bs = [UnpackCiphertext(t['ct']) for t in self.traces]

    # Collects all samples in a single m (#traces) by n (#samples) matrix.
    #  _         _
    # | -- T1  -- |
    # | -- T2  -- |
    # | -- ..  -- |
    # | -- TM  -- |
    # |_         _|
    #
    self.T = np.vstack(
        [np.array(t['pm'], dtype=np.float64) for t in self.traces])
    ntraces, nsamples = self.T.shape
    assert (ntraces == len(self.traces))

  # Correlation Power Analysis.
  def cpa(self, sk_coeff, window_start, window_size):
    # Guest two coefficients at a time. Input should be an even index.
    wsr = window_start
    assert (sk_coeff % 2 == 0)
    ntraces, nsamples = self.T.shape
    best_a1 = collections.deque(maxlen=NUM_A1_CANDIDATES)

    window_end = window_start + window_size
    with tqdm(
        desc='Guessing SK[{}]'.format(sk_coeff + 1),
        total=Q,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}| {postfix}',
        postfix=None, disable = db) as t:
      for key in range(1, Q):
        if key == 1044 or key == 2285:
            continue
        X = LeakModelForA1(key, sk_coeff + 1, self.bs)
        for i in range(window_start, window_end):
          Y = self.T[:, i]
          assert (X.shape == Y.shape)
          pcc = stats.pearsonr(X, Y)[0]
          if np.isnan(pcc):
            continue
          # Best guess is the key with the highest correlation between all possible keys,
          # across all possible time-slices.
          pcc = np.abs(pcc)
          InsertQueue(best_a1, Candidate(-pcc, key, i))
          t.postfix = {'Window': (window_start, window_end), 'A1': best_a1[0]}
        t.update()

    if not db:
      print('Result: best guess for sk coeff {0} is {1}'.format(sk_coeff + 1, best_a1))

    best_a0 = collections.deque(maxlen=1)
    used = set()
    for a1 in best_a1:
      window_start = a1.timestamp
      window_end = window_start + window_size
      if a1 in used:
          continue
      used.add(a1)
      if a1.value == 2285 or a1.value == 1044:
          continue
      with tqdm(
          desc='Guessing SK[{}]'.format(sk_coeff),
          total=Q,
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}| {postfix}',
          postfix=None, disable = db) as t:
        for key in range(1, Q):
          # Generate hypothetical.
          X = LeakModelForA0(key, sk_coeff, a1.value, self.bs)
          for i in range(window_start, window_end):
            Y = self.T[:, i]
            assert (X.shape == Y.shape)
            pcc = stats.pearsonr(X, Y)[0]
            if np.isnan(pcc):
              continue
            pcc = np.abs(pcc)
            InsertQueue(best_a0, Candidate(-pcc, (key, a1.value), i))
            t.postfix = {
                'Window': (window_start, window_end),
                'A1': a1.value,
                'A0': best_a0[0]
            }
          t.update()
          
    return (best_a0[0])

  def min_pcc(self, window_start, window_size, sk):
    ntraces, nsamples = self.T.shape
    res = 1
    ress = []
    for i in range(0, N * K, 2):
      vec_idx, sk_coeff = i // N, i % N

      best_a0 = collections.deque(maxlen=1)
      a0, a1 = sk[vec_idx][sk_coeff], sk[vec_idx][sk_coeff + 1]
      window_end = window_start + window_size
      X = LeakModelForA0(a0, i, a1, self.bs)
      for i in range(window_start, window_end):
        Y = self.T[:, i]
        assert (X.shape == Y.shape)
        pcc = stats.pearsonr(X, Y)[0]
        if np.isnan(pcc):
            continue
        pcc = np.abs(pcc)
        InsertQueue(best_a0, Candidate(-pcc, i, i))
      best_t = best_a0[0][1]
      best_pcc = -best_a0[0][0]
      print(best_pcc, best_t)
      ress.append(best_pcc)
      if best_pcc < res:
        res = best_pcc
      window_start = best_t
    return res, ress

def CPA(cracker, start_coeff, end_coeff, window_start, sk):
    cr = 0
    fp = 0
    rj = 0
    wr = 0
    for i in range(start_coeff, end_coeff, 2):
        vec_idx, sk_coeff = i // N, i % N
        key = (cracker.cpa(i, window_start, window_size))
        a0, a1 = key.value
        score = abs(key.score)
        print('Found correct {0}\'th coefficients ({1}, {2}) at time {3}'.format(i, a0, a1, key.timestamp))
        if a0 == sk[vec_idx][sk_coeff] and a1 == sk[vec_idx][sk_coeff + 1]:
            print('correct!')
            cr += 1
            if score < MIN_PCC_SCORE:
                print('rejected!')
                rj += 1
        else:
            wr += 1
            print('wrong!')
            if score >= MIN_PCC_SCORE:
                print('false positive!')
                fp += 1
        window_start = key.timestamp

        print(cr, fp, rj, wr)

if __name__ == '__main__':
    capture = json.loads(open("traces1.json", 'r').read())
    sk = unpack_sk(capture['sk'])
    cracker = Cracker(capture, trace_num)
    starts = [0, 31, 56, 82, 107, 134, 159, 186, 213, 240, 264, 292, 316, 343, 368, 395, 420, 447, 474, 501, 526, 553, 577, 604, 629, 656, 681, 708, 735, 762, 787, 814, 838, 865, 890, 917, 942, 969, 996, 1023, 1048, 1075, 1099, 1126, 1151, 1178, 1203, 1230, 1257, 1284, 1309, 1336, 1360, 1387, 1412, 1439, 1464, 1491, 1518, 1545, 1570, 1597, 1621, 1648, 1673, 1700, 1725, 1752, 1779, 1806, 1831, 1858, 1882, 1909, 1934, 1961, 1986, 2013, 2040, 2067, 2091, 2119, 2143, 2170, 2195, 2222, 2247, 2274, 2301, 2328, 2358, 2380, 2404, 2431, 2456, 2483, 2508, 2535, 2562, 2589, 2614, 2641, 2665, 2692, 2717, 2744, 2769, 2796, 2823, 2850, 2875, 2902, 2927, 2953, 2978, 3005, 3030, 3057, 3084, 3111, 3136, 3163, 3187, 3215, 3239, 3266, 3291, 3318, 3345]
    jobs = []
    for i in range(num_proc):
        sc = start_coeff + (end_coeff - start_coeff) // num_proc * i
        ec = sc + (end_coeff - start_coeff) // num_proc
        ws = starts[sc//2]
        process = multiprocessing.Process(target = CPA, args = (cracker, sc, ec, ws, sk))
        jobs.append(process)

    for j in jobs:
        j.start()
        
    for j in jobs:
        j.join()
