import json
from scipy.signal import decimate

data = bytes.fromhex(''.join(open("./ELMO/data.txt", 'r').read().split()))

#sk = data[:1632]
#data = data[2432:]
sk = data[:3168]
data = data[4736:]

traces = {}
traces['sk'] = list(sk)

sess = []
for i in range(800):
    ses = {}
    ses['ct'] = list(data[:768])
    data = data[768:]
    pm = []
    with open(f"./ELMO/output/traces/trace{str(i+1).rjust(5,'0')}.trc", 'r') as f:
        for i in f:
            pm.append(float(i))
    pm = list(decimate(pm, 5))
    ses['pm'] = pm
    sess.append(ses)

traces["sessions"] = sess
f = open("./traces1.json", 'w')
f.write(json.dumps(traces))
f.close()
