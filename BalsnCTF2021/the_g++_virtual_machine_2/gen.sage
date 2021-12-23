import numpy as np

flag = "balsn{maataprorromintovm}"
print(f"flag = {flag}, len(flag) = {len(flag)}")
L = len(flag)
F = Zmod(31)

def str2mat(flag, mod = 31):
    a = np.ones((5, 5))
    for i in range(5):
        for j in range(5):
            a[i, j] = (ord(flag[i * 5+j]) - ord('a')) % 31
    return a.astype(np.int32)


def mat2str(mat, mod = 31):
    a = ""
    for i in range(5):
        for j in range(5):
            a += chr(int(mat[i, j]) + ord('a'))
    return a

def sbox(gen):
    ret = [0]*33
    j = 0;
    for i in range(33):
        test = pow(gen, i, 31)
        if test < 25:
            ret[j] = test
            j = j + 1
        
    ret[24] = 0
    return ret[:25]


def setMatrix(the, mod = 97):
    a = np.ones((5, 5), dtype=np.int32)
    for i in range(5):
        for j in range(5):
            a[i, j] = int(pow(the, i * 5 + j, mod))%31
    return a


def permute(v, p):
    ret = [0] * 25
    v= np.array(v).reshape(-1)
    for i in range(25):
        ret[p[i]] = v[i]
    return np.array(ret).reshape((5, 5)).astype(np.int32)

def sumMatrix(v):
    return np.array(v).reshape(-1).astype(np.int32).sum()

def inverse_sbox(b):
    return [b.index(i) for i in range(25)]

def f(v1, m_seed, sbox_seed):
    v1 = Matrix(F, v1)
    v2 = Matrix(F, setMatrix(m_seed))
    v3 = (v1 * v2)
    vnew = permute(v3, sbox(sbox_seed))
    
    return vnew

def invf(v, m_seed, sbox_seed):
    v3 = permute(v, inverse_sbox(sbox(sbox_seed)))
    v2_1 = Matrix(F, setMatrix(m_seed)).inverse()
    v1 = Matrix(F, v3) * v2_1
    
    return v1

out = f(str2mat(flag), 5, 3)
out = f(out, 15, 11)
vans = f(out, 27, 24)

print(vans)

out = invf(vans, 27, 24)
out = invf(out, 15, 11)
out = invf(out, 5, 3)

print(mat2str(out))
