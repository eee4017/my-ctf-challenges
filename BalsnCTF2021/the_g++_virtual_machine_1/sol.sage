
exps = [398527879, 293888053, 470714879, 379903019, 412387981, 461497417]
mods = [521206709, 280983943, 481821731, 446513681, 295950349, 519344851]
outs = [316196015, 183449189, 325026406, 93125040, 247649200, 358564396]
sp = 6
t = 26

def sqrt(ans, exp, mod):
		F = Zmod(mod)
		return F(ans).nth_root(exp)

ss = []
for i, o in enumerate(outs):
		ans = (sqrt(o, exps[i], mods[i]))
		ss.append(ans)


def rev(ans):
    mod = 488148229
    ret = []
    curr = ans
    for i in range(sp):
        the = int(curr) % t
        ret.append(the)
        curr = int(curr - the) % mod
        curr = int(curr * inverse_mod(t, mod)) % mod

    return (''.join([ chr(c + ord('a')) for c in ret ]))


print(''.join([rev(a) for a in ss]))

