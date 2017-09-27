import argparse
import numpy as np
import h5py
import sys, re

I = complex(0,1)

local_ops = {
    "=loc:1="         : {"idx": 5, "fct": {"dn": -I, "up":  I}},
    "=loc:g0="        : {"idx": 4, "fct": {"dn":  1, "up":  1}}, 
    "=loc:g5="        : {"idx": 0, "fct": {"dn":  I, "up": -I}}, 
    "=loc:gx="        : {"idx": 1, "fct": {"dn":  1, "up":  1}}, 
    "=loc:gy="        : {"idx": 2, "fct": {"dn":  1, "up":  1}}, 
    "=loc:gz="        : {"idx": 3, "fct": {"dn":  1, "up":  1}}, 
    "=loc:g5g0="      : {"idx": 9, "fct": {"dn": -1, "up": -1}}, 
    "=loc:g5gx="      : {"idx": 6, "fct": {"dn": -1, "up": -1}}, 
    "=loc:g5gy="      : {"idx": 7, "fct": {"dn": -1, "up": -1}}, 
    "=loc:g5gz="      : {"idx": 8, "fct": {"dn": -1, "up": -1}}, 
    "=loc:g5si0x="    : {"idx":13, "fct": {"dn":  1, "up": -1}}, 
    "=loc:g5si0y="    : {"idx":14, "fct": {"dn":  1, "up": -1}}, 
    "=loc:g5si0z="    : {"idx":15, "fct": {"dn":  1, "up": -1}}, 
    "=loc:g5sixy="    : {"idx":10, "fct": {"dn":  1, "up": -1}}, 
    "=loc:g5sixz="    : {"idx":11, "fct": {"dn":  1, "up": -1}}, 
    "=loc:g5siyz="    : {"idx":12, "fct": {"dn":  1, "up": -1}}, 
}

noether_ops = {
    "=noe:g0="        : {"idx": 3, "fct": {"dn":  1, "up":  1}}, 
    "=noe:gx="        : {"idx": 0, "fct": {"dn":  1, "up":  1}}, 
    "=noe:gy="        : {"idx": 1, "fct": {"dn":  1, "up":  1}}, 
    "=noe:gz="        : {"idx": 2, "fct": {"dn":  1, "up":  1}}, 
}

deriv_ops = {
    "=der:g0D0:sym="  : {"dir": (3,3), "idx": (4,4), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gxD0:sym="  : {"dir": (3,0), "idx": (1,4), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gyD0:sym="  : {"dir": (3,1), "idx": (2,4), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gzD0:sym="  : {"dir": (3,2), "idx": (3,4), "fct": {"dn":  1, "up":  1}, "sym": 1},
    #.
    "=der:gxDx:sym="  : {"dir": (0,0), "idx": (1,1), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gyDx:sym="  : {"dir": (0,1), "idx": (2,1), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gzDx:sym="  : {"dir": (0,2), "idx": (3,1), "fct": {"dn":  1, "up":  1}, "sym": 1},
    #.
    "=der:gyDy:sym="  : {"dir": (1,1), "idx": (2,2), "fct": {"dn":  1, "up":  1}, "sym": 1},
    "=der:gzDy:sym="  : {"dir": (1,2), "idx": (3,2), "fct": {"dn":  1, "up":  1}, "sym": 1},
    #.
    "=der:gzDz:sym="  : {"dir": (2,2), "idx": (3,3), "fct": {"dn":  1, "up":  1}, "sym": 1},
    #.
    #..
    #.
    "=der:g5g0D0:sym="  : {"dir": (3,3), "idx": (9,9), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gxD0:sym="  : {"dir": (3,0), "idx": (6,9), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gyD0:sym="  : {"dir": (3,1), "idx": (7,9), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gzD0:sym="  : {"dir": (3,2), "idx": (8,9), "fct": {"dn": -1, "up": -1}, "sym": 1},
    #.
    "=der:g5gxDx:sym="  : {"dir": (0,0), "idx": (6,6), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gyDx:sym="  : {"dir": (0,1), "idx": (7,6), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gzDx:sym="  : {"dir": (0,2), "idx": (8,6), "fct": {"dn": -1, "up": -1}, "sym": 1},
    #.
    "=der:g5gyDy:sym="  : {"dir": (1,1), "idx": (7,7), "fct": {"dn": -1, "up": -1}, "sym": 1},
    "=der:g5gzDy:sym="  : {"dir": (1,2), "idx": (8,7), "fct": {"dn": -1, "up": -1}, "sym": 1},
    #.
    "=der:g5gzDz:sym="  : {"dir": (2,2), "idx": (8,8), "fct": {"dn": -1, "up": -1}, "sym": 1},
    #.
    #..
    #.
    "=der:g5gxD0:asy="  : {"dir": (3,0), "idx": (6,9), "fct": {"dn": -1, "up": -1}, "sym":-1},
    "=der:g5gyD0:asy="  : {"dir": (3,1), "idx": (7,9), "fct": {"dn": -1, "up": -1}, "sym":-1},
    "=der:g5gzD0:asy="  : {"dir": (3,2), "idx": (8,9), "fct": {"dn": -1, "up": -1}, "sym":-1},
    #.
    "=der:g5gyDx:asy="  : {"dir": (0,1), "idx": (7,6), "fct": {"dn": -1, "up": -1}, "sym":-1},
    "=der:g5gzDx:asy="  : {"dir": (0,2), "idx": (8,6), "fct": {"dn": -1, "up": -1}, "sym":-1},
    #.
    "=der:g5gzDy:asy="  : {"dir": (1,2), "idx": (8,7), "fct": {"dn": -1, "up": -1}, "sym":-1},
    #.
    #..
    #.
    "=der:g5six0D0:sym=": {"dir": (3,3), "idx": (13,13), "fct": {"dn": -1, "up":  1}, "sym": 1},
    "=der:g5siy0D0:sym=": {"dir": (3,3), "idx": (14,14), "fct": {"dn": -1, "up":  1}, "sym": 1},
    "=der:g5siz0D0:sym=": {"dir": (3,3), "idx": (15,15), "fct": {"dn": -1, "up":  1}, "sym": 1},
    #.
    "=der:g5si00Dx:sym=": {"dir": (3,3), "idx": (13,13), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5six0Dx:sym=": {"dir": (0,0), "idx": (13,13), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
    "=der:g5siy0Dx:sym=": {"dir": (0,3), "idx": (14,10), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    "=der:g5siz0Dx:sym=": {"dir": (0,3), "idx": (15,11), "fct": {"dn":  -1, "up":   1}, "sym": 1},    
    "=der:g5si0xDx:sym=": {"dir": (0,0), "idx": (13,13), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5siyxDx:sym=": {"dir": (0,0), "idx": (10,10), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    "=der:g5sizxDx:sym=": {"dir": (0,0), "idx": (11,11), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    #.
    "=der:g5si00Dy:sym=": {"dir": (3,3), "idx": (14,14), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5si0xDy:sym=": {"dir": (1,0), "idx": (13,14), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5si0yDy:sym=": {"dir": (1,1), "idx": (14,14), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5six0Dy:sym=": {"dir": (1,3), "idx": (13,10), "fct": {"dn":  -1, "up":   1}, "sym":-1},    
    "=der:g5sixxDy:sym=": {"dir": (0,0), "idx": (10,10), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5sixyDy:sym=": {"dir": (1,1), "idx": (10,10), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5siy0Dy:sym=": {"dir": (1,1), "idx": (14,14), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
    "=der:g5siyxDy:sym=": {"dir": (1,1), "idx": (10,10), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
    "=der:g5siz0Dy:sym=": {"dir": (1,3), "idx": (15,12), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    "=der:g5sizxDy:sym=": {"dir": (1,0), "idx": (11,12), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    "=der:g5sizyDy:sym=": {"dir": (1,1), "idx": (12,12), "fct": {"dn":  -1, "up":   1}, "sym": 1},
    #.
    "=der:g5si00Dz:sym=": {"dir": (3,3), "idx": (15,15), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5si0xDz:sym=": {"dir": (2,0), "idx": (13,15), "fct": {"dn":   1, "up":  -1}, "sym": 1}, 
    "=der:g5si0yDz:sym=": {"dir": (2,1), "idx": (14,15), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5si0zDz:sym=": {"dir": (2,2), "idx": (15,15), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5six0Dz:sym=": {"dir": (2,3), "idx": (13,11), "fct": {"dn":  -1, "up":   1}, "sym":-1},
    "=der:g5sixxDz:sym=": {"dir": (0,0), "idx": (11,11), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5sixyDz:sym=": {"dir": (2,1), "idx": (10,11), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5sixzDz:sym=": {"dir": (2,2), "idx": (11,11), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5siy0Dz:sym=": {"dir": (2,3), "idx": (14,12), "fct": {"dn":  -1, "up":   1}, "sym":-1},
    "=der:g5siyxDz:sym=": {"dir": (2,0), "idx": (10,12), "fct": {"dn":  -1, "up":   1}, "sym":-1},
    "=der:g5siyyDz:sym=": {"dir": (1,1), "idx": (12,12), "fct": {"dn": 0.5, "up":-0.5}, "sym": 1},
    "=der:g5siyzDz:sym=": {"dir": (2,2), "idx": (12,12), "fct": {"dn":   1, "up":  -1}, "sym": 1},
    "=der:g5siz0Dz:sym=": {"dir": (2,2), "idx": (15,15), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
    "=der:g5sizxDz:sym=": {"dir": (2,2), "idx": (11,11), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
    "=der:g5sizyDz:sym=": {"dir": (2,2), "idx": (12,12), "fct": {"dn":-0.5, "up": 0.5}, "sym": 1},
}

def pconv(tsink, proj, flav):
    ts = "dt%2d" % int(tsink.split("_")[1])
    pr = {"G4": "P0", "G5G1": "P4",  "G5G2": "P5",  "G5G3": "P6"}[proj.split("_")[1]]
    fl = {"up": "dn", "down": "up"}[flav]
    return ts,pr,fl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("FNAME", type=str)
    parser.add_argument("-o", "--output", metavar="F", type=str,
                        help="output file name (default: FNAME.mom)")
    args = parser.parse_args()
    fname = args.FNAME
    output = args.output

    with h5py.File(fname, "r") as fp:
        for top in fp:
            if "conf" in top:
                break
        moms = np.array(fp["Momenta_list_xyz"])
        msqs = np.array(list(sorted(set((moms**2).sum(axis=1)))))
        spos = list(fp[top])[0]
        for tsink in fp[top][spos]:
            for proj in fp[top][spos][tsink]:
                for flav in fp[top][spos][tsink][proj]:
                    ts,pr,fl = pconv(tsink, proj, flav)
                    prsign = {"P0": +1,
                              "P4": -1,
                              "P5": -1,
                              "P6": -1}[pr]
                    fn = "ft_thrp_%s_gN90a0p2_aN50a0p5_%s_%s.%s.h5" % (spos, pr, ts, fl)
                    print(fn)
                    with h5py.File(fn, "w") as hf:
                        grp = hf.require_group("thrp/%s/%s/%s/%s/" % (spos, pr, ts, fl))
                        for msq in msqs:
                            idx = (moms**2).sum(axis=1) == msq
                            idx = list(reversed(np.arange(len(idx))[idx]))
                            mv = moms[idx,:]
                            #... ultra-local
                            arr = np.array(fp[top][spos][tsink][proj][flav]["ultra_local/threep"],
                                           dtype=np.float64)
                            subarr = arr[:,idx,:,:]*prsign
                            for op in local_ops:
                                ix = local_ops[op]["idx"]
                                fct = local_ops[op]["fct"][fl]
                                dat = (subarr[:,:,ix,0] + I*subarr[:,:,ix,1])*fct
                                dgrp = grp.require_group("%s/msq%04d" % (op, msq))
                                dgrp.create_dataset("arr", shape=dat.shape, dtype=dat.dtype, data=dat)
                                dgrp.create_dataset("mvec", shape=mv.shape, dtype=mv.dtype, data=mv)
                            #... noether
                            arr = np.array(fp[top][spos][tsink][proj][flav]["noether/threep"],
                                           dtype=np.float64)
                            subarr = arr[:,idx,:,:]*prsign
                            for op in noether_ops:
                                ix = noether_ops[op]["idx"]
                                fct = noether_ops[op]["fct"][fl]
                                dat = (subarr[:,:,ix,0] + I*subarr[:,:,ix,1])*fct
                                dgrp = grp.require_group("%s/msq%04d" % (op, msq))
                                dgrp.create_dataset("arr", shape=dat.shape, dtype=dat.dtype, data=dat)
                                dgrp.create_dataset("mvec", shape=mv.shape, dtype=mv.dtype, data=mv)
                            #... derivative operators
                            subarr = list()
                            for di in range(4):
                                x = np.array(fp[top][spos][tsink][proj][flav]["oneD"]["dir_%02d/threep" % di])*prsign
                                y = x[:,idx,:,:]
                                subarr.append(np.array(y[...,0]+I*y[...,1], np.complex64))
                            for op in deriv_ops:
                                di = deriv_ops[op]["dir"]
                                ix = deriv_ops[op]["idx"]
                                fct = deriv_ops[op]["fct"][fl]
                                sym = deriv_ops[op]["sym"]
                                le = subarr[di[0]][:,:,ix[0]]
                                ri = subarr[di[1]][:,:,ix[1]]
                                dat = 0.5*(le + sym*ri)*fct
                                dgrp = grp.require_group("%s/msq%04d" % (op, msq))
                                dgrp.create_dataset("arr", shape=dat.shape, dtype=dat.dtype, data=dat)
                                dgrp.create_dataset("mvec", shape=mv.shape, dtype=mv.dtype, data=mv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
