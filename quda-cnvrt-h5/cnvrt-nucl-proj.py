import argparse
import numpy as np
import h5py
import sys, re

NS = 4
I = complex(0,1)

one = np.eye(NS, dtype=complex)

gt = np.array([[ 1, 0, 0, 0],
               [ 0, 1, 0, 0],
               [ 0, 0,-1, 0],
               [ 0, 0, 0,-1]])

gproj = (one + gt)/4, (one - gt)/4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("FNAME", type=str)
    parser.add_argument("-o", "--output", metavar="D", type=str,
                        help="output directory name (default: FNAME.proj)")
    args = parser.parse_args()
    fname = args.FNAME
    output = args.output
    if output is None:
        output = fname + ".proj"
    
    with h5py.File(fname, "r") as fp:
        for top in fp:
            if "conf" in top:
                break
        moms = np.array(fp["Momenta_list_xyz"])
        msqs = np.array(list(sorted(set((moms**2).sum(axis=1)))))
        spos = list(fp[top])[0]
        nucl = np.array(
            [
                np.array(fp[top][spos]["nucl_nucl"]["twop_baryon_1"], np.float64),
                np.array(fp[top][spos]["nucl_nucl"]["twop_baryon_2"], np.float64),
            ]
        )
        nucl = nucl[...,0] + I*nucl[...,1]
        with h5py.File(output, "w") as hp:
            grp = hp.require_group("nucleons/%s" % spos)
            for msq in msqs:
                idx = (moms**2).sum(axis=1) == msq
                idx = np.array(list(reversed(np.arange(len(idx))[idx])))
                mv = moms[idx,:]
                LT = nucl.shape[1]
                for j,n in enumerate(["pmm","ppm"]):
                    for i,d in enumerate(["fwd","bwd"]):
                        dat = (nucl[j,...][:,idx,:].reshape(-1,NS,NS).dot(gproj[i])).trace(axis1=1, axis2=2)
                        dat = dat.reshape(LT,-1)
                        dgrp = grp.require_group("%s/1-1/%s/msq%04d" % (n,d,msq))
                        dgrp.create_dataset("arr", shape=dat.shape, dtype=dat.dtype, data=dat)
                        dgrp.create_dataset("mvec", shape=mv.shape, dtype=mv.dtype, data=mv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
