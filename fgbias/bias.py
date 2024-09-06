 #Having generated alms, do the reconstuction
#and cross-correlate with true kappa
import pickle
import healpy as hp
import numpy as np
from falafel import utils, qe
import pytempura
from pytempura import norm_general, noise_spec
import solenspipe
from pixell import lensing, curvedsky, enmap
from pixell import utils as putils
from os.path import join as opj
import argparse
import yaml
from collections import OrderedDict
from orphics import maps
from websky_model import WebSky
from copy import deepcopy
import sys
from scipy.signal import savgol_filter

def get_all_secondary_terms(
        qfunc_TT, qfunc_TE, 
        Tf1, cmb_alm, 
        cmb_prime_alm, Tf2=None,
        qfunc_tb=None):
    #Get all secondary terms
    #i.e. TTTT, TTTE, TETE, TTTB, TBTB
    #if Tf2_alm != Tf1_alm, there would be more
    #potentially. But let's assume for now
    #that T1 is used for the T-pol estimators
    equal_Tf = False
    if Tf2 is None:
        equal_Tf=True
        Tf2=Tf1
        
    #make sure cmb alms are the right format
    for x in [cmb_alm, cmb_prime_alm]:
        assert len(x)==3

    #First do TTTT
    #Tcmb, Tcmb_prime = cmb_alm[0], cmb_prime_alm[0]
    phi_Tcmb_Tf2 = qfunc_TT(cmb_alm, Tf1)
    phi_Tf1_Tcmb = qfunc_TT(Tf1, cmb_alm)
    phi_Tcmbp_Tf2 = qfunc_TT(cmb_prime_alm, Tf2)
    phi_Tf1_Tcmbp = qfunc_TT(Tf2, cmb_prime_alm)

    kappa_TT = lensing.phi_to_kappa(
        phi_Tcmb_Tf2[0]+phi_Tf1_Tcmb[0])
    kappa_TTp = lensing.phi_to_kappa(
        phi_Tcmbp_Tf2[0]+phi_Tf1_Tcmbp[0])

    S_TTTT = curvedsky.alm2cl(kappa_TT)-curvedsky.alm2cl(kappa_TTp)

    #Now pol
    #E, E_prime, B, B_prime = (cmb_alm[1], cmb_prime_alm[1],
    #                          cmb_alm[2], cmb_prime_alm[2])
    #print("E[100:110]:", E[100:110])
    #print("E_prime[100:110]:", E_prime[100:110])
    kappa_Tf1_Ecmb = lensing.phi_to_kappa(
        qfunc_TE(Tf1, cmb_alm)[0]
        )
    print("kappa_Tf1_Ecmb[100:110]:", kappa_Tf1_Ecmb[100:110])
    kappa_Tf1_Ecmbp = lensing.phi_to_kappa(
        qfunc_TE(Tf1, cmb_prime_alm)[0]
        )
    print("kappa_Tf1_Ecmbp[100:110]:", kappa_Tf1_Ecmbp[100:110])

    S_TTTE = (
        curvedsky.alm2cl(kappa_TT, kappa_Tf1_Ecmb)
        - curvedsky.alm2cl(kappa_TTp, kappa_Tf1_Ecmbp)
        )
    S_TETE = (curvedsky.alm2cl(kappa_Tf1_Ecmb)
              - curvedsky.alm2cl(kappa_Tf1_Ecmbp)
              )
    #let's return a dictionary here
    #becuase there's more than a couple of
    #things to return
    S = {"TTTT" : S_TTTT,
         "TTTE" : S_TTTE,
         "TETE" : S_TETE,}
    
    if qfunc_tb is not None:
        kappa_Tf1_Bcmb = lensing.phi_to_kappa(
            qfunc_tb(Tf1, B)[0]
            )
        kappa_Tf1_Bcmbp = lensing.phi_to_kappa(
            qfunc_tb(Tf1, B_prime)[0]
            )
        S_TTTB = (
            curvedsky.alm2cl(kappa_TT, kappa_Tf1_Bcmb)
            - curvedsky.alm2cl(kappa_TTp, kappa_Tf1_Bcmbp)
            )
        S_TBTB = (curvedsky.alm2cl(kappa_Tf1_Bcmb)
                  - curvedsky.alm2cl(kappa_Tf1_Bcmbp)
                  )
        S["TTTB"] = S_TTTB
        S["TBTB"] = S_TBTB
    return S

def get_bias_terms(fg_alms, recon_setup, 
                   phi_alms, cmb_alm, cmbp_alm,
                   ests=["qe","psh","prh"], comm=None):
    
    fg_alms_filtered = recon_setup["filter"](fg_alms)
    
    jobs = []
    for est in ests:
        jobs.append(
            (est, recon_setup["qfunc_%s"%est],
             None,
             recon_stuff["get_fg_trispectrum_phi_N0_%s"%est]
            )
        )
        
    outputs = {}
        
    for i,job in enumerate(jobs):
        est_name, qfunc, qfunc_te, get_tri_N0 = job
        #Do secondary
        print("doing secondary")
        if qfunc_te is not None:
            secondary_terms = get_all_secondary_terms(
                qfunc, qfunc_te, fg_alms_filtered,
                cmb_alms_filtered, cmbp_alms_filtered
                )
            outputs["secondary_%s"%est_name] = secondary_terms["TTTT"]
            outputs["secondary_TTTE_%s"%est_name] = secondary_terms["TTTE"]
            outputs["secondary_TETE_%s"%est_name] =	secondary_terms["TETE"]
        else:
            outputs["secondary_%s"%est_name] = get_TT_secondary(
                qfunc, fg_alms_filtered,
                cmb_alms_filtered[0], cmbp_alms_filtered[0], Tf2=None)

        print("doing fg reconstruction")
        phi_fg_fg = qfunc(
            fg_alms_filtered, fg_alms_filtered)

        kappa_fg_fg = lensing.phi_to_kappa(phi_fg_fg[0])

        if args.save_fg_reconstruction:
            f = opj(out_dir, 'kappa_fg_%s_%s.fits'%(freq,est_name))
            print("saving kappa fg reconstruction to %s"%f)
            hp.write_alm(f, kappa_fg_fg, overwrite=True)

        #Do primary
        outputs['primary_'+est_name] = 2*curvedsky.alm2cl(kappa_fg_fg, kappa_alms)


        #Do trispectrum
        cl_tri_raw = curvedsky.alm2cl(kappa_fg_fg, kappa_fg_fg)
        N0_phi = get_tri_N0(cl_fg)[0] #0th element for gradient
        N0_tri = N0_phi * (L*(L+1.)/2)**2

        outputs['trispectrum_'+est_name] = cl_tri_raw - N0_tri
        outputs['tri_N0_'+est_name] = N0_tri

        outputs['total_'+est_name] = (outputs['primary_%s'%est_name]
                                   +outputs['secondary_%s'%est_name]
                                   +outputs['trispectrum_%s'%est_name]
        )

    output_data = np.zeros((recon_config["mlmax"]+1),
                           dtype=[(k,float) for k in outputs.keys()])
    for k,v in outputs.items():
        try:
            print(k,v)
            assert len(v)==recon_config["mlmax"]+1
            output_data[k] = v
        except ValueError as e:
            print("failed to write column %s to output array"%k)
            raise(e)
    output_file = opj(out_dir, 'fg_terms_%s.npy'%freq)
    print("saving to %s"%output_file)
    np.save(output_file, output_data)
        
    