import numpy as np
import healpy as hp
import pytempura
from pytempura import norm_general, noise_spec
import os
from falafel import utils as futils, qe
from orphics import maps
from pixell import lensing, curvedsky
import matplotlib.pyplot as plt
from os.path import join as opj
import pickle

try:
    WEBSKY_DIR=os.environ["WEBSKY_DIR"]
except KeyError:
    WEBSKY_DIR="/global/project/projectdirs/act/data/maccrann/websky"
try:
    SEHGAL_DIR=os.environ["SEHGAL_DIR"]
except KeyError:
    SEHGAL_DIR="/global/project/projectdirs/act/data/maccrann/sehgal"
    


def norm_qtt_asym(est,lmax,glmin,glmax,llmin,llmax,
                   rlmax,TT,OCTG,OCTL,gtype='',profile=None):
    if ((est=='src') and (profile is not None)):
        norm = norm_general.qtt_asym(
            est,lmax,glmin,glmax,llmin,llmax,
            rlmax, TT, OCTG/(profile[:glmax+1]**2),
            OCTL/(profile[:llmax+1]**2), 
            gtype=gtype)
        print(norm[0].shape, profile.shape)
        return (norm[0]*profile**2, norm[1]*profile**2)
    else:
        return norm_general.qtt_asym(
            est,lmax,glmin,glmax,llmin,llmax,
                   rlmax,TT,OCTG,OCTL,gtype=gtype)
    
def norm_xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                   TT,OCTG,OCTL,gtype='',profile=None):

    if ((est=="lenssrc") and (profile is not None)):
        r = norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                  TT, OCTG/(profile[:llmax+1]), OCTL/(profile[:llmax+1]), gtype=gtype)
        return r/profile
    elif ((est=="srclens") and (profile is not None)):
        r = norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                  TT, OCTG/(profile[:glmax+1]), OCTL/(profile[:glmax+1]), gtype=gtype)
        return r/profile
    else:
        return norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                     TT, OCTG, OCTL, gtype=gtype)
    
def dummy_teb(alms):
    return np.array([alms, np.zeros_like(alms), np.zeros_like(alms)])

def setup_recon(px, lmin, lmax, mlmax,
                tcls_A, tcls_B=None,
                tcls_C=None, tcls_D=None,
                tcls_AC=None, tcls_BD=None,
                tcls_AD=None, tcls_BC=None,
                do_Tpol=False, do_psh=False, do_prh=False,
                do_psh_prh=False, profile=None):
    if tcls_B is None:
        return setup_AAAA_recon(
            px, lmin, lmax, mlmax, tcls_A,
            do_Tpol=do_Tpol, do_psh=do_psh,
            do_prh=do_prh, do_psh_prh=do_psh_prh,
            profile=profile)
    else:
        raise("NotImplementedError")
    
    
def setup_AAAA_recon(px, lmin, lmax, mlmax,
                tcls, do_Tpol=False,
                do_psh=False, do_prh=False, do_psh_prh=False,
                profile=None):
    """
    Setup needed for reconstruction and foreground
    bias estimation. This is for the simplest case
    where we use the same temperature map "A" in all 
    four legs.
    """
    ucls,_ = futils.get_theory_dicts(grad=True, lmax=mlmax)
        
    recon_stuff = {}
    recon_stuff["mlmax"] = mlmax
    recon_stuff["lmax"] = lmax
    recon_stuff["lmin"] = lmin

    #Get norms from tempura
    norms = pytempura.get_norms(
        ["TT"]+["TE", "TB", "EE", "EB", "BB"], ucls, ucls,
        {c:tcls[c][:mlmax+1] for c in tcls.keys()},
        lmin, lmax, k_ellmax=mlmax)
    recon_stuff["norms"] = norms

    norm_lens = norms['TT']

    def filter_alms(alms):
        if len(alms)!=3:
            alms = dummy_teb(alms)
        alms_filtered = futils.isotropic_filter(alms,
                tcls, lmin, lmax, ignore_te=True)
        return alms_filtered

    recon_stuff["filter"] = filter_alms

    def qfunc_tt_qe(X_filtered, Y_filtered):
        phi_nonorm = qe.qe_all(px, ucls, mlmax,
                                fTalm=X_filtered, fEalm=None,fBalm=None,
                                estimators=['TT'],
                                xfTalm=Y_filtered, xfEalm=None,xfBalm=None)['TT']
        #normalize and return
        return (curvedsky.almxfl(phi_nonorm[0], norm_lens[0]),
                curvedsky.almxfl(phi_nonorm[1], norm_lens[1]))
    
    recon_stuff["qfunc_tt_qe"] = qfunc_tt_qe
    recon_stuff["qfunc_tt_qe_incfilter"] = lambda X,Y: qfunc(filter_alms_X(X),
                                                                filter_alms_X(Y))

    #Get the N0
    recon_stuff["N0_phi"] = norm_lens

    #Also define functions here for getting the
    #trispectrum N0
    def get_fg_trispectrum_phi_N0(cl_fg):
        #N0 is (A^phi)^2 / A_fg (see eqn. 9 of 1310.7023
        #for the normal estimator, and a bit more complex for
        #the bias hardened case
        Ctot = tcls['TT']**2 / cl_fg
        norm_fg = pytempura.norm_lens.qtt(
            mlmax, lmin,
            lmax, ucls['TT'],ucls['TT'],
            Ctot,gtype='')

        return (norm_lens[0]**2 / norm_fg[0],
                norm_lens[1]**2 / norm_fg[1])

    recon_stuff["get_fg_trispectrum_phi_N0_qe"] = get_fg_trispectrum_phi_N0

    if do_Tpol:
        def qfunc_te_qe(X_filtered, Y_filtered):
            phi_nonorm = qe_all(px, ucls, mlmax,
                                fTalm=Y_filtered[0], fEalm=Y_filtered[1],fBalm=Y_filtered[2],
                                estimators=['TE'],
                                xfTalm=X_filtered[0], xfEalm=X_filtered[1],xfBalm=X_filtered[1])['TE']
            #normalise and return
            return np.asarray(
                (curvedsky.almxfl(phi_nonorm[0], norms["TE"][0])),
                (curvedsky.almxfl(phi_nonorm[1], norms["TE"][1])),
            )

        if do_psh:
            raise NotImplementedError("not yet implemented Tpol with psh")
        
    
    if do_psh:
        R_src_tt = pytempura.get_cross(
            'SRC','TT',ucls,tcls,lmin,lmax,
            k_ellmax=mlmax)
        norm_src = pytempura.get_norms(
                ['src'], ucls, ucls, tcls,
                lmin, lmax,
                k_ellmax=mlmax)['src']

        #Get N0
        recon_stuff["N0_phi_psh"] = (
            norm_lens[0] / (1 - norm_lens[0]*norm_src
                              *R_src_tt**2),
            norm_lens[1] / (1 - norm_lens[1]*norm_src
                              *R_src_tt**2),

            )

        def qfunc_tt_psh(X_filtered, Y_filtered):
            
            phi_nonorm = qe.qe_all(px, ucls, mlmax,
                        fTalm=X_filtered, fEalm=None,fBalm=None,
                        estimators=['TT'],
                        xfTalm=Y_filtered, xfEalm=None,xfBalm=None)['TT']
            
            src_nonorm = qe.qe_source(px, mlmax, Y_filtered,
                                      xfTalm=X_filtered)
            
            #print("src_nonorm", src_nonorm)
            #print("norm_src", norm_src)
            #print("R_src_tt", R_src_tt)
            
            phi_psh_grad = (
                curvedsky.almxfl(phi_nonorm[0], norm_lens[0]) - 
                curvedsky.almxfl(src_nonorm, norm_lens[0] * norm_src * R_src_tt)
                           )
            phi_psh_grad = curvedsky.almxfl(phi_psh_grad,
                                            1./(1. - norm_lens[0] * norm_src * R_src_tt**2.)
                                           )
            phi_psh_curl = curvedsky.almxfl(phi_nonorm[1], norm_lens[1])
            return np.array([phi_psh_grad, phi_psh_curl])
        
        #qfunc_psh = solenspipe.get_qfunc(
        #    px, ucls, mlmax, "TT", est2='SRC', Al1=norms['TT'],
        #    Al2=norm_src, R12=R_src_tt)
        recon_stuff["qfunc_tt_psh"] = qfunc_tt_psh
        recon_stuff["qfunc_tt_psh_incfilter"] = lambda X,Y: qfunc_psh(filter_alms(X),
                                                                filter_alms(Y))
        def get_fg_trispectrum_phi_N0_psh(cl_fg):
            Ctot = tcls['TT']**2 / cl_fg
            norm_fg = pytempura.norm_lens.qtt(
                mlmax, lmin,
                lmax, ucls['TT'], ucls['TT'],
                Ctot,gtype='')
            norm_src_fg = pytempura.get_norms(
                ['src'], ucls, ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)['src']
            R_src_fg = pytempura.get_cross(
                'SRC','TT', ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)
            N0_tris = []
            #gradient and curl:
            for i in [0,1]:
                N0 = norm_lens[i]**2/norm_fg[i]
                N0_s = norm_src**2/norm_src_fg
                N0_tri = (N0 + R_src_tt**2 * norm_lens[i]**2 * N0_s
                          - 2 * R_src_tt * norm_lens[i]**2 * norm_src * R_src_fg)
                N0_tri /= (1 - norm_lens[i]*norm_src*R_src_tt**2)**2
                N0_tris.append(N0_tri)
            return tuple(N0_tris)

        recon_stuff["get_fg_trispectrum_phi_N0_psh"] = get_fg_trispectrum_phi_N0_psh

    if do_prh:
        R_prof_tt = pytempura.get_cross(
            'SRC', 'TT', ucls, tcls, lmin,
            lmax, k_ellmax=mlmax,
            profile=profile)
        R_prof_te = pytempura.get_cross(
            'SRC', 'TE', ucls, tcls, lmin,
            lmax, k_ellmax=mlmax,
            profile=profile)

        norm_prof = pytempura.get_norms(
            ['TT','src'], ucls, tcls,
            lmin, lmax,
            k_ellmax=mlmax, profile=profile)['src']


        recon_stuff["N0_phi_prh"] = (
            norm_lens[0] / (1 - norm_lens[0]*
                           norm_prof*
                           R_prof_tt**2),
            norm_lens[1] / (1 - norm_lens[1]*
                           norm_prof*
                           R_prof_tt**2)
        )

        
        #qfunc_prh = solenspipe.get_qfunc(
        #    px, ucls, mlmax, "TT",
        #    Al1=norms['TT'], est2='SRC', Al2=norm_prof,
        #    R12=R_prof_tt, profile=profile)
        
        def qfunc_tt_prh(X_filtered, Y_filtered):
            
            phi_nonorm = qe.qe_all(px, ucls, mlmax,
                        fTalm=X_filtered, fEalm=None,fBalm=None,
                        estimators=['TT'],
                        xfTalm=Y_filtered, xfEalm=None,xfBalm=None)['TT']
            
            prof_nonorm = qe.qe_source(px, mlmax, Y_filtered,
                                      xfTalm=X_filtered, 
                                      profile=profile)
            
            #print("src_nonorm", src_nonorm)
            #print("norm_src", norm_src)
            #print("R_src_tt", R_src_tt)
            
            phi_prh_grad = (curvedsky.almxfl(phi_nonorm[0], norm_lens[0]) - 
                            curvedsky.almxfl(prof_nonorm, norm_lens[0] * norm_prof * R_prof_tt))
            phi_prh_grad = curvedsky.almxfl(phi_prh_grad,
                                            1./(1. - norm_lens[0] * norm_prof * R_prof_tt**2.)
                                           )
            phi_prh_curl = curvedsky.almxfl(phi_nonorm[1], norm_lens[1])
            return np.array([phi_prh_grad, phi_prh_curl])

        recon_stuff["profile"] = profile
        recon_stuff["qfunc_tt_prh"] = qfunc_prh
        recon_stuff["qfunc_tt_prh_incfilter"] = lambda X,Y: qfunc_prh(filter_alms(X),
                                                                filter_alms(Y))
        recon_stuff["R_prof_tt"] = R_prof_tt
        recon_stuff["norm_prof"] = norm_prof

        def get_fg_trispectrum_phi_N0_prh(cl_fg):
            Ctot = tcls['TT']**2 / cl_fg
            norm_lens = norm_lens
            norm_fg = pytempura.norm_lens.qtt(
                mlmax, lmin,
                lmax, ucls['TT'],
                Ctot,gtype='')

            norm_src_fg = pytempura.get_norms(
                ['TT','src'], ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax, profile=profile)['src']
            R_src_fg = pytempura.get_cross(
                'SRC','TT', ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax, profile=profile)

            #gradient and curl
            N0_tris=[]
            for i in [0,1]:
                N0 = norm_lens[i]**2/norm_fg[i]
                N0_s = norm_prof**2/norm_src_fg
                N0_tri = (N0 + R_prof_tt**2 * norm_lens[i]**2 * N0_s
                          - 2 * R_prof_tt * norm_lens[i]**2 * norm_prof * R_src_fg)
                N0_tri /= (1 - norm_lens[i]*norm_prof*R_prof_tt**2)**2
                N0_tris.append(N0_tri)
            return tuple(N0_tris)

        recon_stuff["get_fg_trispectrum_phi_N0_prh"] = get_fg_trispectrum_phi_N0_prh

        if do_psh_prh:
            try:
                assert (do_psh and do_prh)
            except AssertionError as e:
                print("need do_psh=True and do_prh=True for do_psh_prh=True")
            #Since we've set up profile-hardening, we might as well also set up
            #source + profile-hardening. Should just need the profile-source response
            #in addition to everything else we've computed already.
            R_prof_src = (1./pytempura.get_norms(
                ['src'], ucls, tcls,
                lmin, lmax, k_ellmax=mlmax,
                profile = profile**0.5)['src'])
            R_prof_src[0] = 0.
            R_matrix = np.ones((mlmax, 3, 3))
            R_matrix[:,0,1] = norm_lens_X[0] * R_src_tt
            R_matrix[:,0,2] = norm_lens_X[0] * R_prof_tt
            R_matrix[:,1,0] = norm_src * R_src_tt
            R_matrix[:,1,2] = norm_src * R_prof_src
            R_matrix[:,2,0] = norm_prof * R_prof_tt
            R_matrix[:,2,1] = norm_prof * R_prof_src
            R_inv = np.zeros_like(R_matrix)
            for l in range(mlmax+1):
                R_inv[l] = np.linalg.inv(R_matrix[l])

            def qfunc_psh_prh(X_filtered, Y_filtered):
                phi = qfunc(X_filtered, Y_filtered)[0]
                ps = qfunc_psh(X_filtered, Y_filtered)
                pr = qfunc_prh(X_filtered, Y_filtered)
                #to get phi, we just need the first two of the
                #R_inv matrix
                phi_bh = (phi[0] + curvedsky.almxfl(R_inv[0,1], ps)
                          + curvedsky.almxfl(R_inv[0,2], pr),
                          phi[1]
                          )
                return phi_bh

            R_other = R_matrix[:, 1:, 1:]
            detR = np.array([np.linalg.det(R_matrix[l]) for l in range(mlmax+1)])
            detR_other = np.array([np.linalg.det(R_other[l]) for l in range(mlmax+1)])
            print("R_inv_ps_pr.shape:", R_inv_ps_pr.shape)
            recon_stuff["N0_phi_prh"] = (
                norm_lens[0] * detR_other / detR,
                norm_lens[1]
            )
            def get_fg_trispectrum_phi_N0_psh_prh(cl_fg):
                Ctot = tcls['TT']**2 / cl_fg
                norm_lens = norm_lens
                norm_fg = pytempura.norm_lens.qtt(
                    mlmax, lmin,
                    lmax, ucls['TT'],
                    Ctot,gtype='')

                norm_ps_fg = pytempura.get_norms(
                    ['TT','src'], ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax)['src']
                R_ps_fg = pytempura.get_cross(
                    'SRC','TT', ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax)
                norm_prof_fg = pytempura.get_norms(
                    ['TT','src'], ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax, profile=profile)['src']
                R_prof_fg = pytempura.get_cross(
                    'SRC','TT', ucls, {'TT':Ctot},
                    lmin, lmax,
                    k_ellmax=mlmax, profile=profile)
                R_prof_ps_fg = (1./pytempura.get_norms(
                    ['src'], ucls, {'TT':Ctot},
                    lmin, lmax, k_ellmax=mlmax,
                    profile = profile**0.5)['src'])

                #gradient and curl
                N0_tris=[]
                for i in [0,1]:
                    N0 = norm_lens[i]**2 / norm_fg[i]
                    N0_ps = norm_ps**2/norm_ps_fg
                    N0_prof = norm_prof**2/norm_prof_fg
                    #continue from here
                    N0_tri = (N0 + R_prof_tt**2 * norm_lens[i]**2 * N0_prof
                              - 2 * R_prof_tt * norm_lens[i]**2 * norm_prof * R_src_fg)
                    N0_tri /= (1 - norm_lens[i]*norm_prof*R_prof_tt**2)**2
                    N0_tris.append(N0_tri)
                return tuple(N0_tris)

            recon_stuff["get_fg_trispectrum_phi_N0_prh"] = get_fg_trispectrum_phi_N0_prh

    return recon_stuff
