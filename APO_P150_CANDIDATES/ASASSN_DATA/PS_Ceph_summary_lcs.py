#!/usr/bin/env python

import os
import numpy as np
import ML_stuff
#from matplotlib import rc
import matplotlib.pyplot as plt
import healpy as hp
from gatspy.periodic import LombScargleMultiband
import pandas as pd
from time import time
from glob import glob
from lomb_scargle_refine import lomb as lombr
import sys

lightcurves = np.load('/home/inno/projects/PS1/Cepheids/apo_cand.npy')
ra_all, dec_all = lightcurves['ra'], lightcurves['dec']

objids_all =lightcurves['obj_id']





# load the list of objids to be processed
#objids = np.loadtxt(sys.argv[1], dtype='u8')
#objids = np.atleast_1d(objids)
#ra = lightcurves['ra']
#dec = lightcurves['dec']
#objids_name = ["" for x in range(len(ra))]
#for this_star in range(len(ra)):
#        if (this_star % 10):
#        objids_name[this_star] = str(np.around(ra[this_star],5))+'_'+str(np.around(dec[this_star],5))

#objids_name_list=np.unique(objids_name)

objids = np.unique(lightcurves['obj_id'])
#ls_results = pd.read_csv('ASASSN/PS_Ceph/PS_Ceph_diffflux_summary.csv', header=0)
#lc_objid = ls_results['objid'].values
#lc_periods = ls_results['period_all'].values


def matchtols(ra,dec):
    ra=np.float(ra)
    dec=np.float(dec)
    sel = ((np.absolute(ra_all-ra) <0.00002) & (np.absolute(dec_all-dec)< 0.00002))
    print sel
    objid_ps = objids_all[sel]
    return objid_ps[0]

#n_proc = objids.size




plot_lc = True
save = True

if __name__ == "__main__":
    fnames = np.sort(glob('diffflux/*.dat'))
    f = open('PS_Ceph_diffflux_summary.csv', 'w')
    f.write('objid_ps,objid,ra,dec,n_obs_good,n_obs_all,mean_flux_good,mean_mag_good,mean_flux_all,period_good,period_all,baseline\n')
    for fname in fnames:
        objid = fname.split('/')[1].split('.dat')[0]
        ra = objid.split('_')[0]
        dec = objid.split('_')[1]
        objid_ps=matchtols(ra,dec)
        print objid, objid_ps
        lc_data = pd.read_csv(fname, sep='\s+', header=None,names=['JD','HJD','UT_date','IMAGE','FWHM','Diff', 'Limit','mag','mag_err','counts','count_err','flux','flux_err'],comment='/',skiprows=[0,1],error_bad_lines=False)
        lc_data = lc_data[lc_data['mag_err'] == lc_data['mag_err']]
        n_obs_all =lc_data['HJD'].size
        good_data = lc_data[lc_data['mag_err'] < 99.]
        time_sort = np.argsort(lc_data['HJD'])
        baseline = lc_data['UT_date'][time_sort]
        base= np.int(baseline[baseline.size-1].split('-')[0])-np.int(baseline[0].split('-')[0])
        mean_mag_all = 99999.9
        period_all = 99999.9
        mean_mag_good = 99999.9
        period_good = 99999.9
        mag_all = lc_data['flux'].values.astype(float)
        emag_all = lc_data['flux_err'].values.astype(float)
        mean_mag_all = np.average(mag_all,  weights=1./emag_all)
        mean_mag_good  = 99999.9
        n_obs_good = good_data['HJD'].size
        mjd_all = lc_data['HJD'].values.astype(float)- 2400000.5
        Dt = mjd_all.max()-mjd_all.min()
        f0 = 100./Dt
        df = 0.1/Dt
        fe = 1.00
        numf = int((fe-f0)/df)
        freqin = f0 + df*np.arange(numf,dtype='float64')
        ## Find the period
        psd, res = lombr(mjd_all, mag_all, emag_all, f0, df, numf, detrend_order=0, nharm=8)
        res_all = res
        if psd[0] == psd[0]:
            p1 = 1./freqin[np.where(psd == max(psd))][0]
            
            ## Find the epoch of maximum light
            phi1 = mjd_all[np.argmin(res['model'])]
            print p1, phi1
            period_all = p1
            # Plot the folded light curve and model
            if plot_lc:
                import matplotlib.pyplot as plt
                ax = plt.axes([0.1, 0.1, 0.35, 0.8])
                ax.set_title(r'%s, Period = %.5f' % (objid, period_all))
                plt.plot(freqin, psd)
                plt.ylabel('PWS')
                plt.xlabel('freq')
                tt=(mjd_all/p1) % 1.; s=tt.argsort()
                ax = plt.axes([0.6, 0.1, 0.35, 0.35])
                plt.errorbar(tt, mag_all, yerr=emag_all, fmt='o')
                plt.plot(tt[s], res['model'][s])
                plt.ylabel('Flux (mJy)')
                plt.xlabel('phase')
                plt.axis([0, 1, res['model'][s].min()-0.07,res['model'][s].max()+0.07])
                
                #plt.show()
                
                #if save is True:
                #    plt.savefig('diffflux/plots_diff/%s.png' % objid)
                #plt.close()
        if n_obs_good > 10:
            mjd_good = good_data['HJD'].values.astype(float)- 2400000.5
            mag_good = good_data['flux'].values.astype(float)
            emag_good = good_data['flux_err'].values.astype(float)
            mean_mag_good = np.average(mag_good,  weights=1./emag_good)
            Dt = mjd_good.max()-mjd_good.min()
            f0 = 100./Dt
            df = 0.1/Dt
            fe = 1.00
            numf = int((fe-f0)/df)
            freqin = f0 + df*np.arange(numf,dtype='float64')
            ## Find the perioda
            psd, res = lombr(mjd_good, mag_good, emag_good, f0, df, numf, detrend_order=0, nharm=8)
            p1 = 1./freqin[np.where(psd == max(psd))][0]
            
            ## Find the epoch of maximum light
            phi1 = mjd_good[np.argmin(res['model'])]
            print p1, phi1
            period_good = p1
            # Plot the folded light curve and model
            if plot_lc:
                
                mag_good_p = good_data['mag'].values.astype(float)
                emag_good_p = good_data['mag_err'].values.astype(float)
                #psd_p, res_p = lombr(mjd_good, mag_good_p, emag_good_p, f0, df, numf, detrend_order=0, nharm=8)
                #ax = plt.axes([0.1, 0.1, 0.35, 0.8])
                #ax.set_title(r'%s, Period = %.5f' % (objid, period_good))
                #plt.plot(freqin, psd)
                #plt.ylabel('PWS')
                #plt.xlabel('freq')
                tt=(mjd_good/period_all) % 1.; s=tt.argsort()
                ax = plt.axes([0.6, 0.55, 0.35, 0.35])
                plt.errorbar(tt, mag_good_p-np.median(mag_good_p), yerr=emag_good_p, fmt='o', ecolor='g', mfc='g')
                ax.set_title(r'<V> = %.5f' % (np.median(mag_good_p)))
                #plt.plot(tt[s], res_p['model'][s], 'g-')
                plt.ylabel('V - <V> [mag]')
                plt.xlabel('phase')
                plt.axis([0, 1,-2, +2])
                #plt.axis([0, 1, 1.05*res_p['model'][s].max(), 0.95*res_p['model'][s].min()])
                #plt.axis([0, 1, res['model'][s].max()+0.05],res['model'][s].min()-0.05)


        if n_obs_good > 2:
        
            magg_good = good_data['mag'].values.astype(float)
            emagg_good = good_data['mag_err'].values.astype(float)
            mean_magg_good = np.average(magg_good,  weights=1./emagg_good)
        if save is True:
            plt.savefig('diffflux/plots_diff/%s.png' % objid_ps)
        plt.close()
    
        print objid_ps,objid,ra, dec, n_obs_good, n_obs_all, mean_mag_good,mean_magg_good,mean_mag_all,period_good,period_all,base
        f.write('%s,%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n' % (objid_ps,objid,ra, dec,n_obs_good, n_obs_all, mean_mag_good,mean_magg_good,mean_mag_all,period_good,period_all,base))

f.close()







