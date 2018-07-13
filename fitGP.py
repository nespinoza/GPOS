from mpl_toolkits.axes_grid.inset_locator import inset_axes
import batman
import seaborn as sns
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pymultinest
from scipy import interpolate
import numpy as np
import utils
import os

parser = argparse.ArgumentParser()
# This reads the lightcurve file. First column is time, second column is flux:
parser.add_argument('-lcfile', default=None)
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument('-eparamfile', default=None)
# This reads an output folder:
parser.add_argument('-ofolder', default='')
# This reads a value for the mean of Rp/Rs:
parser.add_argument('-pmean', default=None)
# This reads the standard deviation:
parser.add_argument('-psd', default=None)
# This defines the limb-darkening to be used:
parser.add_argument('-ldlaw', default='quadratic')
# Number of live points:
parser.add_argument('-nlive', default=1000)
args = parser.parse_args()

# Extract lightcurve and external parameters. When importing external parameters, 
# standarize them and save them on the matrix X:
lcfilename = args.lcfile
t,f = np.genfromtxt(lcfilename,unpack=True,usecols=(0,1))
# Float the times (batman doesn't like non-float 64):
t = t.astype('float64')
out_folder = args.ofolder
t0,P,aR,inc,ecc,omega = t[len(t)/2],3.0,10.,88.0,0.0,90.0

eparamfilename = args.eparamfile
data = np.genfromtxt(eparamfilename,unpack=True)
for i in range(len(data)):
    x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
    if i == 0:
        X = x
    else:
        X = np.vstack((X,x))

# Sace other inputs:
ld_law = args.ldlaw
pmean = np.double(args.pmean)
psd = np.double(args.psd) 
n_live_points = int(args.nlive)

# Cook the george kernel:
import george
kernel = np.var(f)*george.kernels.ExpSquaredKernel(np.ones(X.shape[0]),ndim=X.shape[0],axes=range(X.shape[0]))
# Cook jitter term
jitter = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

# Wrap GP object to compute likelihood
gp = george.GP(kernel, mean=0.0,fit_mean=False,white_noise=jitter,fit_white_noise=True)
#print gp.get_parameter_names(),gp.get_parameter_vector()
#print dir(gp)
#sys.exit()
gp.compute(X.T)

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    elif ld_law == 'linear':
        return q1,q2
    return coeff1,coeff2

def init_batman(t,law):
    """  
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    if law == 'linear':
        params.u = [0.5]
    else:
        params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law):
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    if ld_law == 'linear':
        params.u = [coeff1]
    else:
                params.u = [coeff1,coeff2]
    return m.light_curve(params)

# Initialize batman:
params,m = init_batman(t,law=ld_law)

# Now define MultiNest priors and log-likelihood:
def prior(cube, ndim, nparams):
    # Prior on "median flux" is uniform:
    cube[0] = utils.transform_uniform(cube[0],-2.,2.)
    # Pior on the log-jitter term (note this is the log VARIANCE, not sigma); from 1 to 10,000 ppm:
    cube[1] = utils.transform_uniform(cube[1],np.log((1e-6)**2),np.log((10000e-6)**2))
    # Prior on the planet-to-star radius ratio:
    cube[2] = utils.transform_truncated_normal(cube[2],pmean,psd)
    # (Transformed) limb-darkening coefficients:
    cube[3] = utils.transform_uniform(cube[3],0.,1.)
    pcounter = 4
    if ld_law != 'linear':
        cube[pcounter] = utils.transform_uniform(cube[pcounter],0.,1.)
        pcounter = pcounter + 1
    # Prior on kernel maximum variance; from 1 to 10,000 ppm: 
    cube[pcounter] = utils.transform_loguniform(cube[pcounter],(1*1e-6)**2,(10000*1e-6)**2)
    pcounter = pcounter + 1
    # Now priors on the alphas = 1/lambdas; gamma(1,1) = exponential, same as Gibson+:
    for i in range(X.shape[0]):
        cube[pcounter] = utils.transform_exponential(cube[pcounter])
        pcounter = pcounter + 1    

def loglike(cube, ndim, nparams):
    # Evaluate the log-likelihood. For this, first extract all inputs:
    mflux,ljitter,p,q1 = cube[0],cube[1],cube[2],cube[3]
    pcounter = 4
    if ld_law != 'linear':
        q2 = cube[pcounter]
        coeff1,coeff2 = reverse_ld_coeffs(ld_law,q1,q2)
        params.u = [coeff1,coeff2]
        pcounter = pcounter + 1
    else:
        params.u = [q1]
    max_var = cube[pcounter]
    pcounter = pcounter + 1
    alphas = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        alphas[i] = cube[pcounter]
        pcounter = pcounter + 1
    gp_vector = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas))
         
    # Evaluate transit model:
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = aR
    params.inc = inc
    params.ecc = ecc
    params.w = omega
    model = m.light_curve(params)    

    residuals = f - (mflux + model)
    gp.set_parameter_vector(gp_vector)
    return gp.log_likelihood(residuals)

if ld_law != 'linear':
    n_params = 6 + X.shape[0]
else:
    n_arams = 5 + X.shape[0]

out_file = out_folder+'out_multinest_trend_george_'

import pickle
# If not ran already, run MultiNest, save posterior samples and evidences to pickle file:
if not os.path.exists(out_folder+'posteriors_trend_george.pkl'):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    mc_samples = output.get_equal_weighted_posterior()[:,:-1]
    a_lnZ = output.get_stats()['global evidence']
    out = {}
    out['posterior_samples'] = mc_samples
    out['lnZ'] = a_lnZ
    pickle.dump(out,open(out_folder+'posteriors_trend_george.pkl','wb'))
else:
    mc_samples = pickle.load(open(out_folder+'posteriors_trend_george.pkl','rb'))['posterior_samples']
    

# Extract posterior parameter vector:
cube = np.median(mc_samples,axis=0)
cube_var = np.var(mc_samples,axis=0)

mflux,ljitter,p,q1 = cube[0],cube[1],cube[2],cube[3]
pcounter = 4
if ld_law != 'linear':
        q2 = cube[pcounter]
        coeff1,coeff2 = reverse_ld_coeffs(ld_law,q1,q2)
        params.u = [coeff1,coeff2]
        pcounter = pcounter + 1
else:
        params.u = [q1]

print 'p:',p,np.sqrt(cube_var[2])  
print 'jitter:',np.exp(ljitter)*1e6
max_var = cube[pcounter]
pcounter = pcounter + 1
alphas = np.zeros(X.shape[0])
for i in range(X.shape[0]):
        alphas[i] = cube[pcounter]
        pcounter = pcounter + 1
gp_vector = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas))#np.append(ljitter,1./alphas)

params.t0 = t0
params.per = P 
params.rp = p 
params.a = aR
params.inc = inc 
params.ecc = ecc 
params.w = omega
model = m.light_curve(params) 

one_array = np.ones(len(t))
ferr = one_array*np.sqrt(np.exp(ljitter))
gp.set_parameter_vector(gp_vector)

# Get prediction from GP:
#x = np.linspace(np.min(t)-0.1, np.max(t)+0.1, 5000)
pred_mean, pred_var = gp.predict(f - mflux - model, X.T, return_var=True)
pred_std = np.sqrt(pred_var)

# Plot:
sns.set_context("talk")
sns.set_style("ticks")
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['axes.linewidth'] = 1.2 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.markeredgewidth'] = 1 
fig = plt.figure(figsize=(10,4.5))
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

tzero = int(t[0])
# Plot MAP solution:
ax = plt.subplot(gs[0])
color = "cornflowerblue"
plt.plot(t-tzero, f,".k",markersize=1,label='K2 data')
plt.plot(t-tzero, pred_mean + mflux + model, linewidth=1, color='red',label='GP',alpha=0.5)
#plt.np.min(t-tzero)-0.1,np.max(t-tzero)+0.1
#plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
#plt.plot([10.,1],[np.max(pred_mean+mflux),0.995],'black',linewidth=1,alpha=0.5)
#plt.plot([11.,22.],[np.max(pred_mean+mflux),0.995],'black',linewidth=1,alpha=0.5)
plt.ylabel('Relative flux')
plt.legend(loc='lower right')

# Get prediction from GP to get residuals:
ax = plt.subplot(gs[1])
pred_mean, pred_var = gp.predict((f-mflux-model), X.T, return_var=True)
#plt.errorbar(t, (f-theta[0]-pred_mean)*1e6, yerr=ferr*1e6, fmt=".k",label='K2 data',markersize=1,alpha=0.1)
plt.plot(t-tzero,(f-mflux - model - pred_mean)*1e6,'.k',markersize=1)
print 'rms:',np.sqrt(np.var((f-mflux-pred_mean)*1e6))
print 'med error:',np.median(ferr*1e6)
plt.xlabel('Time (BJD-'+str(tzero)+')')
plt.ylabel('Residuals')
#plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
plt.tight_layout()
plt.savefig(out_folder+'GP_fit_george.pdf')
