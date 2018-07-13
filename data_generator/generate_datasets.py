import matplotlib.pyplot as plt
import george
import itertools
import numpy as np
import batman
import os

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

inputs = np.genfromtxt('w19_parameters.dat',unpack=True)

# Standarize the inputs:
for i in range(len(inputs)):
    norm_input = (inputs[i] - np.mean(inputs[i]))/np.sqrt(np.var(inputs[i]))
    if i == 0:
        X = norm_input
        times = inputs[i]
    else:
        X = np.vstack((X,norm_input))

# Define base flux (white) noise:
sigma = 200*1e-6
yerr = np.ones(len(times))*sigma

# Define maximum variance (i.e., the total variance of the GP):
max_sigma = 2000.
max_var = (max_sigma*1e-6)**2

# Define number of simulations:
nsims = 300

# Define transit model:
t0 = times[len(times)/2]
P = 3.0
p = 0.1
aR = 10.
inc = 88.0
q1,q2 = 0.5,0.5
model = get_transit_model(times.astype('float64'),t0,P,p,aR,inc,q1,q2,'quadratic')

# Name of the variables:
names = ['times','Deltas','FWHM','Z','g','trace']
idx_names = range(len(names))

# Generate all possible combinations of external parameters, and generate datasets:
for L in range(0, len(idx_names)+1):
    for subset in itertools.combinations(idx_names, L):
        if len(subset) != 0:
            for n in range(nsims):
                if n == 0:
                    cnames = list( names[i] for i in subset)
                    fname = '_'.join(cnames)
                    os.mkdir(fname)
                fout = open(fname+'/dataset_'+str(n)+'.dat','w')
                # Generate nsims datasets per model:
                Xc = X[subset,:]
                # Generate gaussian process. For this, sample lambdas from uniform distribution:
                ndim = Xc.shape[0]
                lambdas = np.random.uniform(0,10,ndim)
                fout.write('# Lambdas: '+' '.join(lambdas.astype('str'))+' | Sigma: '+str(sigma*1e6)+' ppm | Max (GP) Sigma: '+str(max_sigma)+' ppm\n')
                fout.write('# Times \t Simulated data \t Transit Model \t GP\n')
                # Compute kernel:
                kernel = max_var*george.kernels.ExpSquaredKernel(lambdas,ndim=ndim,axes=range(ndim))
                # Prepare GP object:
                gp = george.GP(kernel)
                gp.compute(Xc.T)
                # Sample GP, add gaussian noise and save
                GP = gp.sample(Xc.T)
                noise = np.random.normal(0.,sigma,len(times))
                total = model + GP + noise
                for i in range(len(times)):
                    fout.write('{0:.10f} \t {1:.10f} \t {2:.10f} \t {3:.10f}\n'.format(times[i],total[i],model[i],GP[i]))
