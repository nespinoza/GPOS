import pickle
import glob
from pylab import *

ps = glob.glob('*.pkl')
all_p = np.zeros(len(ps))
all_p_err = np.zeros(len(ps))
counter = 0
for p in ps:
    posterior = pickle.load(open(p,'r'))
    hist(posterior['posterior_samples'][:,2],bins=100,normed=True,alpha=0.5)
    all_p[counter] = np.mean(posterior['posterior_samples'][:,2]**2)
    all_p_err[counter] = np.sqrt(np.var(posterior['posterior_samples'][:,2]**2))
    counter = counter + 1
show()

plt.errorbar(np.arange(len(ps)),(all_p-0.1**2)*1e6,yerr=all_p_err*1e6,fmt='.')
show()
