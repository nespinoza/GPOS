import numpy as np
import itertools
import os

target = 'times_FWHM'
data = np.genfromtxt('eparams.dat',unpack=True)
columns = range(len(data))
eparamstouse = []
for i in range(1, len(columns)+1):
    for j in itertools.combinations(columns, i):
        eparamstouse.append(','.join(np.array(j).astype('str')))
print eparamstouse
sys.exit()

for eparamtouse in eparamstouse:
    if eparamtouse == 'all':
        extra_sufix = ''
    else:
        extra_sufix = '_'.join(eparamtouse.split(','))
    outfolder = target+'_'+extra_sufix
    for i in range(100):
        if i == 0:
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
        if not os.path.exists(outfolder+'/posteriors_trend_george_'+str(i)+'.pkl'):
            os.system('python fitGP.py -lcfile data_generator/'+target+'/dataset_'+str(i)+'.dat -ofolder '+outfolder+'/ -eparamfile eparams.dat -eparamtouse '+eparamtouse+' -pmean 0.1 -psd 0.1')
            os.system('rm '+outfolder+'/out* ')
            os.system('mv '+outfolder+'/posteriors_trend_george.pkl '+outfolder+'/posteriors_trend_george_'+str(i)+'.pkl')
            os.system('mv '+outfolder+'/GP_fit_george.pdf '+outfolder+'/GP_fit_george_'+str(i)+'.pdf')
