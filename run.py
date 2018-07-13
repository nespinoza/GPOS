import os

target = 'times_Deltas_FWHM_Z_g_trace'
outfolder = target
for i in range(100):
    if i == 0:
        os.mkdir(outfolder)
    os.system('python fitGP.py -lcfile data_generator/'+target+'/dataset_'+str(i)+'.dat -ofolder '+outfolder+'/ -eparamfile eparams.dat -pmean 0.1 -psd 0.1')
    os.system('rm '+outfolder+'/out* ')
    os.system('mv '+outfolder+'/posteriors_trend_george.pkl '+outfolder+'/posteriors_trend_george_'+str(i)+'.pkl')
    os.system('mv '+outfolder+'/GP_fit_george.pdf '+outfolder+'/GP_fit_george_'+str(i)+'.pdf')
