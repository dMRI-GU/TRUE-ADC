import numpy as np
#download from https://github.com/dMRI-GU/OBSIDIAN
import algorithm as ra

#Example for calculating ADC values found in Table 1 and Table S1 for publication:

#Kuczera S, Langkilde F, Maier S.
#Truly reproducible uniform estimation of the ADC with multi‐b diffusion data—Application in prostate diffusion imaging.
#Magn Reson Med. 2022;1‐15. doi:10.1002/mrm.29533

#Here: New Method biexponential normal tissue

bias_corr=ra.RicianBiasCorr(debug=False)
SNR=20
#desired b-value sequence
bmax=2000
num_b=21
b=np.linspace(0,bmax,num_b)[1:]
#simulation parameters 
sim_para=[2.2,0.4,.8]
#number of repetitions
num_repeat=50000
#model function for simulation
model_func=bias_corr.func_dict['biexp']
in_para=sim_para+[SNR]
#sigma of simulation
sig=1
#genrate input signal with rician noise
sig_input=model_func(b,*in_para)
sig_input_all=np.tile(sig_input,num_repeat)
rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                      sig).reshape(num_repeat,len(b))
#first fit with biexponential function
para_sim={} #run parameters for first fit
para_sim['model_func_name']='biexp'
para_sim['b']=b
para_sim['guess']=(2.,.5,.5,SNR*.8)
para_sim['bound']=([0.,0.,0.1,0.],[4.,1.,.9,np.inf])
para_sim['sigma_av_corr']=2.3 #sigma correction for biexponential function
#running with bias=0 is like doing normal fit without bias correction
res=bias_corr.run_bias_corr(para_sim,rice_signal,bias=0,make_output=False)
res_fit=res[0][np.arange(num_repeat),res[2]]

#2-point ADC
print('2-point ADC')
#extract signal at b-value and calculate ADC
b_ADC=np.array([100,1000])
fit_ADC=model_func(b_ADC.reshape(-1,1),*res_fit.T).T
res_ADC=-np.log(fit_ADC[:,1]/fit_ADC[:,0])/(b_ADC[1]-b_ADC[0])*1e3
print('MEAN: {:.3f}'.format(np.mean(res_ADC)))
print('STD: {:.3f}'.format(np.std(res_ADC)))
print('SNR: {:.3f}'.format(np.mean(res_ADC)/np.std(res_ADC)))


#multi-point ADC (non-linear regression)
#extract signal at ADC b-values and fit with monoexponential
print('multi-point ADC')
b_ADC=np.array([100,1000,1500])
fit_ADC=model_func(b_ADC.reshape(-1,1),*res_fit.T).T
#monoexponential LSE
corr_dict={'model_func_name': 'monoexp',
           'b': b_ADC,
           'guess': [1,.7*SNR],
           'bound': ([0.,0.],[4.,np.inf]),
          }
res_ADC=bias_corr.run_bias_corr(corr_dict,fit_ADC,make_output=False)
res_fit_ADC=res_ADC[0][:,0,:-1]
print('MEAN: {:.3f}'.format(np.mean(res_fit_ADC,axis=0)[0]))
print('STD: {:.3f}'.format(np.std(res_fit_ADC,axis=0)[0]))
print('SNR: {:.3f}'.format(np.mean(res_fit_ADC,axis=0)[0]/
                           np.std(res_fit_ADC,axis=0)[0]))
