import numpy as np
import os
# import pandas
from collections import OrderedDict
import weis.aeroelasticse
def turb_specs(V_ref, L_u, L_v, L_w, sigma_u, sigma_v, sigma_w, template_file, filename):
    
    f=(np.array([np.arange(0.0015873015873015873015873015873, 20.00001, 0.0015873015873015873015873015873)])).T
    
    a=int(len(f))
    
    U=np.zeros((a,1),dtype=float)
    V=np.zeros((a,1),dtype=float)
    W=np.zeros((a,1),dtype=float)
    
    for i in range(0,a):
        U[i,0]= (4*L_u/V_ref)*sigma_u**2/((1+6*f[i,0]*L_u/V_ref)**(5./3.))
        V[i,0]= (4*L_v/V_ref)*sigma_v**2/((1+6*f[i,0]*L_v/V_ref)**(5./3.))
        W[i,0]= (4*L_w/V_ref)*sigma_w**2/((1+6*f[i,0]*L_w/V_ref)**(5./3.))
        
    df=pandas.DataFrame({'Frequency (Hz)':f[:,0],'u-component PSD (m^2/s)': U[:,0],'v-component PSD (m^2/s)': V[:,0],'w-component PSD (m^2/s)':W[:,0]})
    with open(template_file, 'r') as f:
        get_all=f.readlines() 
    
    
        
    with open(filename,'w') as f:
        for i,line in enumerate(get_all):
                if i < 11:  ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)
                        f.writelines(line)                             ## OVERWRITES line:2
                else: 
                        f.write(df.to_string(index=False,header=False,col_space=15))
                        break
                
                                                                                                               
    

