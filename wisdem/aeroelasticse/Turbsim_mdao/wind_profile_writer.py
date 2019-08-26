import numpy as np
import os
import decimal
# import pandas
from collections import OrderedDict

def write_wind(V_ref, alpha, Beta, Z_hub, filename, template_file):

    Height=(np.array([np.arange(0,181,10)],dtype=float))
    
    new_Height=(Height/Z_hub).T
    
    Height=(np.array([np.arange(0,181,10)])).T
    a=len(Height)
 
    
    U=np.zeros((a,1),dtype=float)
    Beta1=np.zeros((a,1),dtype=float)
    
    
    for i in range(0,a):
        U[i,0]= V_ref*(new_Height[i,0])**alpha
        Beta1[i,0]= (Beta/63)*(Height[i,0])-90*(Beta/63)
    
    
    df1= ['%.3f'% x for x in Height]
    df2 = ['%.3f'% x for x in U]
    df3 =['%.3f'% x for x in Beta1]
    

    with open(template_file,'r') as f:
        get_all=f.readlines() 
        
    with open(filename,'w') as f2:
        for i,line in enumerate(get_all,1):
            if i < 12:  
                f2.writelines(line)
            else:
                for p in range(len(df1)):
                    if len(str(df1[p]))<5 :
                        f2.write(str(df1[p]) + "            " + str(df2[p]) + "            " + str(df3[p]) + "\n")
                    else:
                        f2.write(str(df1[p]) + "           " + str(df2[p]) + "            " + str(df3[p]) + "\n")
                break
                f2.close()
                        


