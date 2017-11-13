import numpy as np

def Group_Element(data):
    no_data = 4
    out_put = []
    for i in range(len(data[0]) / no_data):
        out_put.append(data[:,no_data*i:no_data*(i+1)])
    return out_put

#Get an array of cumulative displacements
def Cumulative_Displacement(Us):
    import numpy as np
    Ua = np.array([0.])
    for i in range(len(Us)):
        Ua = np.append(Ua, Ua[-1] + abs(Us[i] - Us[i-1])) if i !=0 else np.append(Ua, abs(Us[i]))
    return Ua

#Get an array of VGI cyclic
def VGI_Cyclic(Ele):
    import numpy as np
    #Seperate the input data
    PEEQ, Se, Sm, E = [Ele[:,i] for i in range(np.shape(Ele)[1])]
    VGIc = np.array([0.])
    Ts = np.array([0.])
    #Start Calculate
    for i in range(len(PEEQ)):
        de = E[i] - E[i-1] if i!=0 else E[i]
        dep = PEEQ[i]  - PEEQ[i-1] if i!=0 else PEEQ[i] 
        sm = Sm[i]
        se = Se[i]
        T = sm / se
        Ts = np.append(Ts, T)
        if de >= 0.:
            VGIc = np.append(VGIc, VGIc[-1] + np.exp(abs(T)) * dep)
        else:
            VGIc = np.append(VGIc, VGIc[-1] - np.exp(abs(T)) * dep)
        if VGIc[-1] < 0.:
            VGIc[-1] = 0
    return VGIc, Ts
    
#Get an array of critical VGI cyclic
def VGI_Critical(Ele, VGIm, lamuda):
    import numpy as np
    #Seperate the input data
    PEEQ, Se, Sm, E = [Ele[:,i] for i in range(np.shape(Ele)[1])]
    epa = 0.
    pre_de = 1.
    VGI_critical = np.array([VGIm * np.exp(-1. * lamuda * epa)])
    VGI_critical = np.append(VGI_critical, VGI_critical)
    for i in range(1, len(PEEQ)):
        de = E[i] - E[i-1]
        dep = PEEQ[i]  - PEEQ[i-1]
        if de < 0.:
            epa = epa + abs(dep)
        if de >= 0. and pre_de <= 0.:
            VGI_critical  = np.append(VGI_critical , VGIm * np.exp(-1. * lamuda * epa))
        else:
            VGI_critical  = np.append(VGI_critical , VGI_critical [-1])
        pre_de = de
    return VGI_critical

#Find the crack point
def Check_Crack(VGIc, VGI_critical, Ua):
    import numpy as np
    is_crack = False
    for i in range(len(VGIc)):
        if VGIc[i] >=  VGI_critical[i]:
            is_crack = True
            break
    if is_crack:
        return (Ua[i], VGIc[i])
    else:
        return (0.0, 0.0)

#CVGM of one case
def CVGM_one(Us, Ele, VGIm, lamuda):
    Ua = Cumulative_Displacement(Us)
    VGIc, Ts = VGI_Cyclic(Ele)
    VGI_critical = VGI_Critical(Ele, VGIm, lamuda)
    Crack_point = Check_Crack(VGIc, VGI_critical, Ua)
    return Ua, VGIc, VGI_critical, Crack_point, Ts

#Get VGI critical of all elements
def VGI_Critical_Multi(Eles, VGIm, lamuda):
    import numpy as np
    for i in range(len(Eles)):
        VGI_critical = VGI_Critical(Eles[i], VGIm, lamuda)
        VGI_critical_multi = np.hstack((VGI_critical_multi, np.array([VGI_critical]).transpose())) if i!=0 else np.array([VGI_critical]).transpose()
    return VGI_critical_multi
    
#Get VGI cyclic of each elements
def VGI_Cyclic_Multi(Eles):
    import numpy as np
    for i in range(len(Eles)):
        VGIc, Ts  = VGI_Cyclic(Eles[i])
        VGIc_multi = np.hstack((VGIc_multi, np.array([VGIc]).transpose())) if i!=0 else np.array([VGIc]).transpose()
    return VGIc_multi
    
#Check crack of multi elements
def Check_Crack_multi(VGIc_multi, VGI_critical_multi, Ua):
    #!!VGI_critical is changing
    import numpy as np
    Crack_disps = np.array([])
    Crack_points = np.array([0.0,0.0])
    for i in range(np.shape(VGIc_multi)[1]):
        VGIc = VGIc_multi[:,i]
        VGI_critical = VGI_critical_multi[:,i]
        Crack_point = Check_Crack(VGIc, VGI_critical, Ua)
        Crack_points = np.vstack((Crack_points, Crack_point))
    Crack_points = Crack_points[1:,:]
    Crack_disps = Crack_points[:,0]
    if max(Crack_disps) == 0.0:
        return (0.0, 0.0), -1
    else:
        temp = np.min(Crack_disps[np.nonzero(Crack_disps)])
        Crack_i = np.where(Crack_disps==temp)[0][0]
        return Crack_points[Crack_i], Crack_i

#CVGM of multi elements
def CVGM(Us, Eles, VGIm, lamuda):
    import numpy as np
    Ua = Cumulative_Displacement(Us)
    VGI_critical_multi = VGI_Critical_Multi(Eles, VGIm, lamuda)
    VGIc_multi = VGI_Cyclic_Multi(Eles)
    Crack_point, Crack_i = Check_Crack_multi(VGIc_multi, VGI_critical_multi, Ua)
    if Crack_i == -1:
        return -1, -1, -1, -1, (0.0, 0.0)
    else:
        return Crack_i, Ua, VGIc_multi[:,Crack_i], VGI_critical_multi[:,Crack_i], Crack_point

    

def Principal_stresses(data, dimension = 3):
    import numpy as np
    if  dimension == 3:
        S11, S22, S33, S12, S13, S23 = [data[:,i] for i in range(np.shape(data)[1])]
        I1 = S11 + S22 + S33
        I2 = S11 * S22 + S22 * S33 + S33 * S11 - S12**2 - S23**2 - S13**2
        I3 = S11 * S22 * S33 - S11 * S23**2 - S22 * S13**2 - S33 * S12**2 + 2.0 * S12 * S23 * S13
        theta = np.arccos(np.divide((2. * I1**3 - 9. * I1 * I2 + 27. * I3), (2. * (I1**2 - 3. * I2)**1.5))) / 3.0
        S1 = I1 / 3. + 2. / 3. * (np.sqrt(I1**2 - 3. * I2)) * np.cos(theta)
        S2 = I1 / 3. + 2. / 3. * (np.sqrt(I1**2 - 3. * I2)) * np.cos(theta - 2. * np.pi / 3.)
        S3 = I1 / 3. + 2. / 3. * (np.sqrt(I1**2 - 3. * I2)) * np.cos(theta - 4. * np.pi / 3.)
        return S1, S2, S3
    elif dimension == 2:
        S11, S22, S12 = [data[:,i] for i in range(np.shape(data)[1])]
        S1 = (S11 + S22) / 2.0 + np.sqrt(((S11 - S22) / 2.0)**2 + S12**2)
        S2 = (S11 + S22) / 2.0 - np.sqrt(((S11 - S22) / 2.0)**2 + S12**2)
        return S1, S2