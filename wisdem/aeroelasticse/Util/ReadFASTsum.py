from wisdem.aeroelasticse.Util.FileTools import str_repeats

def read_ElastoDynSum(elastodyn_sum):

    ed_sum = {}
    f = open(elastodyn_sum)

    # Get runtime
    f.readline()
    ln = f.readline().strip().split()
    ed_sum['Run_Date'] = ln[-3]
    ed_sum['Run_Time'] = ln[-1]

    # Get DOF
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['PtfmSgDOF']   = f.readline().strip().split()[0]
    ed_sum['PtfmSwDOF']   = f.readline().strip().split()[0]
    ed_sum['PtfmHvDOF']   = f.readline().strip().split()[0]
    ed_sum['PtfmRDOF']    = f.readline().strip().split()[0]
    ed_sum['PtfmPDOF']    = f.readline().strip().split()[0]
    ed_sum['PtfmYDOF']    = f.readline().strip().split()[0]
    ed_sum['TwFADOF1']    = f.readline().strip().split()[0]
    ed_sum['TwSSDOF1']    = f.readline().strip().split()[0]
    ed_sum['TwFADOF2']    = f.readline().strip().split()[0]
    ed_sum['TwSSDOF2']    = f.readline().strip().split()[0]
    ed_sum['YawDOF']      = f.readline().strip().split()[0]
    ed_sum['Furling']     = f.readline().strip().split()[0]
    ed_sum['GenDOF']      = f.readline().strip().split()[0]
    ed_sum['DrTrDOF']     = f.readline().strip().split()[0]
    ed_sum['TeetDOF']     = f.readline().strip().split()[0]
    ed_sum['FlapDOF1_B1'] = f.readline().strip().split()[0]
    ed_sum['FlapDOF2_B1'] = f.readline().strip().split()[0]
    ed_sum['EdgeDOF_B1']  = f.readline().strip().split()[0]
    ed_sum['FlapDOF1_B2'] = f.readline().strip().split()[0]
    ed_sum['FlapDOF2_B2'] = f.readline().strip().split()[0]
    ed_sum['EdgeDOF_B2']  = f.readline().strip().split()[0]
    ed_sum['FlapDOF1_B3'] = f.readline().strip().split()[0]
    ed_sum['FlapDOF2_B3'] = f.readline().strip().split()[0]
    ed_sum['EdgeDOF_B3']  = f.readline().strip().split()[0]
    for var in list(ed_sum.keys()):
        if ed_sum[var] == 'Enabled':
            ed_sum[var] = True
        else:
            ed_sum[var] = False
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['dt'] = float(f.readline().strip().split()[-1])

    # Structural Properties
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['HubHeight']    = float(f.readline().strip().split()[-1])
    ed_sum['TowerLength']  = float(f.readline().strip().split()[-1])
    ed_sum['BladeLength']  = float(f.readline().strip().split()[-1])
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['RotorMass']    = float(f.readline().strip().split(')')[-1].strip())
    ed_sum['RotorInertia'] = float(f.readline().strip().split(')')[-1].strip())
    f.readline()
    f.readline()
    ln = f.readline().strip().split(')')[-1].strip().split()
    ed_sum['Blade1Mass']        = float(ln[0])
    ed_sum['Blade2Mass']        = float(ln[1])
    ed_sum['Blade3Mass']        = float(ln[2])
    ln = f.readline().strip().split(')')[-1].strip().split()
    if len(ln) == 3:
        ed_sum['Blade1MassMoment1'] = float(ln[0])
        ed_sum['Blade2MassMoment1'] = float(ln[1])
        ed_sum['Blade3MassMoment1'] = float(ln[2])
    else:
        # for long values, ED sometimes outputs with no spaces between the numbers, 
        # find repeated sequences in the long string of numbers
        massmoment1 = str_repeats(ln[0])
        ed_sum['Blade1MassMoment1'] = float(massmoment1)
        ed_sum['Blade2MassMoment1'] = float(massmoment1)
        ed_sum['Blade3MassMoment1'] = float(massmoment1)

    ln = f.readline().strip().split(')')[-1].strip().split()
    ed_sum['Blade1MassMoment2'] = float(ln[0])
    ed_sum['Blade2MassMoment2'] = float(ln[1])
    ed_sum['Blade3MassMoment2'] = float(ln[2])
    ln = f.readline().strip().split(')')[-1].strip().split()
    ed_sum['Blade1CenterMass']  = float(ln[0])
    ed_sum['Blade2CenterMass']  = float(ln[1])
    ed_sum['Blade3CenterMass']  = float(ln[2])
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['TowerTopMass']      = float(f.readline().strip().split()[-1])
    ed_sum['TowerMass']         = float(f.readline().strip().split()[-1])
    ed_sum['PlatformMass']      = float(f.readline().strip().split()[-1])
    ed_sum['MassWPlatform']     = float(f.readline().strip().split()[-1])
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['Tower'] = {}
    ed_sum['Tower']['Node']     = []
    ed_sum['Tower']['TwFract']  = []
    ed_sum['Tower']['HNodes']   = []
    ed_sum['Tower']['DHNodes']  = []
    ed_sum['Tower']['TMassDen'] = []
    ed_sum['Tower']['FAStiff']  = []
    ed_sum['Tower']['SSStiff']  = []
    ln = f.readline().strip().split()
    while len(ln) > 1:
        ed_sum['Tower']['Node'].append(int(ln[0]))
        ed_sum['Tower']['TwFract'].append(float(ln[1]))
        ed_sum['Tower']['HNodes'].append(float(ln[2]))
        ed_sum['Tower']['DHNodes'].append(float(ln[3]))
        ed_sum['Tower']['TMassDen'].append(float(ln[4]))
        ed_sum['Tower']['FAStiff'].append(float(ln[5]))
        ed_sum['Tower']['SSStiff'].append(float(ln[6]))
        ln = f.readline().strip().split()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['Blade1'] = {}
    ed_sum['Blade1']['Node']      = []
    ed_sum['Blade1']['BlFract']   = []
    ed_sum['Blade1']['RNodes']    = []
    ed_sum['Blade1']['DRNodes']   = []
    ed_sum['Blade1']['PitchAxis'] = []
    ed_sum['Blade1']['StrcTwst']  = []
    ed_sum['Blade1']['BMassDen']  = []
    ed_sum['Blade1']['FlpStff']   = []
    ed_sum['Blade1']['EdgStff']   = []
    ln = f.readline().strip().split()
    while len(ln) > 1:
        ed_sum['Blade1']['Node'].append(int(ln[0]))
        ed_sum['Blade1']['BlFract'].append(float(ln[1]))
        ed_sum['Blade1']['RNodes'].append(float(ln[2]))
        ed_sum['Blade1']['DRNodes'].append(float(ln[3]))
        ed_sum['Blade1']['PitchAxis'].append(float(ln[4]))
        ed_sum['Blade1']['StrcTwst'].append(float(ln[5]))
        ed_sum['Blade1']['BMassDen'].append(float(ln[6]))
        ed_sum['Blade1']['FlpStff'].append(float(ln[7]))
        ed_sum['Blade1']['EdgStff'].append(float(ln[8]))
        ln = f.readline().strip().split()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['Blade2'] = {}
    ed_sum['Blade2']['Node']      = []
    ed_sum['Blade2']['BlFract']   = []
    ed_sum['Blade2']['RNodes']    = []
    ed_sum['Blade2']['DRNodes']   = []
    ed_sum['Blade2']['PitchAxis'] = []
    ed_sum['Blade2']['StrcTwst']  = []
    ed_sum['Blade2']['BMassDen']  = []
    ed_sum['Blade2']['FlpStff']   = []
    ed_sum['Blade2']['EdgStff']   = []
    ln = f.readline().strip().split()
    while len(ln) > 1:
        ed_sum['Blade2']['Node'].append(int(ln[0]))
        ed_sum['Blade2']['BlFract'].append(float(ln[1]))
        ed_sum['Blade2']['RNodes'].append(float(ln[2]))
        ed_sum['Blade2']['DRNodes'].append(float(ln[3]))
        ed_sum['Blade2']['PitchAxis'].append(float(ln[4]))
        ed_sum['Blade2']['StrcTwst'].append(float(ln[5]))
        ed_sum['Blade2']['BMassDen'].append(float(ln[6]))
        ed_sum['Blade2']['FlpStff'].append(float(ln[7]))
        ed_sum['Blade2']['EdgStff'].append(float(ln[8]))
        ln = f.readline().strip().split()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    ed_sum['Blade3'] = {}
    ed_sum['Blade3']['Node']      = []
    ed_sum['Blade3']['BlFract']   = []
    ed_sum['Blade3']['RNodes']    = []
    ed_sum['Blade3']['DRNodes']   = []
    ed_sum['Blade3']['PitchAxis'] = []
    ed_sum['Blade3']['StrcTwst']  = []
    ed_sum['Blade3']['BMassDen']  = []
    ed_sum['Blade3']['FlpStff']   = []
    ed_sum['Blade3']['EdgStff']   = []
    ln = f.readline().strip().split()
    while len(ln) > 1:
        ed_sum['Blade3']['Node'].append(int(ln[0]))
        ed_sum['Blade3']['BlFract'].append(float(ln[1]))
        ed_sum['Blade3']['RNodes'].append(float(ln[2]))
        ed_sum['Blade3']['DRNodes'].append(float(ln[3]))
        ed_sum['Blade3']['PitchAxis'].append(float(ln[4]))
        ed_sum['Blade3']['StrcTwst'].append(float(ln[5]))
        ed_sum['Blade3']['BMassDen'].append(float(ln[6]))
        ed_sum['Blade3']['FlpStff'].append(float(ln[7]))
        ed_sum['Blade3']['EdgStff'].append(float(ln[8]))
        ln = f.readline().strip().split()

    return ed_sum

if __name__=="__main__":
    elastodyn_sum = "/mnt/c/Users/egaertne/Projects/BAR_Fatigue/run_dir2/BAR_00_FatigueTesting_00.ED.sum"
    ed_sum        = read_ElastoDynSum(elastodyn_sum)
