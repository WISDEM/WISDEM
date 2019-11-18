"""
csmPPI.py

Created by George Scott on 2012-08-01.
Modified by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""
import wisdem.nrelcsm as nrelcsm
import os
import sys
import re
 
class Escalator:
    ''' 
        cost escalator class - computes weighted sum of different PPI_tables 
    '''
    
    def __init__(self,x):
        ''' x is a list '''
        self.name = x[0]  # name
        self.tbls = x[1]  # list of table names
        self.wts  = x[2]  # list of weights in pct (0-100.0)
        
        sum = 0.0
        for i in range(len(self.wts)):
            self.wts[i] *= 0.01
            sum += self.wts[i]
        if (abs(sum-1.0) > 0.0001):
            print('Weights add up to {0:.5f}'.format(sum))
            
    def compute(self,ppitbls,sy,sm,ey,em):
        ''' 
            returns cost escalator between start_yr/start_mon and end_yr/end_mon 
            ppitbls is a dictionary of PPITbl objects, indexed by NAICS code
        '''
        sum = 0.0
        for i in range(len(self.tbls)):
            key = self.tbls[i].strip()
            '''if (key == 'GDP'):
                print('Skipping GDP')
                continue'''
            if (key not in ppitbls):
                print ('No PPI table %s'.format(key))
                continue           
            if (key == "326150P") and (self.name == "Advanced Blade material costs       "):
            		ce = ppitbls[key].getEsc(2002,sm,ey,em)
            else:
            		ce = ppitbls[key].getEsc(sy,sm,ey,em)
            sum += ce * self.wts[i]
        return sum

#------------------------------------------------

class PPITbl:
    ''' a PPITbl object represents a cost table '''
    
    def __init__(self,code="",name="PPITbl"):
        self.cost = [] # empty list
        self.years = []
        self.name = name
        self.code = code
        pass
        
    def add_row(self,year,cost_array):
        self.cost.append(cost_array)
        self.years.append(year) 
        return 1
        
    def getEsc(self,start_yr,start_mon,end_yr,end_mon,printFlag=0):
        ''' return cost escalator between two dates (mon==13 is annual value) '''
        start_row = start_yr-self.years[0]
        end_row   = end_yr-self.years[0]
        
        if (start_row < 0):
            print('\n*** Year start_yr ${0:.2f} before table start {1}\n'.format(start_yr,self.years[0]))
            return None
        if (end_row >= len(self.cost)):
            print('\n*** Year end_yr ${0:.2f} after table end {1}\n'.format(end_yr,self.years[-1]))
            return None            
        if (len(self.cost[start_row]) < start_mon):
            raise IndexError("Start_mon out of range")
        if (len(self.cost[int(end_row)]) < int(end_mon)):
            print('\n*** EM {0:2d} > LER {0:2d} in table {}'.format(end_mon, len(self.cost[end_row]), self.code))
            raise IndexError("End_mon out of range")
            
        try:
            '''print('SR {0:2d} ER {0:2d}'.format(start_row,end_row))
            print('LSR {0:2d} LER {0:2d}'.format(len(self.cost[start_row]),len(self.cost[end_row])))'''
            cost_start = self.cost[start_row][start_mon-1]
            cost_end   = self.cost[int(end_row)][int(end_mon)-1]
        except:
            print('Index out of range for table {} {}'.format(self.code, self.name))
            return None
        esc = cost_end / cost_start
        return esc

#--------------------------------------------------------------------------------------

class PPI:
    def __init__(self,ref_yr,ref_mon,curr_yr,curr_mon,debug=0):
        '''
        Initialize the PPI class for calculation of PPI indices given a referene year/month and current year/month.
        
        Parameters
        ----------
        ref_yr : int
          reference PPI year
        ref_mon : int
          reference PPI month
        curr_yr : int
          current PPI year
        curr_mon : int
          current PPI month   
        '''     
        
        #self.escData = [None] * 37
        self.escData = {}  # try a dictionary
        self.tblfile = os.path.join('static','PPI_Tables.txt')   #TODO: temporary solution - should update so it can locate it from dictionary etc
        self.ppitbls = {} # dictionary of PPITbl objects 
        self.yrs_gdp = []
        self.ppi_gdp = []
        
        self.ref_yr   = ref_yr
        self.ref_mon  = ref_mon
        self.curr_yr  = curr_yr
        self.curr_mon = curr_mon
        self.debug = debug

        fullfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.tblfile)
        try:
            infile = open(fullfile, 'r') #infile = open(self.tblfile)
        except:
            sys.stdout.write ("Error opening or reading %s\n" % fullfile)
            quit()
        else:
            if (self.debug > 0):
                sys.stdout.write ("Opened %s\n" % self.tblfile)
            
        itable = -1
        found_tables = False
        found_GDP = False
        iCode   = {}
        code = ''
    
        for line in infile:
            words = line.split("\t")
            if not words:
                continue
                
            if (words[0].startswith("Gross Domestic Product")):
                found_GDP = True
                continue
            if (found_GDP and words[0].startswith("Year")):
                y = []
                for i in range(1,len(words)):
                    if (words[i].startswith("20")):
                        y.append(int(words[i]))
                self.yrs_gdp = y
                continue
            if (found_GDP and words[0].startswith("Absolute Value")):
                g = []
                num_re = re.compile(r"[\d\.]+")
                for i in range(1,len(words)):
                    if (num_re.search(words[i])):
                        g.append(float(words[i]))
                self.ppi_gdp = g
                
                code = 'GDP'
                self.ppitbls[code] = PPITbl(code=code, name="Gross Domestic Product")  # add a new element to self.ppitbls
                for i in range(len(y)):
                    rvals = []
                    for mon in range(13):
                        rvals.append(g[i]) # fill all monthly values with annual value
                    self.ppitbls[code].add_row(y[i], rvals)
                ippi = len(self.ppitbls)-1
                iCode[code] = ippi  # index of tables by code
                if (self.debug > 0): 
                    print('Created {0:2d} {} {}'.format(ippi, code, self.ppitbls[code].name))
                continue
                
            if (words[0].startswith("NAICS")):
                found_tables = True
                itable += 1
                iyr = 0
                
                words[2] = words[2].replace(r'"', '')  # strip quotes from name
                code = words[1]
                self.ppitbls[code] = PPITbl(code=code, name=words[2])  # add a new element to self.ppitbls
                ippi = len(self.ppitbls)-1
                if (self.debug > 0): 
                    print('Created {0:2d} {} {}'.format(ippi, code, self.ppitbls[code].name))
                iCode[code] = ippi  # index of tables by code 
                
            if (found_tables and words[0].startswith("20")): # a year number
                rvals = []
                i = 1
                while (re.match(r"\d+\.",words[i])):
                    rvals.append(float(words[i]))
                    i += 1
                self.ppitbls[code].add_row(int(words[0]), rvals)
                iyr += 1
        
        self.escData['IPPI_BLD'] = Escalator( ['Baseline Blade material costs       ',   ['3272123', '3255204', '332722489', '326150P'], [ 60.00,  23.00,  8.00,   9.00 ]  ] )
        self.escData['IPPI_BLA'] = Escalator( ['Advanced Blade material costs       ',   ['3272123', '3255204', '332722489', '326150P'], [ 61.00,  27.00,  3.00,   9.00 ]  ] )
        self.escData['IPPI_BLL'] = Escalator( ['Blade Labor costs                   ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_HUB'] = Escalator( ['Hub                                 ',   ['3315113',                                  ], [ 100.00                       ]  ] )
        self.escData['IPPI_PMB'] = Escalator( ['Pitch Mechanisms/Bearings           ',   ['332991P', '3353123', '333612P  ', '334513' ], [ 50.00,  20.00, 20.00,  10.00 ]  ] )
        self.escData['IPPI_LSS'] = Escalator( ['Low speed shaft                     ',   ['3315131'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_BRN'] = Escalator( ['Bearings                            ',   ['332991P'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_GRB'] = Escalator( ['Gearbox                             ',   ['333612P'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_BRK'] = Escalator( ['Mech brake, HS cpling etc           ',   ['3363401'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_GEN'] = Escalator( ['Generator                           ',   ['335312P'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_VSE'] = Escalator( ['Variable spd electronics            ',   ['335314P'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_YAW'] = Escalator( ['Yaw drive & bearing                 ',   ['3353123', '332991P',                       ], [ 50.00, 50.00                 ]  ] )
        self.escData['IPPI_MFM'] = Escalator( ['Main frame                          ',   ['3315113'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_ELC'] = Escalator( ['Electrical connections              ',   ['335313P', '3359291', 'GDP      ',          ], [ 25.00, 60.00, 15.00          ]  ] )
        self.escData['IPPI_HYD'] = Escalator( ['Hydraulic system                    ',   ['3339954'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_NAC'] = Escalator( ['Nacelle                             ',   ['3272123', '3255204', 'GDP      ',          ], [ 55.00, 30.00, 15.00          ]  ] )
        self.escData['IPPI_CTL'] = Escalator( ['Control, safety system              ',   ['334513 '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_MAR'] = Escalator( ['Marinization                        ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_WAR'] = Escalator( ['Offshore Warranty Premium           ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_TWR'] = Escalator( ['Tower                               ',   ['331221 '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_MPF'] = Escalator( ['Monopole Foundations                ',   ['BHVY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_TPT'] = Escalator( ['Transportation On/Offshore          ',   ['4841212'                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_STP'] = Escalator( ['Off Shore Site Prep                 ',   ['BHVY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_LAI'] = Escalator( ['Land Based Assembly & installation  ',   ['BHVY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_OAI'] = Escalator( ['Offshore Assembly & installation    ',   ['BHVY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_LEL'] = Escalator( ['Land Based Elect                    ',   ['3353119', '335313P', '3359291  ', 'GDP'    ], [ 40.00, 15.00, 35.00, 10.00   ]  ] )
        self.escData['IPPI_OEL'] = Escalator( ['Offshore Elect                      ',   ['3353119', '335313P', '3359291  ', 'GDP'    ], [  5.00,  5.00, 70.00, 20.00   ]  ] )
        self.escData['IPPI_LPM'] = Escalator( ['Permits, engineering (Land Based)   ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_OPM'] = Escalator( ['Permits, engineering (Offshore)     ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_LLR'] = Escalator( ['Land Based Levelized Replacement    ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_OLR'] = Escalator( ['Offshore Levelized Replacement      ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_LOM'] = Escalator( ['O&M Land Based                      ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_OOM'] = Escalator( ['O&M Offshore                        ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_LSE'] = Escalator( ['Land Based & Offshore Lease Cost    ',   ['GDP    '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_FND'] = Escalator( ['Foundations                         ',   ['BHVY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_RDC'] = Escalator( ['Road & Civil Work                   ',   ['BHWY   '                                   ], [ 100.00                       ]  ] )
        self.escData['IPPI_PAE'] = Escalator( ['Personnel Access Equipment          ',   ['GDP    '                                   ], [ 100.00                       ]  ] )

    def compute(self,escCode,debug=0):
        """
        Returns the cost escalator for escData object 'escCode', using reference and current yr/mon values.
        
        Parameters
        ----------
        escCode : object
          codes for different PPI data   
        
        Returns
        -------
        esc : float
          cost escalator for a particular PPI from reference year/month to current year month
        """ 
        # returns the cost escalator for escData object 'escCode', using reference and current yr/mon values
        
        if (escCode not in self.escData):
            print('Warning - no such code in PPI {0:.2f}'.format(escCode))
            #return 0
        esc = self.escData[escCode].compute(self.ppitbls,self.ref_yr,self.ref_mon,self.curr_yr,self.curr_mon)
        if (debug > 0):
            print('Escalator {} from {}{:02} to {}{:02} = {:.4}'.format(escCode,self.ref_yr,self.ref_mon,self.curr_yr,self.curr_mon,esc))
        return esc
        
#--------------------------------------------------------------------------------------

def example():
    
    # test cases
    
    sy = 2002; sm = 9;
    ey = 2010; em = 13;
    ppi = PPI(sy,sm,ey,em)
         
    for i in list(ppi.escData.keys()):
        print(ppi.escData[i].name)    
        # Use ppi.compute(code) to compute ppi for default interval
        print('{0:2d} {:.4} '.format(i, ppi.compute(i)))     
        print(' ' '')

if __name__ == "__main__":

    example()                                                                                           
