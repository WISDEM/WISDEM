import os
import sys
import glob
import subprocess
import urllib2
import tarfile
import shutil
import platform

'''
This script is intended to install all of the core WISDEM packages and utilities 
to alleviate the burden of installing each individually.

This script is used infrequently for a couple of different reasons:
- It is not always robust to the variabilities of different operating systems
- It downloads tarballs, or frozen snapshots of the WISDEM repository, which 
  makes it tough for the user to maintain updates 
'''

failures = []


def get_options():
    import os, os.path
    from optparse import OptionParser
    parser = OptionParser()    
    parser.add_option('-f', '--force', dest='force', help='reinstall', action='store_true', default=False)

    (options, args) = parser.parse_args()

    return options, args

def install_url(f, url, subdir=None, force=False):
    print 'install url: package=', f, 'url= ', url

    if os.path.exists(f):
        if force:
            shutil.rmtree(f)
        else:
            print 'Path exists: ', f
            return;

    response = urllib2.urlopen(url)
    thetarfile = tarfile.open(fileobj=response, mode='r|gz')
    thetarfile.extractall()
    
    dirname = glob.glob('*%s*' % (f))
    curdir = os.getcwd()
    dirname = dirname[0]
    os.rename(dirname, f)   # potential bug if something exists/not empty?
    if (subdir == None):
        os.chdir(f)        
    else:
        os.chdir(subdir)    
    res = 0
    try:
        if url.find('kima') >= 0:
            res = subprocess.call(['python', 'setup.py', 'install'])
        else:
            res = subprocess.call(['python','setup.py','develop'])
        print 'subprocess returned ', res
    except:
        print 'Package %s FAILED to install correctly' % f       
        failures.append(f)
    if res != 0:
        print 'Package %s FAILED to install correctly' % f
        if (dirname not in failures):
            failures.append(f)

    os.chdir(curdir)


options, args = get_options()

files = ['akima', 'AirfoilPreppy', 'CCBlade', 'pBEAM', 'pyFrame3DD', 'pyMAP', 'Turbine_CostsSE', 'CommonSE', 'OffshoreBOS', 'Turbine_CostsSE', 'Plant_FinanceSE', 'DriveSE', 'NREL_CSM', 'RotorSE', 'TowerSE', 'FloatingSE'] 

wis = 'http://github.com/WISDEM/'

# download and install all the necessary WISDEM packages
for f in files:
    url = '%s%s/tarball/master' % (wis, f)
    install_url(f, url,force=options.force)
    
# finally install WISDEM itself
os.chdir(rootdir)

# summary
print
print
print 'Attempted to install WISDEM and its sub-packages: ', files
print 'Failed to install: ', failures

        'https://github.com/OpenMDAO/pyoptsparse/tarball/master#egg=pyoptsparse',

