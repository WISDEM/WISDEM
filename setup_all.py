import os,sys,glob,subprocess, urllib2, tarfile, shutil, platform


failures = []


def get_options():
    import os, os.path
    from optparse import OptionParser
    parser = OptionParser()    
    parser.add_option("-f", "--force", dest="force", help="reinstall even existing plugins", action="store_true", default=False)

    (options, args) = parser.parse_args()

    return options, args

def install_url(f, url, subdir=None, plugin=True, force=False):
    print "install url: package=", f, "url= ", url

    if os.path.exists(f):
        if force:
            shutil.rmtree(f)
        else:
            print "Path exists: ", f
            return;

    response = urllib2.urlopen(url)
    thetarfile = tarfile.open(fileobj=response, mode="r|gz")
    thetarfile.extractall()
    
    dirname = glob.glob("*%s*" % (f))
    curdir = os.getcwd()
    dirname = dirname[0]
    os.rename(dirname, f)   # potential bug if something exists/not empty?
    if (subdir == None):
        os.chdir(f)        
    else:
        os.chdir(subdir)    
    res = 0
    try:
        if (plugin):
            res = subprocess.call(["plugin", "install"])
        else:
            if platform.system() == 'Windows':
                res = subprocess.call(["python", "setup.py", "config", "--compiler=mingw32", "build", "--compiler=mingw32", "install"])
            else:
                res = subprocess.call(["python","setup.py","install"])
        print "subprocess returned ", res
    except:
        print "plugin %s FAILED to install correctly" % f       
        failures.append(f)
    if res != 0:
        print "plugin %s FAILED to install correctly" % f
        if (dirname not in failures):
            failures.append(f)

    os.chdir(curdir)


options, args = get_options()

#files = ["Turbine_CostsSE", "CommonSE", "Plant_CostsSE", "Plant_FinanceSE", "Plant_EnergySE"]
files = ["Turbine_CostsSE", "CommonSE", "Plant_CostsSE", "Plant_FinanceSE", "Plant_EnergySE",
         "AeroelasticSE", "AirfoilPreppy", "CCBlade", "DriveSE", "DriveWPACT", "NREL_CSM", "RotorSE",
         "TowerSE", "pyFrame3DD", "JacketSE", "akima", "pBEAM"] 

#files = ["pBEAM"]

wis = "http://github.com/WISDEM/"
subdir = "plugins"

# install pandas and algopy
subprocess.call(["easy_install", "pandas"])
subprocess.call(["easy_install", "algopy"])
subprocess.call(["easy_install", "zope.interface"])
subprocess.call(["easy_install", "sphinx"])
subprocess.call(["easy_install", "xlrd"])
subprocess.call(["easy_install", "pyopt"])
if platform.system() == 'Windows': 
    subprocess.call(["easy_install", "py2exe"])
subprocess.call(["easy_install", "pyzmq"])
subprocess.call(["easy_install", "sphinxcontrib-bibtex"])
subprocess.call(["easy_install", "sphinxcontrib-napoleon"])
#subprocess.call(["easy_install", "sphinxcontrib-zopeext"])
subprocess.call(["easy_install", "numpydoc"])
subprocess.call(["easy_install", "ipython"])
subprocess.call(["easy_install", "python-dateutil"])

# make plug in dir and cd to it:
rootdir = os.getcwd()
if not os.path.exists(subdir):
    os.mkdir(subdir)
os.chdir(subdir)

# install fused wind
f = "fusedwind"
#subdir = os.path.join(f,f) # fusedwind is nested!... not anymore, I guess
subdir = f
url = "http://github.com/FUSED-Wind/fusedwind/tarball/develop"  ## note, develop branch
install_url(f,url,force=options.force)

# download and install all the necessary WISDEM plugins
for f in files:
    url = "%s%s/tarball/0.1" % (wis, f)
    install_url(f, url,force=options.force)
    
# finally install WISDEM itself
os.chdir(rootdir)
os.system("plugin install")

# summary
print
print
print "Attempted to install WISDEM and its sub-plugins: ", files
print "Failed to install: ", failures
