import os,sys,glob,subprocess, urllib2, tarfile, shutil


failures = []

def install_url(f, url, subdir=None, plugin=True):
    print "install url: package=", f, "url= ", url
    response = urllib2.urlopen(url)
    thetarfile = tarfile.open(fileobj=response, mode="r|gz")
    thetarfile.extractall()

    if os.path.exists(f):
        shutil.rmtree(f)

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
            res = subprocess.call(["python", "setup.py", "install"])
        print "subprocess returned ", res
    except:
        print "plugin %s FAILED to install correctly" % f       
        failures.append(f)
    if res != 0:
        print "plugin %s FAILED to install correctly" % f
        if (dirname not in failures):
            failures.append(f)

    os.chdir(curdir)


#files = ["Turbine_CostsSE", "CommonSE", "Plant_CostsSE", "Plant_FinanceSE", "Plant_EnergySE"]
files = ["Turbine_CostsSE", "CommonSE", "Plant_CostsSE", "Plant_FinanceSE", "Plant_EnergySE",
         "AeroelasticSE", "AirfoilPreppy", "CCBlade", "DriveSE", "DriveWPACT", "NREL_CSM", "RotorSE",
         "TowerSE", "TurbineSE", "pyFrame3DD", "pBEAM"]

#files = ["pBEAM"]

wis = "http://github.com/WISDEM/"
subdir = "plugins"

# install pandas and algopy
subprocess.call(["easy_install", "pandas"])
subprocess.call(["easy_install", "algopy"])

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
install_url(f,url)

#install Andrew Ning's akima
url = "http://github.com/andrewning/akima/tarball/master"
install_url("akima", url, plugin=False)

# download and install all the necessary WISDEM plugins
for f in files:
    url = "%s%s/tarball/master" % (wis, f)
    install_url(f, url)
    
# finally install WISDEM itself
os.chdir(rootdir)
os.system("plugin install")

# summary
print
print
print "Attempted to install WISDEM and its sub-plugins: ", files
print "Failed to install: ", failures
