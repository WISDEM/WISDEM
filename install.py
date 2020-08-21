import subprocess
import sys


devel = "-e"


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", devel, "."], cwd=package)
    
install(".")  # Install WEIS
install("wisdem")  # Install WISDEM