""" ----------------
This tool was adapted from FusedWind-0.1.0 runSuite.  The interface is wrapped for better itegration with
AeroElasticSE, but the overall functionality is unchanged.
---------------- """

"""
This module allows for sampling of distributions encountered in wind energy research using a flexible 
text input file specification.  The distributions currently available are:

- Weibull
- Gamma
- vonMises
- uniform
- normal
- bivariate normal

In addition, we allow for "enumeration" of sets of variables, and their cartesian product.

The impetus is the desire for a generic representation of rows in Table 1 of IEC document 88_329
describing design standards for loads analysis, where what is being represented are the
distributions of wind/wave environment parameters.
We can assume these all boil down to distributions of _numerical_ (rather than discrete) parameters.
A twist is that some of the parameters depend on eachother, that is, the distributions are conditional distributions.

In principle all we need to say is  :math:`\{v_i\} \sim D_k (\{p_j(v)\})`
where :math:`\{v_i\}` are the random variables we're defining (including multivariate distributions),
:math:`D_k` are distributions, and :math:`\{p_j\}` are parameters to these distributions.
Currently :math:`p_j` can only depend on :math:`v_i` for :math:`i<j`, but otherwise
the dependence can be expressed as an arbitrary python math expression.

An example of the input this system can handle is::

    #--WW4Ddist--
    Vhub = W(2.120, 9.767)
    WaveDir = VM(0.029265 + 0.119759 * Vhub, -0.620138 + 0.030709 * Vhub)
    Hs = G(4.229440 + 0.222745 * WaveDir  + 0.328602 * Vhub, 0.145621 + -0.006188 * WaveDir  + 0.007561 * Vhub)
    Tp = G(19.677865 + 6.617868 * Hs  + -0.532952 * Vhub, 0.202077 + 0.286631 / Hs  + -0.004321 * Vhub)

which causes the code to sample the joint distribution of hub height wind speed (Weibull), wave direction (von Mises), 
wave height (Gamma), and wave period (Gamma) for a recent study.

Or::

    #--WW4Ddist--
    Vhub = {5,15,25}
    Hs = {1,5}
    
which causes the code to generate the cartesian product {(5,1),(5,5),(15,1),(15,5),(25,1),(25,5)}.

"""

### This is some more of the thinking that was part of development, so not included in the docstring:
#Some cases:
#x = 7
#x ~ unif({7})

#x ~ N(0,1)
#y = x
#y ~ delta(x)

#x = 7
#y = E[y|x] = \int{ p(y|x) dy }
#requires
#p(y|x), eg
#y|x ~ N(2x/7, 1)

#x,y "jointly distributed"
#x,y ~ N([0,0], [[sigmx, covxy],[covxy,sigy]])

#Given these specifications, we can then sample the distn's however we want
#ie we end up with variables like
#d = Distn()
#d.type = "gaussian"
#d.var = "x"
#d.mean = 7
#d.var = 11

#d = Distn()
#d.type = "joint_gaussian"
#d.var = ["x","y"]
#d.mean = [mux, muy]
#d.varcovar = [[etc.][]]

#or then a whole class hierarchy for distributions
#d = JointGaussian(variables, means, varcovarmatrix)
#d = JointUniform(vars,[[x1,x2][y1,y2])
#d = DiscreteUniform(var, {a,b,c})   (for enumerations)

#or special d = Enumeration(var, {a,b,c})

#There are enumerations and distributions
#Enumerations have fixed set of values returned by d.sample()
#Distributions have a number of samples returned by d.sample()

#Then we take 
#d1.sample() X d2.sample() X ...
#until all the relevant variables are specified


teststr = """
x ~ {7,8}
sigy = 1
y = N(2*x, sigy)
"""

import re, random, numpy as np, numpy.random as npr
from copy import deepcopy

from scipy.stats import vonmises, gamma
import scipy.integrate as integrate
# this is not available until scipy 0.14:
#from scipy.stats import  multivariate_normal
# so we have to us numpy.multivariate_normal (at least on my mac)
import scipy.linalg as linalg
from math import *
import math
from numpy import matrix
from math import pi, isnan, exp
from numpy import mean
from scipy.stats import vonmises, gamma

global gIgnoreJointDistn
gIgnoreJointDistn = True

# from CaseGen_General import save_case_matric, case_naming

def draw_uniform(x0,x1):
    val = npr.uniform(x0,x1)
    return val

def draw_normal(mu, sigma):            
    val = npr.normal(mu, sigma)
    return val

def draw_weibull(shape, scale, nsamples=1):
    from numpy.random import weibull
    x = scale * weibull(shape, nsamples)
    return x

def draw_gamma(shape, scale, nsamples=1):
#    print("gamma params:", shape, scale)
    shape = max(1e-3,shape)
    scale = max(1e-3,scale)
    x = gamma.rvs(shape, loc=0, scale=scale, size=nsamples)
    return x

def draw_vonmises(kappa, loc, nsamples=1):
#    print("kappa, loc: ", kappa, loc)
#    x = 180/pi * vonmises.rvs(kappa, loc=loc, size=nsamples)
    x = vonmises.rvs(kappa, loc=loc, size=nsamples)
    return x

def draw_multivariate_normal(mean, cov):
    val = npr.multivariate_normal(mean, cov)
#    val = multivariate_normal.rvs(mean, cov)
    return val

############

def prob_uniform(x,x0,x1):
    return 1/float(x1-x0)

def prob_normal(x, mu, sigma):
    val = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
    return val    

def prob_weibull(x, shape, scale):
    p = (shape/scale)*(x/scale)**(shape-1)*exp(-(x/scale)**shape)
    return p

def prob_gamma(x, shape, scale):
    shape = max(1e-1,shape)
    p = gamma.pdf(x,shape, loc=0, scale=scale)
    return p

def prob_vonmises(x, kappa, loc):
#    print("kappa, loc: ", kappa, loc)
#    p = vonmises.pdf(x*pi/180.0,kappa, loc=loc)
#    p = [vonmises.pdf(x*pi/180.0 + m,kappa, loc=loc) for m in [-2*pi, 0, 2*pi]]
    p = [vonmises.pdf(x + m,kappa, loc=loc) for m in [-2*pi, 0, 2*pi]]
    p = max(p)
    return p

def prob_multivariate_normal(x, mu, sigma):
## not in scipy < 0.14:
#    p = multivariate_normal.pdf(x,mean,cov)
#    return p
# instead:
    x = np.array(x)
    mu = np.array(mu)
    sigma = matrix(sigma)
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


###############################
###############################

#def math_op(op,v1,v2):
#    if (op == "+"):
#        return v1+v2
#    elif (op == "-"):
#        return v1-v2
#    elif (op == "*"):
#        return v1*v2
#    elif (op == "/"):
#        return v1/v2
#    else:
#        raise ValueError("unknown math op %s" % op)

def is_float(s):
    try:
        v = float(s)
        return True
    except:
        return False


def parse_arg(a):
    from shlex import shlex
    obj = shlex(a)
    obj.wordchars += "." # enables parsing decimals
    alist = list(obj)
#    print("parsed ", a, "to ", alist)
#eg: parsed  10 + Vhub/100 - 0.02*(WaveDir+1) to  ['10', '+', 'Vhub', '/', '100', '-', '0.02', '*', '(', 'WaveDir', '+', '1', ')']
    return alist

def merge_dicts(d1,d2):
    d =  {key:d1[key] for key in d1}
    for key in d2:
        d[key] = d2[key]
    return d


class Distribution(object):
    def __init__(self, vstr, ctx):
        self.vstr = vstr # what variable is being defined
        self.ctx = ctx

    def get_bounds(self):
        pass
    def get_partitions(self):
        pass
    def get_name(self):
        return self.vstr
    def calc_prob(self, x):
        return 1
    
    
class EnumDistn(Distribution):
    """ class for distributions that are just list of numbers or single numbers,
    'sampling' is just going to return all values"""
    def __init__(self, vstr, items, ctx):
        super(EnumDistn, self).__init__(vstr, ctx)
        self.items = items

    def sample(self):
#        res = []
#        for i in range(nsample):
#        res.append(random.choice(self.items))
        res = random.choice(self.items)
        res = self.ctx.resolve_value(res)
        return res

    def get_bounds(self):
        """ assumes numbers """
        nums = [float(i[0]) for i in self.items]
        return [min(nums), max(nums)]

    def get_partitions(self):
        return (len(self.items) - 1)
        
    

class FnDistn(Distribution):
    """ class for 'real' distributions like Gaussians, etc. 
    the keys in 'items' are the names of the distributions, assumed to be known 
    and sample-able. items = e.g. {"N": ["2x","sig1"} """
    def __init__(self, vstr, fn, args, ctx):
        super(FnDistn, self).__init__(vstr,ctx)
        self.fn = fn
        self.args = args

        self.is_truncated = False
        self.C = None
        if (fn[0] == "T"):  ## this is a truncated distribution
            self.is_truncated = True
            self.fn = fn[1:]  # strip "T"            
            print("distn is truncated", fn, self.fn)

    def ensure_C(self, argvals):
        if not self.is_truncated:
            return
        if self.C == None:
            self.Vin = argvals[-2]  # last two args to distn are truncation points
            self.Vout = argvals[-1]
            self.is_truncated = False  # temp turn it off b/c we want true integral of pdf over truncated range
            self.C = integrate.quad(self.calc_prob, self.Vin, self.Vout)
            self.is_truncated = True
            self.C = self.C[0]
            print("truncated distribution normalization constant = ", self.C, "range: ", self.Vin, self.Vout)

    def sample(self):
        argvals = []
        for i in range(len(self.args)):
            a = self.args[i]
            argvals.append(self.ctx.resolve_value(a))

        if (self.is_truncated):
            self.ensure_C(argvals)
            val = self.Vin - 1
            while (val < self.Vin or val > self.Vout):
                val = self.raw_sample(argvals)
            return val
        else:
            return self.raw_sample(argvals)

    def raw_sample(self, argvals):
        if (self.fn == "N"):
#            print(" need to sample normal with args = ", argvals)
            val = draw_normal(argvals[0], argvals[1])
        elif (self.fn == "N2"):
#            print(" need to sample 2D normal with args = ", argvals)
            val = draw_multivariate_normal([argvals[0], argvals[1]], [[argvals[2], argvals[4]],[argvals[4], argvals[3]]])
        elif (self.fn == "U"):
#            print(" need to sample uniform with args = ", argvals)
            val = draw_uniform(argvals[0], argvals[1])
        elif (self.fn == "G"):
#            print(" need to sample gamma with args = ", argvals)
            val = draw_gamma(argvals[0], argvals[1])
        elif (self.fn == "VM"):
#            print(" need to sample von Mises with args = ", argvals)
            val = draw_vonmises(argvals[0], argvals[1])
        elif (self.fn == "W"):
            val = draw_weibull(argvals[0], argvals[1])
#            print(" sampled Wiebull with args = ", argvals, "got ", val)
        else:
            raise ValueError("Sorry, unknown distribution: %s" % self.fn)
        return val

    def calc_prob(self, x):
#        print("calc_prob", self.fn, x)
#        x = x[0]
        argvals = []
        for i in range(len(self.args)):
            a = self.args[i]
            argvals.append(self.ctx.resolve_value(a))

        if (self.is_truncated):
            self.ensure_C(argvals)
            if x < self.Vin or x > self.Vout:
                return 0  ## bail here for samples outside range
            C = self.C ## this is normalization for truncated distn
        else:
            C = 1.0  ## untruncated distn's have normalization of 1
            
        if (self.fn == "N"):
            val = prob_normal(x,argvals[0], argvals[1])
        elif (self.fn == "N2"):
            val = prob_multivariate_normal(x,[argvals[0], argvals[1]], [[argvals[2], argvals[4]],[argvals[4], argvals[3]]])
        elif (self.fn == "U"):
            val = prob_uniform(x,argvals[0], argvals[1])
        elif (self.fn == "G"):
            val = prob_gamma(x,argvals[0], argvals[1])
        elif (self.fn == "VM"):
            val = prob_vonmises(x,argvals[0], argvals[1])
        elif (self.fn == "W"):
            val = prob_weibull(x,argvals[0], argvals[1])
#            print("prob W", x, argvals[0], argvals[1], val)
        else:
            raise ValueError("unknown distribution %s" % self.fn)
        
        if (math.isnan(val)):
            print("NAN", val, x, self.fn, argvals)
        return val/C


class DistnParser(object):
    def __init__(self):
        self.vars = []
        self.dlist = []
        self.dlist_map = {}
        
    def parse_file(self,fname):
        mystr = file(fname).readlines()
        return self.parse(mystr)

    def parse(self, mystr):
        for ln in mystr:
            ln = ln.strip()
            newdist = None
            if (len(ln) > 0 and ln[0] != "#"):
                # first get rid of anything past any "#" on the right
                ln = ln.split("#")[0]
 #               print("your line: ", ln)
                # separate vars from the distn
                tok = ln.split("=")
                if len(tok) > 1:
#                    print("found distn: ", tok)
                    vtok = tok[0].split(",")
                    # parse out the vars in question:
                    vstr = ""
                    for v in vtok:
                        v = v.strip()
                        if (v in self.vars):
                            print("ERROR: Variable %s is doubly defined" %v)
                        self.vars.append(v)
                        vstr = "%s" % v
                    # now parse the distn spec.
                    dspec = tok[1].strip()
#                    print("look at", dspec)
                    q = re.match("([^(]+)(\(.*\))", dspec)
                    if (q != None):
                        # found a function-like defn
                        dist = q.group(1) 
                        args = q.group(2)
#                        print("split ", dspec , "into ", dist, args)
                        #eg: split  G(10 + Vhub/100 - 0.02*(WaveDir+1),.25) into  G (10 + Vhub/100 - 0.02*(WaveDir+1),.25)
                        args=args.strip("(").strip(")").split(",")
                        args = [s.strip() for s in args]
                        # now parse variables out of expressions
                        # simple math allowed: <val> := <number> | <val> <op> <val> where <op> = +-*/
                        arglist = []
                        for a in args:
                            alist = parse_arg(a)
                            arglist.append(alist)    
#                        print("found dist=", dist, " arglist=", arglist)
                        newdist = FnDistn(vstr,dist,arglist, self)
                        self.dlist.append(newdist)
                    elif dspec[0] == "{":
                        # found a set
                        args = [s.strip() for s in dspec.strip("{").strip("}").split(",")]
#                        args = [float(s) for s in args]
#                        print("found set ", args)
                        args = [parse_arg(a) for a in args]
#                        print("FOUND set ", args)
                        newdist = EnumDistn(vstr,args, self)
                        self.dlist.append(newdist)
                    else:
                        try:
                            arg = [parse_arg(dspec)]
                            # found a value. TODO: tuples
#                            print("found number ", arg)
                            newdist = EnumDistn(vstr, arg, self)
                            self.dlist.append(newdist)
                        except:
                            print("cannot parse distribution spec ", dspec)
                if (newdist != None):
                    self.dlist_map[vstr] = newdist
                        
        print("defined distns. for vars ", self.vars)
#        print(self.dlist       )
#        print("dlist map = ", self.dlist_map)

        return self.dlist

    def clear_values(self,):
        self.values = {}
    def set_value(self, s,v):
        global gIgnoreJointDistn
        self.values[s] = v
        if (not gIgnoreJointDistn):
            # also deconstruct joint distribution values into their individual variables
            # e.g. Hs.Tp: [1,3] -> Hs: 1, Tp:3
            subv = s.split(".")
            if (len(subv) > 1):
                for i in range(len(subv)):
                    self.values[subv[i]] = v[i]

    def set_values(self, e):
        for key in e:
            self.set_value(key, e[key])

    def sample(self):
        self.clear_values()
        for d in self.dlist:
            s = d.sample()
#            print("sampled ", d.vstr, " and got ", s)
            self.set_value(d.vstr,s)

    def add_enum(self, slist, enum):
        """
        slist = list of samples we're building
        enum = distn with enum.items list of possible values
        output = slist X enum.items
        """
        newlist = []
#        print("add enum, slist, enum.items", slist, enum.items)
        for s in slist: 
            for x in enum.items:
                item = {}
                for y in s:
                    item[y] = s[y]
                    self.set_value(y,s[y])
#                item[enum.vstr] = self.resolve_value(x)
                item[enum.vstr] = x  # not resolved yet!
                newlist.append(item)                
#                print("appended item to newlist" , item, newlist)
#        print(newlist)
        return newlist

    def expand_enums(self):
        """ return list of dicts """
        self.clear_values()
        slist = [{}]
        for d in self.dlist:
            if (hasattr(d,"items")):  #### bad programming!
                slist = self.add_enum(deepcopy(slist), d)
        return slist

    def sample_fns(self):
        """ return list of dicts """
        for d in self.dlist:
            if (hasattr(d,"fn")):
                s = d.sample()
#                print("sampled ", d.vstr, " and got ", s)
                self.set_value(d.vstr,s)
        return self.values

    def multi_sample(self, numsamples, expand_enums=False):
        slist = []
        if (expand_enums):
#            print("expanding set variables, then sampling %d times" % numsamples)
            # make slist of product space of set vars.
            enum_list = self.expand_enums()
#            print("enum_list", enum_list)
            for e in enum_list:
#                self.clear_values()
#                self.set_values(e)
                for i in range(numsamples):
                    # now truly sampled vars to each of them                    
                    # combine real samples to enum cases
                    self.clear_values()

                    for d in self.dlist:
 #                       print("scanning ", d.vstr)
                        if (hasattr(d,"fn")):
                            s = d.sample()
                            #print("sampled ", d.vstr, " and got ", s)
                            self.set_value(d.vstr,s)
                        
                        if (hasattr(d,"items")):  #### bad programming!
                            maxiter = len(e)
                            tries = 0
                            while (tries < maxiter):
                                for it in e:  # find this var and resolve it here
#                                    print(it, e[it])
                                    # this is some crazy stuff: try (maybe out of order b/c of dict) until we succeed
                                    # to resolve everything
                                    try:
                                        val = self.resolve_value(e[it])
                                        self.set_value(it,val)
#                                        print("succes resolving ", e[it])
                                    except:
#                                        print("failed resolving ", e[it])
                                        pass
                                    tries += 1
                    slist.append(self.values)

#                    vals = self.sample_fns()
#                    print(vals, e)
#                    vals = merge_dicts(vals, e)
#                    slist.append(vals)
        else:            
#            print("sampling %d times" % numsamples)
            for i in range(numsamples):
                self.clear_values()
                vals = self.sample()
#                print("sample %d = " % i, self.values)
                slist.append(self.values)
        
        return slist

    def resolve_one_value(self,a):
#        print("resolve_one_value()", a)
        if (is_float(a)):
            return float(a)
        else:
            for d in self.dlist:
                if a == d.vstr:
                    return self.values[a]
#            raise ValueError("did not find variable:%s: in dlist" % a )
        return a

    def resolve_value(self,a):
#        print("resolving:", a)
        vals = [self.resolve_one_value(x) for x in a]
#        print(vals)
        s = ""
        for v in vals:
            if is_float(v):
                s = "%s %f" % (s, v)
            else:
                s = "%s %s" % (s, v)
#        print("evaluating:", s)
        val = eval(s)
#        print("= ", val)
        return val
#        if (len(a) == 1):
#            return self.resolve_one_value(a[0])
#        else:
#            v1 = self.resolve_one_value(a[0])
#            v2 = self.resolve_one_value(a[2])
#            op = a[1]
#            print("2",a)
#            v1 = self.resolve_one_value(a[-1])
#            v2 = self.resolve_value(a[:-2])
#            op = a[-2]
#            return math_op(op, v1, v2)

    def get_num_samples(self):
        self.sample()  # kludge: sample once to get NumSamples set, if it exists
        if "NumSamples" in self.values:
            return int(self.values['NumSamples'])
        else:
            return 1

    def get_bounds(self):
        """ assume all enum distns, get their bounds for conversion to dakota """
        bounds = []
        for d in self.dlist:
            bounds.append(d.get_bounds())
        low = [b[0] for b in bounds]
        high = [b[1] for b in bounds]
        return [low,high]

    def get_partitions(self):
        """ assume all enum, get their length, assume uniform incrs to convert to dakota """
        part = []
        for d in self.dlist:
            part.append(d.get_partitions())
        return part

    def get_names(self):
        """ get names of all the distn's (what vars are defined), for mapping from dakota variable list """
        part = []
        for d in self.dlist:
            part.append(d.get_name())
        return part

    def calc_prob(self, samp):
        """ calc probability of samp according to parsed distribution """
        self.clear_values()
        self.set_values(samp)        
        # for each key/value, need to find the distribution it belongs to
        # plan on using a pre-prepared mapping
        ptot = 1
#        print("calc prob for ", samp)
        for var in self.values:
            if (var in self.dlist_map):
                dist = self.dlist_map[var]
#                print("for ", var)
                p = dist.calc_prob(self.values[var])
                ptot *= p
        return ptot


# def get_options():
#     from optparse import OptionParser
#     parser = OptionParser()    
#     parser.add_option("-i", "--input", dest="dist",  type="string", default="runbatch-dist.txt",
#                                     help="main input file describing distribution, ie cases to run")
#     parser.add_option("-n", "--nsamples", dest="nsamples", help="how many samples to generate", type="int", default=5)
#     parser.add_option("-t", "--tmax", dest="tmax", help="analysis time, default is use tag in distribution file", type="float", default=None)
#     parser.add_option("-o", "--output", dest="main_output",  type="string", default="runcases.txt",
#                                     help="output file (where to write the run cases)")
#     parser.add_option("-p", "--probfile", dest="old_samples",  type="string", default=None,
#                                     help="an input file of samples whose probabilities we want to calculat w.r.t input distn")
#     parser.add_option("-a", "--augment", dest="augment", help="goes with -p, will include _both_ given and newly calculated probs to the output samples", action="store_true", default=False)
            
#     (options, args) = parser.parse_args()
#     return options, args

def read_samples(fname):
    lines = file(fname).readlines()
    hdr = lines[0].split()
    dat = []
    for ln in lines[1:]:
        dat.append([float(x) for x in ln.split()])
    return hdr, dat

def gen_cases(options=None, args=None):
    if options==None:
        options, args = get_options()
    
    dparser = DistnParser()
    dparser.parse_file(options['dist'])

    if (options['old_samples'] != None):
        # in this mode, we are given a distribution and, separately, a set of old samples.  Our job is to calculate the
        # probabilities for the samples w.r.t. the given distribution
        old_hdr, old_samples = read_samples(options['old_samples'])
        pidx = old_hdr.index("Prob")
        new_hdr = old_hdr
        if (options['augment']):
            new_hdr.append("Prob2")
            pidx = len(new_hdr)-1
        new_samples = []
        for s in old_samples:
            samp = {old_hdr[i]:s[i] for i in range(len(s))}
            p = dparser.calc_prob(samp)
            if (options['augment']):
                s.append(p)
            else:
                s[pidx] = p
            new_samples.append(s)

        fout = file(options['main_output'], "w")
        for key in new_hdr:
            fout.write("%s " % key)
        fout.write("\n")

        for s in new_samples:
            for val in s:
                fout.write("%.16e " % val)
            fout.write("\n")
        fout.close()

        print("Calculated probabilities of samples in %s w.r.t. distribution in %s" % (options['old_samples'], options['dist']))
        if (options['augment']):
            print("Augmented output with these probs as Prob2 in field (0-based) %d" % (pidx))
        else:
            print("Replaced old probs with new probs in output")

    else:
        numsamples = options['nsamples']
        slist = dparser.multi_sample(numsamples, expand_enums = True)
        case_list_dist = [{}]*numsamples
        for i in range(len(slist)):
            s = slist[i]
            p =  dparser.calc_prob(s)

            case_list_dist[i] = s
            case_list_dist[i]['p'] = p

        return case_list_dist


def CaseGen_Dists(case_inputs):

    # Default options
    if 'dist' not in case_inputs.keys():
        case_inputs['dist'] = 'runbatch-dist.txt'
    if 'augment' not in case_inputs.keys():
        case_inputs['augment'] = False
    if 'tmax' not in case_inputs.keys():
        case_inputs['tmax'] = 180
    if 'old_samples' not in case_inputs.keys():
        case_inputs['old_samples'] = None
    if 'nsamples' not in case_inputs.keys():
        case_inputs['nsamples'] = 5
    if 'main_output' not in case_inputs.keys():
        case_inputs['main_output'] = 'run_cases.txt'
    if 'dir_matrix' not in case_inputs.keys():
        case_inputs['dir_matrix'] = ''
    if 'namebase' not in case_inputs.keys():
        case_inputs['namebase'] = ''

    # Read distribution input file, generate cases
    case_list_dist = gen_cases(case_inputs, [])

    # Reformat for AeroElasticSE case runner
    case_list = [] 
    for i, case in enumerate(case_list_dist):
        case_list_i = {}
        if 'AnalTime' in case.keys():
            case_list_i[('Fst','TMax')] = [case['AnalTime']]
        if 'Hs' in case.keys():
            case_list_i[('HydroDyn','WaveHs')] = case['Hs']
        if 'Tp' in case.keys():
            case_list_i[('HydroDyn','WaveTp')] = case['Tp']
        if 'WaveDir' in case.keys():
            case_list_i[('HydroDyn','WaveDir')] = case['WaveDir']
        if 'Vhub' in case.keys():
            case_list_i[('InflowWind','HWindSpeed')] = case['Vhub']
        if 'p' in case.keys():
            case_list_i[('P','')] = case['p']

        case_list.append(case_list_i)

    # Save case matrix
    change_vars = sorted(case_list[0].keys())
    matrix_out = np.asarray([[str(case[var][0]) for var in change_vars] for case in case_list])
    if case_inputs['dir_matrix']:
        save_case_matric(matrix_out, change_vars, case_inputs['dir_matrix'])

    # Case naming
    case_name = case_naming(len(case_list), namebase=case_inputs['namebase'])

    return case_list, case_name


if __name__=="__main__":

    case_inputs = {}
    case_inputs['dist']        = 'runbatch-dist.txt'
    case_inputs['tmax']        = 30
    case_inputs['nsamples']    = 5
    case_inputs['main_output'] = 'run_cases.txt'
    case_inputs['dir_matrix']  = 'temp/openFAST'
    case_inputs['namebase']    = 'testing_dists'

    case_list, case_name = CaseGen_Dists(case_inputs)

