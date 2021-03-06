"""
Part II of Correlations and Hypothesis Testing Tutorial
Author: Sheila Kannappan
adapted from ASTR502 activity June 9, 2015
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import numpy.random as npr
import pylab
pylab.ion()

# experiment to illustrate Monte Carlo creation of chi^2 
# distributions for two values of N

i=0
if i == 0:
#for i in xrange(2):

  narr=30.*(1.+9.*i) #30 first time, 300 second time
  chisqs=[]
  iters=100  # experiment with rerunning the plots many times
             # to see how the random variations change the chi^2
             # start w/ iters=100 then change iters to 1000 and repeat

  for j in xrange(iters):
    # create a data set with random errors whose underlying
    # functional form is y=1/x
    xvals = np.zeros(narr)
    yvals = np.zeros(narr)
    xvals = np.arange(narr)/(1.+9.*i) + 1.
    errs=0.1
    yvals = 1./xvals + npr.normal(0,errs,narr)
    # what does npr.normal do?

    resids = np.abs(yvals - 1./xvals)
    # why are we subtracting 1./xvals?

    chisq = np.sum(resids**2 / errs**2)
    # roughly, how is chisq related to N?

    # what if we didn't know the errors and overestimated
    # them as 0.2 -- how would chi^2 change?

    chisqs.append(chisq) # building up iters trial values of chi^2
    
  if i==0:
      redchisqdist1 = np.array(chisqs)/narr
      # compute array of "reduced chi^2" values found for 1st N
      # what does "reduced" mean? why are we dividing by narr?

  if i==1:
      redchisqdist2 = np.array(chisqs)/narr
      # same thing for 2nd N 
      
plt.figure(2)
plt.clf()
n1, bins1, patches1 = plt.hist(redchisqdist1,bins=0.05*iters,normed=1,histtype='stepfilled')
#n2, bins2, patches2 = plt.hist(redchisqdist2,bins=0.05*iters,normed=1,histtype='step')
plt.setp(patches1,'facecolor','g','alpha',0.75)
plt.xlim(0,2.5)
#plt.setp(patches2,'hatch','///','alpha',0.75,color='blue')
# what good plotting strategies are being used here?

plt.title('comparison of reduced chi-squared distributions')
#D,pnull=stats.ks_2samp(???) # fill in -- what are the two things we're comparing?
# why is the K-S test appropriate here?

#confidence=stats.norm.interval(1.-2.*pnull)
#leveltext1='K-S test: distributions differ'
#leveltext2='at %0.1f' % confidence[1]
# what is the K-S test telling you?

#define sigma symbol as a string for use on plots
sigmasym=r'$\sigma$'

#plt.text(1.5,4,leveltext1, size=11, color='r')
#plt.text(1.5,3.7,leveltext2+sigmasym+' confidence', size=11, color='r')
plt.text(1.4,1,"N=30",size=11,color='g')
#plt.text(1.2,3,"N=300",size=11,color='b')

# Q: how can we determine the confidence level associated with
#    a certain deviation of reduced chi^2 from 1?
# A: we have to do the integral under the distribution --
#    e.g. for 3sigma (99.8%) confidence find x such that
#    the integral up to x contains 99.8% of the area
# how can you approximate this using np.argsort?
# make sure to set iters=1000 for this exercise....
inds=np.argsort(redchisqdist1)
#print() # fill in to print red. chi^2 for 99.8% conf. for N=30
#inds=np.argsort(redchisqdist2)
#print() # fill in to print red. chi^2 for 99.8% conf. for N=300
# look at the graph and verify your answers make visual sense
