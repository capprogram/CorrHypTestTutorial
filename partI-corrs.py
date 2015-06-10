"""
Part I of Correlations and Hypothesis Testing Tutorial
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

# pearson vs. spearman rank correlation tests
data=np.loadtxt("anscombe.txt")
x1=data[:,0]
y1=data[:,1]
x2=data[:,2]
y2=data[:,3]
x3=data[:,4]
y3=data[:,5]
x4=data[:,6]
y4=data[:,7]

# multi-panel plotting (see http://matplotlib.org/users/pyplot_tutorial.html)
plt.figure(1)
plt.clf()

plt.subplot(221) # 2 rows x 2 columns of plots, 1st plot (top left)
plt.plot(x1, y1,'g.',markersize=10)
testxvals=np.array([3.,7.,11.,15.,19.]) # use to make line
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y1-(3.+0.5*x1))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
# notice how the "plt.text" command works
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('standard')
plt.ylabel('y')

plt.subplot(222) # 2 rows x 2 columns of plots, 2nd plot (top right)
plt.plot(x2, y2,'g.',markersize=10)
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y2-(3.+0.5*x2))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('curved')

plt.subplot(223) # 2 rows x 2 columns of plots, 3rd plot (bottom left)
plt.plot(x3, y3,'g.',markersize=10)
plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
rms=np.sqrt(np.mean((y3-(3.+0.5*x3))**2))
plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
plt.xlim(2,20)
plt.ylim(2,14)
plt.title('outlier')
plt.xlabel('x')
plt.ylabel('y')

#plt.subplot(???) # you fill in ??? (bottom right)
#plt.plot(x4, y4,'g.',markersize=10)
#plt.plot(testxvals,3.+0.5*testxvals,'r',linestyle=':',linewidth=2.)
#rms=??? # you fill in ???
#plt.text(3,12,'rms %0.2f' % rms,size=11,color='b')
#plt.xlim(2,20)
#plt.ylim(2,14)
#plt.title('garbage')
#plt.xlabel('x')

# fine-tune figure -- clean up space between plots
# try commenting different lines out to see what the plot is like without them
ax=plt.subplot(221) # set ax equal to first subplot
plt.setp(ax.get_xticklabels(), visible=False) # hide its xlabels
ax=plt.subplot(222) # set ax equal to second subplot
plt.setp(ax.get_xticklabels(), visible=False) # hide its xlabels
plt.setp(ax.get_yticklabels(), visible=False) # hide its ylabels
#ax=plt.subplot(224) # set ax equal to fourth subplot
#plt.setp(ax.get_yticklabels(), visible=False) # hide its ylabels

#define sigma symbol as a string for use on plots
sigmasym=r'$\sigma$'

plt.subplot(221) # back to first subplot to do correlation tests
cc,pnull=stats.spearmanr(x1,y1)
# print info to screen
print(" ")
print("Standard:")
print("Spearman rank correlation coefficient %f" % cc)
#print("Spearman rank probability of no correlation %f" % pnull)
# add info to plot at position (x,y)=(8.5,3)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Spearman rank %0.1f' % confidence[1]
#plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')
# similarly:
cc,pnull=stats.pearsonr(x1,y1)
print("Pearson correlation coefficient %f" % cc)
#print("Pearson probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Pearson %0.1f' % confidence[1]
#plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

plt.subplot(222)
cc,pnull=stats.spearmanr(x2,y2)
print(" ")
print("Curved:")
print("Spearman rank correlation coefficient %f" % cc)
#print("Spearman rank probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Spearman rank %0.1f' % confidence[1]
#plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')
cc,pnull=stats.pearsonr(x2,y2)
print("Pearson correlation coefficient %f" % cc)
#print("Pearson probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Pearson %0.1f' % confidence[1]
#plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

plt.subplot(223)
cc,pnull=stats.spearmanr(x3,y3)
print(" ")
print("Outlier:")
print("Spearman rank correlation coefficient %f" % cc)
#print("Spearman rank probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Spearman rank %0.1f' % confidence[1]
#plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')
cc,pnull=stats.pearsonr(x3,y3)
print("Pearson correlation coefficient %f" % cc)
#print("Pearson probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Pearson %0.1f' % confidence[1]
#plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')

plt.subplot(224)
cc,pnull=stats.spearmanr(x4,y4)
print(" ")
print("Garbage:")
print("Spearman rank correlation coefficient %f" % cc)
#print("Spearman rank probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Spearman rank %0.1f' % confidence[1]
#plt.text(8.5,3,leveltext+sigmasym, size=11, color='b')
cc,pnull=stats.pearsonr(x4,y4)
print("Pearson correlation coefficient %f" % cc)
#print("Pearson probability of no correlation %f" % pnull)
#confidence=stats.norm.interval(???) # you fill in ???
#leveltext='Pearson %0.1f' % confidence[1]
#plt.text(8.5,4.5,leveltext+sigmasym, size=11, color='b')
