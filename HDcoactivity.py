
import scipy.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

name = 'Mous 29.mat'

sigma = 5 # window for smoothing
thresholdforneuronstokeep = 1000 # number of spikes to be considered useful

mat = scipy.io.loadmat(name) 
headangle = np.ravel(np.array(mat['headangle']))
cellspikes = np.array(mat['cellspikes'])
cellnames = np.array(mat['cellnames'])
trackingtimes = np.ravel(np.array(mat['trackingtimes']))

## make matrix of spikes
startt = np.min(trackingtimes)
binsize = np.mean(trackingtimes[1:]-trackingtimes[:(-1)])
nbins = len(trackingtimes)
binnedspikes = np.zeros((len(cellnames), nbins))
celldata = np.zeros(np.shape(binnedspikes))
sgood = np.zeros(len(celldata[:,0]))<1
for i in range(len(cellnames)):
  spikes = np.ravel((cellspikes[0])[i])
  for j in range(len(spikes)):
    # note 1ms binning means that number of ms from start is the correct index
    tt = int(np.floor(  (spikes[j] - startt)/float(binsize)  ))
    if(tt>nbins-1 or tt<0): # check if outside bounds of the awake time
      continue
    binnedspikes[i,tt] += 1 # add a spike to the thing

  ## check if actvitity is ok
  if(sum(binnedspikes[i,:])<thresholdforneuronstokeep):
      sgood[i] = False
      continue

  ## smooth and center the activity
  celldata[i,:] = scipy.ndimage.filters.gaussian_filter1d(binnedspikes[i,:], sigma)
  celldata[i,:] = (celldata[i,:]-np.mean(celldata[i,:]))/np.std(celldata[i,:])

celldata = celldata[sgood, :]
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]

##################
##################
##################
##### above is same in both scripts (except sigma=5)! below is different
##################
##################
##################

ncells = len(celldata)

# Binary coactivity vector for each cell; 1 in bin i if at least 5 bins 
# in the 21 window {i-10,...,i,...,i+10} has a quantiled value of over 0.90
biactivity = np.zeros(np.shape(celldata))
for i in range(len(celldata)):
   for j in range(len(celldata[i])):
      N=0
      if j>10:
        for k in range(21):
          try:
            if celldata[i][j-10+k]>0.90:
              N +=1
            else:
              continue
          except IndexError:
            pass
          continue
        if N>=5:
          biactivity[i][j]=1
        else:
          continue 
      else:
        for k in range(0,j+11):
          if celldata[i][k]>0.90:
            N +=1
          else:
            continue
        if N>=5:
          biactivity[i][j]=1
        else:
          continue

#Computes which pairs of cells are significantly coactive
'''
def get_indices(list):
  indices=[]
  for i in range(len(list)):
    if list[i]==1:
      indices.append(i)
  return indices

def getcoactivityscore(x,y):
  xinds=get_indices(x)
  yinds=get_indices(y)
  return (len(list(set(xinds)&set(yinds))) / ((len(xinds)*len(yinds))**0.5)
          - ((len(xinds)*len(yinds))**0.5)/nbins)



num_shuffles = 1000
binary_significance_matrix = np.zeros((len(biactivity),len(biactivity)))


for i in range(1):
  for j in range(1):
    coactiviy_score = getcoactivityscore(biactivity[i],biactivity[j])
    shuffled_coactivity_scores=np.zeros(num_shuffles)
    for k in range(num_shuffles):
      shift_amount=np.random.randint(len(biactivity[j]))
      shuffled_cell=np.roll(biactivity[j],shift_amount)
      shuffled_coactivity_scores[k] = getcoactivityscore(biactivity[i],shuffled_cell)
      print((j,k))
    N=0
    for score in shuffled_coactivity_scores:
      if score<coactiviy_score:
        N+=1
      else:
        continue
    p_estimate = 1-N/1000
    if p_estimate < 0.05:
      binary_significance_matrix[i][j]=1
    else:
      continue

#Slightly faster code to compute which pairs are significantly coactive
#Code below took 13 hours to run, ran it once and saved it locally on pc.
for j in range(len(biactivity)-1,-1,-1):
  shuffles=[]
  for k in range(num_shuffles):
    shift_amount=np.random.randint(len(biactivity[j]))
    shuffles.append(np.roll(biactivity[j],shift_amount))
  for i in range(j,-1,-1):
    coactiviy_score = getcoactivityscore(biactivity[i],biactivity[j])
    shuffled_coactivity_scores=[getcoactivityscore(biactivity[i],shuffle) for shuffle in shuffles]
    N=0
    for score in shuffled_coactivity_scores:
      if score<coactiviy_score:
        N+=1
      else:
        continue
    p_estimate=1-N/1000
    if p_estimate<0.05:
      binary_significance_matrix[i,j]=1

''' 

# Significance assumed to be symmetric i.e. i,j significant iff j,i significant hence  
# its enough to compute the upper triangle and then copy the values into the lower triangle
significance_matrix = np.load('significance_matrix.npy')
significance_matrix = significance_matrix + np.transpose(np.triu(significance_matrix,1))

#Sort cells by headangle
  # Make tuning curves
angles = np.arange(0, 2.*np.pi+0.0001, np.pi/10)
occupancies = np.zeros(len(angles)-1)
tuningcurve = np.zeros((len(cellnames), len(angles)-1))

for i in range(len(angles)-1):
  inds = (headangle>=angles[i]) * (headangle<angles[i+1]) # Number of 
  occupancies[i] = np.sum(inds)
  if(occupancies[i]>0):
    for j in range(len(cellnames)):
      tuningcurve[j,i] = np.sum( binnedspikes[j,inds] ) / float( occupancies[i] )
      tuningcurve[j,i] = tuningcurve[j,i] * 1000./binsize
  else:
    tuningcurve[:,i] = np.NAN
anglevertices = (0.5*(angles[0:(-1)]+angles[1:])) 

dataframe = [[binnedspikes[i], np.argmax(tuningcurve[i])] for i in range(len(tuningcurve))]

import pandas

df = pandas.DataFrame(data=dataframe)
df = df.sort_values(by=1, ascending=True) #Sorted from 0 to 2pi
df
sorted_index_values = np.copy(df.index.values)

#Sort significance matrix by preferred head angle
sorted_significance_matrix = np.zeros(np.shape(significance_matrix))
for i in range(len(significance_matrix)):
  for j in range(len(significance_matrix)):
    sorted_significance_matrix[i,j]=significance_matrix[sorted_index_values[i],sorted_index_values[j]]

sequence_of_coactivity_matrices=[np.zeros(np.shape(sorted_significance_matrix)) for _ in range(nbins)]
for i in range(len(sorted_significance_matrix)):
  for j in range(len(sorted_significance_matrix)):
    if sorted_significance_matrix[i][j]==1:
      for t in range(nbins):
        if biactivity[sorted_index_values[i]][t]==1 and biactivity[sorted_index_values[j]][t]==1:
          sequence_of_coactivity_matrices[t][i][j]=1
        else:
          continue
    else:
      continue

def get_coactivity_over_time():
  total_coactive_pairs = np.sum(np.triu(significance_matrix))  # of unique pairs
  proportion_of_coactivity_over_time = []
  for t in range(nbins):
    number_of_active_coactive_pairs = np.sum(np.triu(sequence_of_coactivity_matrices[t]))
    proportion_of_coactivity_over_time.append(number_of_active_coactive_pairs/total_coactive_pairs)
  return proportion_of_coactivity_over_time
  
coactivity_over_time = get_coactivity_over_time()

# #Saves coactivity matrices locally
# import os
# os.chdir(r"C:\Users\Bruker\Desktop")
# for t in range(6677,nbins):
#   fig = plt.figure(figsize=(10,8), layout='tight')
#   ax1 = plt.subplot(221)
#   ax2 = plt.subplot(222, projection='polar')
#   ax3 = plt.subplot(212)

#   #Coactivity matrix
#   ax1.imshow(sequence_of_coactivity_matrices[t], cmap='viridis')
#   ax1.set_title(f'Coactivity at t={t}',color='w')
#   ax1.set_xticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='w')
#   ax1.set_yticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='w')

#   # Head direction
#   theta=headangle[t]
#   ax2.plot(theta,1,'ro',markersize=20)
#   ax2.set_theta_zero_location('N')  # North is the zero direction
#   ax2.set_theta_direction(-1)  # Clockwise direction
#   ax2.set_rticks([])
#   ax2.set_rmax(1)
#   ax2.set_theta_offset(np.pi / 2.0)  # Set the direction of the plot (0 degrees is North)
#   ax2.set_xticks(np.deg2rad(np.arange(0, 360, 45)))  # Set ticks every 45 degrees
#   ax2.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], color='w')
#   ax2.set_title("Animal's Head Direction", color='w')

#   # Coactivity over time
#   ax3.plot(np.arange(nbins),coactivity_over_time)
#   ax3.axvline(x=t, color='r')
#   ax3.margins(x=0.01)
#   ax3.set_title(f"{round(coactivity_over_time[t]*100,2)}%", loc='left', color='w')
#   ax3.set_yticks([0,0.40], [0,'max'], color='w')
#   ax3.set_ylabel('% of coactive neurons currently active', color='w')
#   ax3.set_xticks([0,400],['10s',''], color='w')
#   ax3.set_xlabel('Time', color='w')

#   plt.savefig(f'HDframes/{00000+t}.png',bbox_inches='tight',pad_inches=0.1,transparent=True)
  plt.close()

###### For testing plots #########
# fig = plt.figure(figsize=(10,8), layout='tight')
# ax1 = plt.subplot(221)
# ax2 = plt.subplot(222, projection='polar')
# ax3 = plt.subplot(212)

# #Coactivity matrix
# ax1.imshow(sequence_of_coactivity_matrices[6000], cmap='viridis')
# ax1.set_title(f'Coactivity at t={6000}',color='k')
# ax1.set_xticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='k')
# ax1.set_yticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='k') 

# # Head direction
# theta=headangle[6000]
# ax2.plot(theta,1,'ro', marker='>', markersize='30')
# ax2.set_theta_zero_location('N')  # North is the zero direction
# ax2.set_theta_direction(-1)  # Clockwise direction
# ax2.set_rticks([])
# ax2.set_rmax(1)
# ax2.set_theta_offset(np.pi / 2.0)  # Set the direction of the plot (0 degrees is North)
# ax2.set_xticks(np.deg2rad(np.arange(0, 360, 45)))  # Set ticks every 45 degrees
# ax2.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
# ax2.set_title("Animal's Head Direction")

# # Coactivity over time
# ax3.plot(np.arange(nbins),coactivity_over_time)
# ax3.axvline(x=6000, color='r')
# ax3.margins(x=0.01)
# ax3.set_title(f"{round(coactivity_over_time[6000]*100,2)}%", loc='left')
# #ax3.text(6000, 0.35, f'{round(coactivity_over_time[6000]*100,2)}%')
# ax3.set_yticks([0,0.40], [0,'max'])
# ax3.set_ylabel('% of coactive neurons currently active')
# ax3.set_xticks([0,400],['10s',''])
# ax3.set_xlabel('Time')

# plt.show()

########################



#### Paralellization of plotting #####
# import os
# os.chdir(r"C:\Users\Bruker\Deskop")
# from multiprocessing import Pool

# def main():
#   fig = plt.figure(figsize=(10,8), layout='tight')
#   ax1 = plt.subplot(221)
#   ax2 = plt.subplot(222, projection='polar')
#   ax3 = plt.subplot(212)

#   #Coactivity matrix
#   ax1.imshow(sequence_of_coactivity_matrices[t], cmap='viridis')
#   ax1.set_title(f'Coactivity at t={t}',color='w')
#   ax1.set_xticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='w')
#   ax1.set_yticks([0,ncells],[r'$0^{\circ}$',r'$360^{\circ}$'],color='w')

#   # Head direction
#   theta=headangle[t]
#   ax2.plot(theta,1,'ro',markersize=20)
#   ax2.set_theta_zero_location('N')  # North is the zero direction
#   ax2.set_theta_direction(-1)  # Clockwise direction
#   ax2.set_rticks([])
#   ax2.set_rmax(1)
#   ax2.set_theta_offset(np.pi / 2.0)  # Set the direction of the plot (0 degrees is North)
#   ax2.set_xticks(np.deg2rad(np.arange(0, 360, 45)))  # Set ticks every 45 degrees
#   ax2.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], color='w')
#   ax2.set_title("Animal's Head Direction", color='w')

#   # Coactivity over time
#   ax3.plot(np.arange(nbins),coactivity_over_time)
#   ax3.axvline(x=t, color='r')
#   ax3.margins(x=0.01)
#   ax3.set_title(f"{round(coactivity_over_time[t]*100,2)}%", loc='left', color='w')
#   ax3.set_yticks([0,0.40], [0,'max'], color='w')
#   ax3.set_ylabel('% of coactive neurons currently active', color='w')
#   ax3.set_xticks([0,400],['10s',''],color='w')
#   ax3.set_xlabel('Time', color='w')

#   plt.savefig(f'HDframes/{00000+t}.png',bbox_inches='tight',pad_inches=0.1,transparent=True)
#   plt.close()
#   return

# if __name__ == '__main__':
#   with Pool(4) as pool:
#     pool.map(main,[t for t in range(6084,6200)])

# import os
# os.chdir(r"C:\Users\Bruker\OneDrive - NTNU\Bachelor i matematiske fag\3. år\6. semester\Bacheloroppgave\Head direction data and script")
    
#####################