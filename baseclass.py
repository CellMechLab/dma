import numpy as np
from scipy.signal import savgol_filter,find_peaks,correlate
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt

class o11dma(object):
    def __init__(self,filename) -> None:
        self.header = []
        self.segments = []
        self.data=[]
        self.channels=[]
        self.chunits=[]
        
        reader = open(filename,'r')
        for line in reader:
            if line[:8]=='Time (s)':
                channels = line.strip().split('\t')
                for ch in channels:
                    name = ch.split(' ')
                    if len(name)==2:
                        self.channels.append(name[0])
                        self.chunits.append(name[1][1:-1])
                break                
            else:
                if line.strip() != '':                    
                    self.header.append(line.strip())
                    if 'DMA absolute start times' in line:                        
                        self.start = [float(n) for n in line.split('\t')[1].split(',')]
                    if 'DMA absolute end times' in line:                        
                        self.stop = [float(n) for n in line[line.find('(s)')+3:].strip().split(',')]
        for line in reader:
            if line.strip() != '':
                numbers = [float(n) for n in  line.split('\t')]
                self.data.append(numbers)                
        reader.close()
        self.profile = []
        profile = False
        self.data=np.array(self.data)
        for linea in self.header:
            if profile is True:
                if 'Depth' in linea:
                    break
                pro={}
                pezzi = linea.split(' ')
                for i in range(0,len(pezzi),2):
                    pro[pezzi[i]]=float(pezzi[i+1])
                self.profile.append(pro)
            if 'Profile' in linea:
                profile = True 
                
    
    def plot(self,ax,trend=False):
        y = self.y
        ax.plot(self.t,self.y)
        if trend is True:
            for i in range(len(self.start)):
                start= self.start[i]
                stop = self.stop[i]
                istart = np.argmin((self.t-start)**2)
                istop = np.argmin((self.t-stop)**2)
                freq = self.profile[i]['Freq']
                pls = ax.plot(self.t[istart:istop],self.y[istart:istop],label=str(freq))
                pointsUp = self.find_peaks(self.t[istart:istop],self.y[istart:istop],freq)            
                p1 = np.polynomial.Polynomial.fit(self.t[istart+pointsUp],self.y[istart+pointsUp], deg=3)
                pointsDown = self.find_peaks(self.t[istart:istop],-self.y[istart:istop],freq)
                p2 = np.polynomial.Polynomial.fit(self.t[istart+pointsDown],self.y[istart+pointsDown], deg=3)
                points = np.append(pointsUp,pointsDown)
                ax.plot(self.t[istart:istop],p1(self.t[istart:istop]),'--',color=pls[0].get_color())
                ax.plot(self.t[istart:istop],p2(self.t[istart:istop])  ,'--',color=pls[0].get_color())
                ax.plot(self.t[istart+points],self.y[istart+points] ,'o',color=pls[0].get_color())
    
    def plot_segments(self,ax):
        for i in range(len(self.start)):
            start= self.start[i]
            stop = self.stop[i]
            istart = np.argmin((self.t-start)**2)
            istop = np.argmin((self.t-stop)**2)
            freq = self.profile[i]['Freq']
            sx = self.t[istart:istop]
            sy = self.y[istart:istop]            
            ax[i].plot(sx,sy,label=str(freq))
            ax[i].legend()
    
    def plot_detrend(self,ax,deg=3):
        for i in range(len(self.start)):
            start= self.start[i]
            stop = self.stop[i]
            istart = np.argmin((self.t-start)**2)
            istop = np.argmin((self.t-stop)**2)
            freq = self.profile[i]['Freq']
            
            sx = self.t[istart:istop]
            sy = self.y[istart:istop]
            
            pointsUp = self.find_peaks(sx,sy,freq)            
            pointsDown = self.find_peaks(sx,-sy,freq)
            p1 = np.polynomial.Polynomial.fit(sx[pointsUp],sy[pointsUp], deg=deg)
            offset1 =  p1(sx[pointsUp[0]])           
            p2 = np.polynomial.Polynomial.fit(sx[pointsDown],sy[pointsDown], deg=deg)
            offset2 =  p2(sx[pointsDown[0]])                       
            delta1 = p1(sx)-offset1
            delta2 = p2(sx)-offset2
            alpha2 = (p1(sx)-sy)
            alpha1 = (sy-p2(sx))
            newsy = (alpha1*delta1+alpha2*delta2)/(alpha1+alpha2)            
            ax[i].plot(sx,sy-newsy,label=str(freq))
            ax[i].legend()
        
    def calculate_phase(self,deg=3):
        t = self.getChannel('Time')
        x = self.getChannel('Indentation')
        y = self.getChannel('Load')
        for i in range(len(self.start)):
            start= self.start[i]
            stop = self.stop[i]
            istart = np.argmin((t-start)**2)
            istop = np.argmin((t-stop)**2)
            freq = self.profile[i]['Freq']
            
            st = t[istart:istop]
            sy = y[istart:istop]
            
            #Load
            pointsUpLoad = self.find_peaks(st,sy,freq)                        
            pointsDownLoad = self.find_peaks(st,-sy,freq)
            p1 = np.polynomial.Polynomial.fit(st[pointsUpLoad],sy[pointsUpLoad], deg)
            offset1 =  p1(st[pointsUpLoad[0]])           
            p2 = np.polynomial.Polynomial.fit(st[pointsDownLoad],sy[pointsDownLoad], deg)
            offset2 =  p2(st[pointsDownLoad[0]])
            delta1 = p1(st)-offset1
            delta2 = p2(st)-offset2
            alpha2 = (p1(st)-sy)
            alpha1 = (sy-p2(st))
            corrected_load = sy- (alpha1*delta1+alpha2*delta2)/(alpha1+alpha2)

            #Indentation
            sx = x[istart:istop]
            pointsUp = self.find_peaks(st,sx,freq)            
            pointsDown = self.find_peaks(st,-sx,freq)
            p1 = np.polynomial.Polynomial.fit(st[pointsUp],sx[pointsUp], deg)
            p2 = np.polynomial.Polynomial.fit(st[pointsDown],sx[pointsDown], deg)
            
            dt = t[1]-t[0]
            semiperiod = 1/freq/2
            iquad = int(semiperiod/dt)
            
            window = int(iquad/9)
            if window%2 ==0:
                window+=1
            
            corrected_load_smooth = savgol_filter(corrected_load,window,3)
            corrected_ind_smooth = savgol_filter(sx,window,3)
            
            self.profile[i]['A_load']=(max(corrected_load_smooth)-min(corrected_load_smooth))
            self.profile[i]['A_ind'] =(max(corrected_ind_smooth)-min(corrected_ind_smooth))
            
            resample_load = -1 + 2*(corrected_load_smooth-min(corrected_load_smooth))/(max(corrected_load_smooth)-min(corrected_load_smooth))
            resample_ind = -1 + 2*(corrected_ind_smooth-min(corrected_ind_smooth))/(max(corrected_ind_smooth)-min(corrected_ind_smooth))
                                                
            time = np.linspace(np.min(st),np.max(st),len(st))
            cmax = 0
            ext_load = np.interp(time, st, resample_load)
            ext_ind = np.interp(time, st, resample_ind)
            kmax=0
            delta=time[1]-time[0]
            for k in range(iquad):
                corr = np.sum(ext_load[:-iquad]*ext_ind[k:-iquad+k])
                if corr>cmax:
                    cmax=corr
                    kmax=k
                    
            self.profile[i]['phase']= np.pi*delta*kmax/semiperiod #phase is in radiants!

            
    def find_peaks(self,t,y,omega):
        dT = 1/omega/2
        idT = np.argmin((t-(dT+t[0]))**2)
        delta = (max(y)-min(y))/3       
        points = find_peaks(y,height=max(y)-delta,distance=idT)
        return points[0]
        
    def getChannel(self,ch):
        if ch in self.channels:
            index = self.channels.index(ch)
        else:
            index = ch
        return self.data[:,index]
    
    def get_corrected(self):
        return None