import matplotlib.pyplot as plt
from magicclass import magicclass, field
from pathlib import Path
from magicgui.widgets import ComboBox
import numpy as np
import classi

@magicclass
class PlotData:
    """Load 1D data and plot it."""
    mode = field(str,options={'name':'mode', 'value':'Load', 'choices':('Load','Indentation','Cantilever','Piezo')})
    degree = field(int,options={'value':4, 'min':1, 'max':5, 'widget_type':"Slider" })
    smoothness = field(int,options={'value':1, 'min':1, 'max':10, 'widget_type':"Slider" })
    def __init__(self, title=None):
        self.title = title
        self.t = None
        self.y = None
        self.path = None
        self.steps=[]
        self.current = 'Load'
    
    @mode.connect
    def _changeMode(self):
        self.file.y = self.file.getChannel(self.mode.value)
        self.file.y_corrected = self.file.get_corrected()
    
    def load(self, path:Path):
        """Load file."""
        self.file = classi.o11dma(str(path))
        self.file.t = self.file.getChannel('Time')
        self.file.y = self.file.getChannel(self.mode.value)
        self.file.y_corrected = self.file.get_corrected()
    
    def plot_time(self):
        """Plot data in time."""
        fig,ax = plt.subplots()
        self.file.plot(ax,trend=True)                
        plt.legend()
        plt.show()
        
    def plot_detrend(self):
        """Plot data in time corrected."""
        fig,ax = plt.subplots(2,len(self.file.start))
        self.file.plot_segments(ax[0][:])                
        self.file.plot_detrend(ax[1][:],deg=self.degree.value)                
        plt.legend()
        plt.show()
    
    def plot_ratio(self):
        """Plot data."""
        fig,ax=plt.subplots(3,len(self.file.start))
        self.file.plot_ratio(ax[0][:])
        self.file.plot_ratio_corrected(ax[1][:],deg=self.degree.value)
        self.file.plot_lissajou(ax[2][:],deg=self.degree.value)
        plt.show()
        
    def plot_compare(self):
        """Plot data."""
        fig,ax=plt.subplots(5,len(self.file.start),sharex='col',sharey='col')
        self.file.plot_lissajou(ax[0][:],deg=1,marker='-',alpha=0.75)
        self.file.plot_lissajou(ax[1][:],deg=2,marker='-',alpha=0.75)
        self.file.plot_lissajou(ax[2][:],deg=3,marker='-',alpha=0.75)
        self.file.plot_lissajou(ax[3][:],deg=4,marker='-',alpha=0.75)
        self.file.plot_lissajou(ax[4][:],deg=5,marker='-',alpha=0.75)
        plt.show()
        
    def correlate(self):
        fig,ax=plt.subplots(1,len(self.file.start))
        self.file.phase_correlate(ax,deg=self.degree.value,smoothness=self.smoothness.value)
        
    def synthetic(self):
        fig,axs=plt.subplots(4,6)
        for i in range(6):
            self.file.fake(axs[0][i],deg=self.degree.value,smoothness=1,noiselevel = i*3, noisetype = 0)
        for i in range(6):
            self.file.fake(axs[1][i],deg=self.degree.value,smoothness=1,noiselevel = i*3, noisetype = 1)
        for i in range(6):
            self.file.fake(axs[2][i],deg=self.degree.value,smoothness=1,noiselevel = i*3, noisetype = 2)
        for i in range(6):
            self.file.fake(axs[3][i],deg=self.degree.value,smoothness=1,noiselevel = i*3, noisetype = 3)
        
            
if __name__ == "__main__":
    ui = PlotData()
    ui.show()