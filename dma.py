import matplotlib.pyplot as plt
from magicclass import magicclass, field
from pathlib import Path
from magicgui.widgets import ComboBox
import numpy as np
import baseclass as classi

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
    
    def correlate(self):
        self.file.calculate_phase(deg=self.degree.value)
        for i in range(self.file.start):
            print (f'--- Segment {i} ---')
            print ( f'Frequency: {self.file.profile[i]["Freq"]}')
            print ( f'A Load: {self.file.profile[i]["A_load"]}')
            print ( f'A Indentation: {self.file.profile[i]["A_ind"]}')
            print ( f'Phase: {self.file.profile[i]["phase"]}')
        
            
if __name__ == "__main__":
    ui = PlotData()
    ui.show()