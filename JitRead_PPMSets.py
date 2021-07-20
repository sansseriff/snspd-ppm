

from TimeTagger import createTimeTagger, FileWriter, FileReader
import os
import numpy as np
from numba import njit
import yaml
import time
import json
import timeit
from bokeh.plotting import figure, output_file, show
from scipy.stats import norm
from scipy.interpolate import interp1d
from math import sin, cos
from scipy.stats import norm
import phd
# import matplotlib
from datetime import datetime

from ClockTools_PPMSets import clockScan
import matplotlib
import matplotlib.pyplot as plt

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.plotting import figure, curdoc

from bokeh.io import export_svg
from bokeh.layouts import gridplot
from bokeh.models import Span
import phd.viz as viz
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis
from bokeh.models import HoverTool
from bokeh.models.markers import Circle

from bokeh.models import ColumnDataSource

#Colors, palette = phd.viz.phd_style( data_width = 0.3, grid = True, axese_width=0.3, text = 2)
matplotlib.rcParams['figure.dpi'] = 150
doc = curdoc()
Colors, pallet = viz.bokeh_theme(return_color_list=True)
colors = Colors
output_file("layout.html")




def checkLocking(Clocks, RecoveredClocks):

    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    x = np.arange(0, len(diffs))
    source = ColumnDataSource(data=dict(
        x=x[1:-1:8],
        y1=diffs[1:-1:8],
        y2=diffsRecovered[1:-1:8]))


    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)")
    ]
    s1 = figure(plot_width=800,
                plot_height=400,
                title="Clock Locking",
                output_backend="webgl",
                tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)

    s1.line('x', 'y1', source = source, color = Colors['light_purple'])
    s1.line('x', 'y2', source = source, color = Colors['black'])
    s1.xaxis.axis_label = "time"
    s1.yaxis.axis_label = "Amplitude"


    s2 = figure(plot_width=800, plot_height=400,title="Clock Jitter", output_backend = "svg")
    hist,bins = np.histogram(diffs[9000:12000], bins=100, density=True)
    s2.line(bins[:-1],hist,color = colors['orange'])
    # ax[1].hist(diffs[9000:12000], bins=100, density=True)
    mean = np.mean(diffs)
    variance = np.var(diffs)
    sigma = np.sqrt(variance)
    x2 = np.linspace(min(diffs), max(diffs), 100)
    lbl = "fit, sigma = " + str(int(sigma))
    # ax[1].plot(x2, norm.pdf(x2, mean, sigma), label=lbl)
    # ax[1].legend()
    s2.line(x2, norm.pdf(x2, mean, sigma), color = Colors['black'],legend_label = lbl)
    print("this is the length of the plotted array: ", len(diffs))

    return s1,s2




def checkLockingEffect(dataTags, dataTagsR, xlim=-1, ylim=-1):
    L = len(dataTags) // 2
    if xlim == -1:
        xlim = 0
    if ylim == -1:
        ylim = np.max(dataTags[L:])

    #print("data tags R", dataTagsR[:50])
    #print("data tags ", dataTags[:50])
    setBins = np.arange(xlim, ylim, 1)
    hist, bin_edges = np.histogram(dataTags[L:], setBins)
    histRough, bins_edges = np.histogram(dataTagsR[L:], setBins)
    # fig,ax = plt.subplots(1,1,figsize = (6,4))
    s3 = figure(plot_width=800, plot_height=400,title="Data with Phased Locked vs Regular Clock")
    s3.line(bin_edges[:-1], hist, legend_label="Phased Locked Clock", color = colors['light_purple'])
    s3.line(bin_edges[:-1], histRough, legend_label="Regular Clock", color = colors['purple'])

    # ax.plot(bin_edges[:-1], hist, label="Phased Locked Clock")
    # ax.plot(bin_edges[:-1], histRough, label="Regular Clock")
    # ax.legend()
    return s3


@njit
def countRateMonitor_b(timetaggs, reduc_factor):
    delta_time = (timetaggs[-1] - timetaggs[0])//reduc_factor
    # delta_time is the bin size
    told = timetaggs[0]
    current = 0
    idx = 0
    counts = []
    index = []  # for slicing the data array in another parent function
    #timeR = timetaggs[0] + delta_time
    timeR = delta_time
    timetaggs = timetaggs - timetaggs[0]
    # basic binning in chunks of time


    for i in range(len(timetaggs)):
        current = current + 1
        if timetaggs[i] > timeR:
            timeR = timeR + delta_time
            t_elapsed = timetaggs[i] - told
            told = timetaggs[i]
            counts.append(current/t_elapsed)
            index.append(i)
            current = 0
    return counts, delta_time, index


@njit
def countRateMonitor(timetaggs,channels, channel_check,reduc_factor):
    delta_time = (timetaggs[-1] - timetaggs[0])//reduc_factor
    # delta_time is the bin size
    told = timetaggs[0]
    current = 0
    idx = 0
    counts = []
    index = []  # for slicing the data array in another parent function
    timeR = timetaggs[0] + delta_time
    timetaggs = timetaggs - timetaggs[0]
    # basic binning in chunks of time


    for i in range(len(timetaggs)):
        if channels[i] == channel_check:
            current = current + 1
            idx = idx + 1
        if timetaggs[i] > timeR:
            timeR = timeR + delta_time
            t_elapsed = timetaggs[i] - told
            told = timetaggs[i]
            counts.append(current/t_elapsed)
            index.append(idx)
            current = 0
    print("option 1: ", len(channels[channels == channel_check]))
    print("option 1: ", len(channels[channels == -14]))
    print("option 2: ", index[-1])
    return counts, delta_time, index


def find_roots(x,y):  # from some stack overflow answer
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


def analyze_count_rate_b(timetags, reduc):
    counts, delta_time, index = countRateMonitor_b(timetags, reduc)
    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

    #s3.line((np.arange(len(counts))[1:]/(1e12*len(counts)))*(timetags[-1] - timetags[0]),np.array(counts)[1:]*1e6)

    X = (np.arange(len(counts))[1:]/len(counts))
    Y = np.array(counts)[1:] * 1e6
    IDX = np.array(index[:-1])
    print(IDX[:40])
    source = ColumnDataSource(data = dict(x=X, y=Y, idx=IDX))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("idx", "@idx")
        ]
    s3 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                output_backend="webgl", tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
    s3.xaxis.axis_label = "time"
    s3.yaxis.axis_label = "count rate (MCounts/s)"
    s3.line('x','y',source = source)


    Y2 = find_roots(X,Y - 1)

    marker_source = ColumnDataSource(data = dict(x = Y2,y = np.zeros(len(Y2)) + 2))
    #print(Y2*len(X))
    glyph = Circle(x="x", y="y", size=3, line_color='red', fill_color="white", line_width=3)
    s3.add_glyph(marker_source, glyph)


    return s3, X, Y


def analyze_count_rate(timetags, channels, checkChan, reduc):
    counts, delta_time, index = countRateMonitor(timetags, channels, checkChan, reduc)
    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

    #s3.line((np.arange(len(counts))[1:]/(1e12*len(counts)))*(timetags[-1] - timetags[0]),np.array(counts)[1:]*1e6)

    X = (np.arange(len(counts))[1:]/len(counts))
    Y = np.array(counts)[1:] * 1e6
    IDX = np.array(index[:-1])
    print(IDX[:40])
    source = ColumnDataSource(data = dict(x=X, y=Y, idx=IDX))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("idx", "@idx")
        ]
    s3 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                output_backend="webgl", tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
    s3.xaxis.axis_label = "time"
    s3.yaxis.axis_label = "count rate (MCounts/s)"
    s3.line('x','y',source = source)


    Y2 = find_roots(X,Y - 1)

    marker_source = ColumnDataSource(data = dict(x = Y2,y = np.zeros(len(Y2)) + 2))
    #print(Y2*len(X))
    glyph = Circle(x="x", y="y", size=3, line_color='red', fill_color="white", line_width=3)
    s3.add_glyph(marker_source, glyph)


    return s3, X, Y



def import_ground_truth(path, cycle_number):
    files = os.listdir(path)
    date_time = files[0].split('_')[1]
    file_sequence_data = str(cycle_number) + '_' + date_time + '_CH1.yml'
    with open(os.path.join(path,file_sequence_data), 'r') as f:
        sequence_data = yaml.load(f, Loader=yaml.FullLoader)


    file_set_data = '0_' + date_time + '_params.yml'
    with open(os.path.join(path,file_set_data), 'r') as f:
        set_data = yaml.load(f, Loader=yaml.FullLoader)

    return sequence_data, set_data



def runAnalysisJit(path_, file_, delay):
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)

    while file_reader.hasData():
        # Number of events to read at once
        n_events = 200000000

        # Read at most n_events.
        # data is an instance of TimeTagStreamBuffer
        data = file_reader.getData(n_events)
        print('Size of the returned data chunk: {:d} events\n'.format(data.size))

        # these are numpy arrays
        channels = data.getChannels()
        timetags = data.getTimestamps()


        step = len(channels)//120
        # c = 0
        # for i in range(120):
        #     print('$$$$$$$$$$$$$$$$$$$$$')
        #     print(channels[c:c+17])
        #     #print(timetags[c:c + 17])
        #     c = c + step
        #     print(c)
        #     print('$$$$$$$$$$$$$$$$$$$$$')
        #     #925348

        R = 5000000
        Clocks,RecoveredClocks, dataTags, dataTagsR, dualData, countM = clockScan(channels[R:-1],timetags[R:-1],18,-5,-14,-9)

        s1, s2 = checkLocking(Clocks[0:-1:20], RecoveredClocks[0:-1:20])
        #s3,x,y = analyze_count_rate(timetags[0:-1],channels[0:-1], -14, 10000)
        s3, x, y = analyze_count_rate_b(countM, 10000)
        print("Length of tags with R: ", len(channels[0:-1]))

        # print(counts)
        # print(delta_time)
        # print("total_time: ", timetags[-1] - timetags[0])

        print("length dualData", len(dualData))
        print("length RecoveredClocks", len(RecoveredClocks))
        print("lenght of countM: ", len(countM))

        # bins = np.arange(min(dualData[126600:210400,0]),max(dualData[126600:210400,0]))
        # print(min(dualData[126600:210400,0]))
        # print(max(dualData[126600:210400, 0]))
        print("length of channels: ", len(channels))
        #Clocks, RecoveredClocks, dataTags, dataTagsR, dualData = clockScan(channels[126600:210400], timetags[126600:210400], 18, -5,-14, -9)
        print("lenght of dualData: ", len(dualData))
        dualData = dualData[434273:605300]

        sequence_data, set_data = import_ground_truth("..//DataGen//tempSave", 1)
        print(sequence_data["times_sequence"])

        #dataTags = dataTags[126600:210400]
        #print(dualData[420:600])
        # port = dualData[:,0]
        # port = port[port>200]
        # print("lenght of port", len(port))
        print('length of dataTags', len(dualData[:,0]))
        # print("Clock difference: ", np.mean(np.diff(Clocks)))
        Bins = np.linspace(0,3200000,1000)
        hist, bins = np.histogram(dualData[:,1], bins = Bins)
        #############

        np.save("x_axis",bins)
        np.save("y_axis",hist)
        source = ColumnDataSource(data=dict(
            x=bins[:-1],
            y=hist
        ))

        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")
        ]
        Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"
        s4 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                    output_backend="svg", tools=Tools,
                    active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
        # s3.xaxis.axis_label = "time"
        # s3.yaxis.axis_label = "count rate (MCounts/s)"
        s4.line('x', 'y', source=source)
        #########



        return s1, s2, s3, s4


path = "..//..//July7//"
file = "25s_.002_.044_25dB_July7_fullSet_78.125clock_0.3sFalse.1.ttbin"
#file = "22s_.002_.044_30dB_July7_fullSet_78.125clock_0.1sFalse.1.ttbin"
s1,s2, s3, s4 = runAnalysisJit(path,file,0)
show(column(s1, s2,s3, s4))
#export_svg(s1, filename="plot.svg")
