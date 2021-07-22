

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
    #print(IDX[:40])
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

    section_list = generate_section_list(X, Y, Y2,IDX)


    return s3, section_list


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





@njit
def jit_convolve(array1,array2):
    q = np.zeros(len(array1))
    for i in range(len(array1)):
        q[i] = np.dot(array1,array2)
        array1 = np.roll(array1,-1)
    return np.arange(len(q)), q

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


def make_ground_truth_hist(ground_truth_path,clock_period, sequence,resolution = 1000):
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    times = np.array(
        sequence_data["times"]) * 1e12 + 10000  # adding on 10ns so that redefined clock is 10ns before first data
    bins = np.linspace(0, clock_period, resolution)
    ground_truth_hist, bins = np.histogram(times, bins=bins)
    return ground_truth_hist, bins

def find_rough_offset(data,sequence,ground_truth_path,clock_period,resolution = 1000):
    real_data_bins = np.linspace(0, clock_period, resolution)
    real_data_hist, bins = np.histogram(data, bins=real_data_bins)
    sequence_data, set_data = import_ground_truth(ground_truth_path, sequence)
    times = np.array(
        sequence_data["times"]) * 1e12 + 10000  # adding on 10ns so that redefined clock is 10ns before first data
    ground_truth_hist, bins = np.histogram(times, bins=bins)
    x,y = jit_convolve(ground_truth_hist.astype(float),real_data_hist.astype(float))
    return -(y.argmax()/resolution)*clock_period

def offset_tags(dual_data,offset,clock_period):
    """
    Change the clock reference from which the data in dual_data is measured from. When the time offset sends
    some timetaggs outside of the range from 0 to [clock_period in ps], then those tags are 'rolled over'
    to the previous or successive group of tags with with a single clock reference.
    """
    dual_data = dual_data - offset
    greater_than_mask = (dual_data[:,0] > clock_period)
    less_than_mask = (dual_data[:,0] < 0)
    dual_data[greater_than_mask] = dual_data[greater_than_mask] - clock_period
    dual_data[less_than_mask] = dual_data[less_than_mask] + clock_period
    return dual_data


def generate_section_list(x,y,x_intercepts,idx):
    """
    Takes in a list of x_intercepts and the x and y axese of the count rate vs time plot on which they were found.
    A list is generated where reach row has an index that is near the beginning of a high count rate region, and has
    an index that is near the end of that region.
    """
    right_old = 0
    section_list = np.zeros((len(x_intercepts),2),dtype='int64') - 1
    q = 0
    threshold = np.mean(y)
    for intercept in x_intercepts:
        left = np.argmax(x > intercept) - 3
        right = np.argmax(x > intercept) + 3
        # print("value at left side: ", y[left])
        # print("value at right side: ", y[right])
        if y[right_old] > threshold and y[left] > threshold:
            # add that pair to the section list
            section_list[q,0] = right_old
            section_list[q, 1] = left
            q = q + 1
        right_old = right

    section_list = section_list[section_list[:,0] >= 0]
    print("identified ", len(section_list), " sections of high count rate.")

    # convert to dualData scaling from plot scaling
    for row in section_list:
        row[0] = idx[row[0]]
        row[1] = idx[row[1]]

    return section_list


def generate_PNR_analysis_regions(dual_data, cycle_number,clock_period, gt_path):
    bins = np.linspace(0, clock_period, 5000000)
    final_hist, final_bins = np.histogram(dual_data[:, 1], bins)
    gthist, bins = make_ground_truth_hist(gt_path, clock_period, cycle_number,
                                          resolution=5000000)
    plt.figure()
    plt.plot(bins[1:], gthist * 20000)
    plt.plot(final_bins[1:], final_hist)

    import_ground_truth(gt_path, cycle_number)



def runAnalysisJit(path_, file_, gt_path):
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


        R = 5000000
        # gt_path = "..//DataGen///TempSave//"
        Clocks,RecoveredClocks, dataTags, dataTagsR, dualData, countM = clockScan(channels[R:-1],timetags[R:-1],18,-5,-14,-9)

        s1, s2 = checkLocking(Clocks[0:-1:20], RecoveredClocks[0:-1:20])
        # s3,x,y = analyze_count_rate(timetags[0:-1],channels[0:-1], -14, 10000)

        # identify sections in the file where the AWG is sending a sequence.
        s3, section_list = analyze_count_rate_b(countM, 10000)

        print("Length of tags with R: ", len(channels[0:-1]))
        #section_list = generate_section_list(x, y, x_intercepts)
        print(section_list)
        # loop over the sequence sent by AWG

        current_number = 20
        for i, section in enumerate(section_list):
            if i != current_number: # this is just used for now so I only look at one section
                continue
            # grab the current section of interest from the large array dualData
            m_data = dualData[section[0]:section[1]]
            CLOCK_PERIOD = 3200000
            # will input datatags that correspond only to individual sequences
            offset = find_rough_offset(m_data,current_number+5,gt_path,CLOCK_PERIOD,resolution = 10000)

            m_data = offset_tags(m_data, offset,CLOCK_PERIOD)

            generate_PNR_analysis_regions(m_data, current_number+5, CLOCK_PERIOD,gt_path)





            #############

            real_data_bins = np.linspace(0, 3200000, 10000)
            real_data_hist, bins = np.histogram(m_data[:, 1], bins=real_data_bins)
            source = ColumnDataSource(data=dict(
                x=bins[:-1],
                y=real_data_hist
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
gt_path = "..//DataGen///TempSave//"
s1,s2, s3, s4 = runAnalysisJit(path, file, gt_path)
show(column(s1, s2, s3, s4))
