

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

Colors2, pallet2 = viz.phd_style(grid=True)




def checkLocking(Clocks, RecoveredClocks,mpl = False):

    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    s = 12800000/1e12
    print("s is this: ", s)
    print(len(diffs))
    x = np.arange(0, len(diffs))*s

    print(np.max(x))

    if mpl:
        plt.figure()
        plt.plot(x,diffs)
        plt.plot(x,diffsRecovered)
        return


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
    print("legnth of passed array: ", len(timetaggs))
    print("first in array: ", timetaggs[0])
    for i in range(len(timetaggs)):
        if timetaggs[i] > 0:
            break
    for j in range(1,len(timetaggs)):
        if timetaggs[-j] > 0:
            break
    delta_time = (timetaggs[-j] - timetaggs[i])//reduc_factor
    # print("original time: ", timetaggs[-j] - timetaggs[i])
    # print("this is i: ", i)
    # print("this is j: ", j)
    # print("delta time: ", delta_time)
    # print("ending time: ", timetaggs[-j])
    # print("beginning time: ", timetaggs[i])
    # delta_time is the bin size
    counts = []
    index = []  # for slicing the data array in another parent function
    #timeR = timetaggs[0] + delta_time
    timetaggs = timetaggs #- timetaggs[i] + 1
    # basic binning in chunks of time
    print(timetaggs[10000:10010])
    times = []
    q = 0
    zero_counter = 0
    bla = 0
    rm_region = np.zeros(reduc_factor)
    for u in range(0, reduc_factor):
        current_time = delta_time*u + timetaggs[i]
        current_sum = 0
        while timetaggs[q] < current_time:
            q = q + 1
            if timetaggs[q] != 0:
                current_sum = current_sum + 1
                zero_counter = 0
            else:
                bla = bla + 1
                zero_counter = zero_counter + 1
        if zero_counter > 3:
            rm_region[u] = 1
        times.append(current_time)
        counts.append(current_sum/delta_time)
        index.append(q)
    print("this is bla: ", bla)
    #return counts, delta_time, index
    return counts, times, index, rm_region


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
    counts, times, index, rm_region = countRateMonitor_b(timetags, reduc)
    counts = np.array(counts)
    times = np.array(times)
    index = np.array(index)

    # cut out the long regions of zeros for easier plotting
    print("sum of zero counter: ", np.sum(rm_region))
    mask = np.invert(rm_region.astype(bool))
    times = times[mask]
    counts = counts[mask]
    index = index[mask]
    print("this is length of counts: ", len(counts))
    plt.figure()
    plt.plot(np.arange(len(rm_region)),rm_region)



    Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

    #s3.line((np.arange(len(counts))[1:]/(1e12*len(counts)))*(timetags[-1] - timetags[0]),np.array(counts)[1:]*1e6)

    X = np.array(times)  # (np.arange(len(counts))[1:]/len(counts))
    Y = np.array(counts) * 1e6
    IDX = np.array(index)
    #print(IDX[:40])
    source = ColumnDataSource(data = dict(x=X, y=Y, idx=IDX))

    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("idx", "@idx")
        ]
    s3 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                #output_backend="webgl",
                tools=Tools,
                active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
    s3.xaxis.axis_label = "time"
    s3.yaxis.axis_label = "count rate (MCounts/s)"
    s3.line('x','y',source = source)

    #mean = np.mean(Y)
    mean = 3000000
    print("the mean is: ", mean)
    Y2 = find_roots(X,Y - mean)

    marker_source = ColumnDataSource(data = dict(x = Y2,y = np.zeros(len(Y2)) + mean))
    #print(Y2*len(X))
    glyph = Circle(x="x", y="y", size=3, line_color='red', fill_color="white", line_width=3)
    s3.add_glyph(marker_source, glyph)

    #section_list = generate_section_list(X, Y, Y2,IDX)
    section_list = []

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


    Y2 = find_roots(X,Y - 0.5)

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

@njit
def jit_convolve_limited(array1,array2, start,end):
    array1 = np.roll(array1,start)
    r = end - start
    q = np.zeros(r)
    for i in range(r):
        q[i] = np.dot(array1,array2)
        array1 = np.roll(array1,1)
    return np.arange(len(q)) - start, q

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
    plt.plot(bins[1:], gthist * 200)
    plt.plot(final_bins[1:], final_hist)

    sequence_data, set_data  = import_ground_truth(gt_path, cycle_number)
    times = np.array(
        sequence_data["times"]) * 1e12 + 10000  # adding on 10ns so that redefined clock is 10ns before first data
    print(times)

    times_left = times - 1000
    times_right = times + 1000  # 400ps window over which to do the calibration
    #sequence_counts = np.array([])
    for i, (left, right, current_time) in enumerate(zip(times_left,times_right,times)):
        mask = (dual_data[:,0] > left) & (dual_data[:,0] < right)
        counts = dual_data[mask] - current_time  # bring the counts to to near-zero
        l_bound = left - current_time
        r_bound = right - current_time
        print(l_bound)
        if i == 0:
            sequence_counts = counts
        else:
            sequence_counts = np.concatenate((sequence_counts,counts))





    bins = np.arange(int(l_bound),int(r_bound))

    hist, bins = np.histogram(sequence_counts[:,0],bins = bins)
    hist2, bins = np.histogram(sequence_counts[:, 1], bins=bins)
    plt.figure()
    plt.plot(bins[1:],hist)
    plt.plot(bins[1:], hist2)



    return sequence_counts


def find_pnr_correction(counts):
    # I might have to do some data cleaning here?
    slices = np.arange(0, 2000, 4)
    corr1 = []
    corr2 = []
    for i in range(len(slices) - 1):
        left_bound = slices[i]
        right_bound = slices[i + 1]
        delta_t = counts[:, 1] - counts[:, 0]
        mask = (delta_t > left_bound) & (delta_t <= right_bound)
        corr1.append(np.mean(counts[:, 0][mask]))
        corr2.append(np.mean(counts[:, 1][mask]))

    corr1 = np.array(corr1)
    corr2 = np.array(corr2)

    # fill in NaNs
    mask = np.isnan(corr1)
    corr1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), corr1[~mask])

    mask = np.isnan(corr2)
    corr2[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), corr2[~mask])

    return slices, corr1, corr2


def viz_counts_and_correction(counts,slices,corr1,corr2):

    bins = np.arange(-1000,1000)
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    ax[0].hist2d(counts[:, 1] - counts[:, 0], counts[:, 0], bins=(bins, bins),
                 norm=matplotlib.colors.LogNorm())
    ax[0].plot(slices[:-1], corr1, color='k')

    ax[1].hist2d(counts[:, 1] - counts[:, 0], counts[:, 1], bins=(bins, bins),
                 norm=matplotlib.colors.LogNorm())
    ax[1].plot(slices[:-1], corr2, color='k')



def apply_pnr_correction(dual_data, slices, corr1, corr2,seperated_arrays = False):
    corrected1 = np.zeros(len(dual_data))
    corrected2 = np.zeros(len(dual_data))
    array_list1 = []
    array_list2 = []
    for i in range(len(slices) - 1):
        left_bound = slices[i]
        right_bound = slices[i + 1]
        delta_t = dual_data[:,1] - dual_data[:,0]
        mask = (delta_t > left_bound) & (delta_t <= right_bound)
        a = dual_data[:,0][mask] - corr1[i]
        b = dual_data[:, 1][mask] - corr2[i]
        corrected1[mask] = a
        corrected2[mask] = b
        if seperated_arrays:
            array_list1.append(a)
            array_list2.append(b)

    if seperated_arrays:
        return corrected1, corrected2, array_list1, array_list2
    else:
        return corrected1, corrected2

def viz_correction_effect(sequence_counts, slices, corr1, corr2,hist_set = False):
    if hist_set:
        corrected1, corrected2, array_list1, array_list2 = apply_pnr_correction(sequence_counts, slices, corr1, corr2,
                                                                            seperated_arrays=True)
    else:
        corrected1, corrected2 = apply_pnr_correction(sequence_counts, slices, corr1, corr2,
                                                                            seperated_arrays=False)

    plt.figure()
    bins = np.arange(sequence_counts.min(), sequence_counts.max())
    hist, bins = np.histogram(corrected1, bins=bins)
    hist2, bins = np.histogram(corrected2, bins=bins)
    plt.plot(bins[1:], hist)
    plt.plot(bins[1:], hist2)

    m = (corrected1 <= 25) & (corrected1 > -25)
    print("ratio: ", m.sum() / len(corrected1))

    if hist_set:
        plt.figure()
        i = 0
        for item in array_list1:
            if len(item) > 100:
                i = i + 1
                bins = np.arange(item.min(), item.max())
                hist, bins = np.histogram(item, bins=bins, density = True)
                plt.plot(bins[1:], hist)

        print("iterations: ", i)



def accurate_delay_scan(m_data_corrected,gt_path,sequence,slot_width,clock_period):
    sequence_data, set_data = import_ground_truth(gt_path, sequence)
    times = np.array(
        sequence_data["times"]) * 1e12 + 10000 # adding 10 ns

    # time_ranges = []
    #
    # sums = []
    # for j in range(-600,600):
    #     m = False
    #     for i,time in enumerate(times):
    #         time_ranges.append([time - slot_width/2,time + slot_width/2])
    #         right = time - slot_width/2
    #         left = time + slot_width/2
    #         m = ((m_data_corrected > right) & (m_data_corrected < left))
    #         this = 0
    #     sums.append(m.sum())
    # plt.figure()
    # plt.plot(np.arange(len(sums)),sums)


    kernel = np.zeros(clock_period + 10000)
    for time in times:
        right = int(time - slot_width / 2)
        left = int(time + slot_width / 2)
        kernel[right:left] = 1

    hist,bins = np.histogram(m_data_corrected,bins = np.arange(clock_period + 10001))
    print("lenght of kernel", len(kernel))
    print("length of hist: ", len(hist))

    plt.figure()
    plt.plot(bins[1:],hist)
    plt.plot(bins[1:],kernel*700)


    x,y = jit_convolve_limited(hist.astype(float), kernel.astype(float), -200, 200)
    plt.figure()
    plt.plot(x,y)
    print("max of y: ", np.argmax(y))




def decode_ppm(m_data_corrected, gt_path, sequence):
    sequence_data, set_data = import_ground_truth(gt_path, sequence)




    print(sequence_data["times_sequence"])
    times = np.array(sequence_data["times"])
    print(times*1e12 + 10000)

    print(set_data['pulses_per_cycle'])
    laser_time = set_data['laser_time']

    pulses_list = sequence_data["times_sequence"]
    pulses_per_cycle = set_data['pulses_per_cycle']
    initial_time = 10000 * 1e-12  # 10 ns
    generated_times = []
    base_times = []
    for pulse in pulses_list:
        time = initial_time + pulse*laser_time
        base_times.append(initial_time)
        initial_time = initial_time + pulses_per_cycle*laser_time
        generated_times.append(time*1e12)

    #print(generated_times)


    section_list = np.split(m_data_corrected, np.where(np.diff(m_data_corrected) <= 0)[0] + 1)
    print(len(section_list))
    print(section_list[0])
    print(section_list[234])
    print(section_list[2455])
    print(section_list[8000])


    # for base_time in base_times:
    #     arr =


    # need to be able to identify a number for each PPM symbol.
    # might also be nice to compare fidelity of the same symbol over multiple sequences. To see if dead
    # time is playing a role.



# @njit
# def


'''
@njit
def section_list_manager(section_list):

    #for

    section_results = []
    section_result = np.zeros(len(GROUND TRUTH LENGTH))
    for section in section_list:
        for tag in section:
            if tag > symbol_left[i] and tag < symbol_right[i]:
                # do something with it
            else:
                # go onto next tag

        section_results.append(section_result)
'''


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


        R = 2400000
        # gt_path = "..//DataGen///TempSave//"
        # print("some channels")
        # print(channels[39099+100000:39099+200+100000])
        Clocks,RecoveredClocks, dataTags, dataTagsR, dualData, countM, dirtyClock = clockScan(channels[R:-1],timetags[R:-1],18,-5,-14,9, clock_mult=4)

        # print("max of dualData: ", np.max(dualData[:,0]))
        #
        #
        # print("some tags from count: ", dualData[0:100,0])
        # print("some tags from count: ", dualData[-100:-1,0])
        # print("dirty clock: ", np.where(dirtyClock != 0))

        # print(len(dirtyClock))
        # print(len(dualData))

        s1, s2 = checkLocking(Clocks[0:-1:20], RecoveredClocks[0:-1:20])

        checkLocking(Clocks[0:-1], RecoveredClocks[0:-1],mpl = True)
        # s3,x,y = analyze_count_rate(timetags[0:-1],channels[0:-1], -14, 10000)

        # identify sections in the file where the AWG is sending a sequence.
        # print("length of dirtyClock: ", len(dirtyClock))
        l = len(dirtyClock)
        s3, section_list = analyze_count_rate_b(dirtyClock, 200000)

        # print("time of 757: ", dirtyClock[7573822])
        # print("time of 8000: ", dirtyClock[8000000])
        # print("e time: ", dirtyClock[8000000] - dirtyClock[7573822])
        # print(section_list)
        # print("Length of tags with R: ", len(channels[0:-1]))



        # loop over the sequence sent by AWG
        ###################################################
        calibrate_number = 1
        sequence_offset = 6
        SEQ = calibrate_number + sequence_offset
        ###################################################

        return s1, s2, s3

        calibrate_section = section_list[calibrate_number]
        m_data = dualData[calibrate_section[0]:calibrate_section[1]]
        #print("shape of m_data: ", np.shape(m_data))
        CLOCK_PERIOD = 3200000
        # will input datatags that correspond only to individual sequences
        offset = find_rough_offset(m_data, SEQ, gt_path, CLOCK_PERIOD, resolution=10000)

        m_data = offset_tags(m_data, offset, CLOCK_PERIOD)

        sequence_counts = generate_PNR_analysis_regions(m_data, SEQ, CLOCK_PERIOD,
                                                        gt_path)

        slices, corr1, corr2 = find_pnr_correction(sequence_counts)
        viz_counts_and_correction(sequence_counts, slices, corr1, corr2)
        viz_correction_effect(sequence_counts, slices, corr1, corr2)
        m_data_corrected, _ = apply_pnr_correction(m_data, slices, corr1, corr2)
        # now do an accurate scan

        # I don't think accurate_delay_scan is necessary based on how the calibraton removes small delays.
        #accurate_delay_scan(m_data_corrected,gt_path,calibrate_number+sequence_offset,50,CLOCK_PERIOD)
        #test, _ = apply_pnr_correction(m_data, slices, corr1, corr2)

        # hist, bins = np.histogram(test,bins = np.arange(0,3200000))
        # plt.figure()
        # plt.plot(bins[1:],hist)

        decode_ppm(m_data_corrected, gt_path,SEQ)


        for i, section in enumerate(section_list[1:]):
            # stuff for capturing data
            continue









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


path = "..//..//July27//"
path = "..//..//July29//"
#file = "25s_.002_.044_25dB_July7_fullSet_78.125clock_0.3sFalse.1.ttbin"

#file = "335s_.002_.053_30dB_july27_PICFalse.1.ttbin"
#file = "335s_.002_.053_20dB_july27_PICFalse.1.ttbin"
#file = "335s_.002_.053_40dB_july27_PICFalse.1.ttbin"
file = "340s_.002_.050_july29_picScan_16.0.1.ttbin"
gt_path = "C://Users//Andrew//Desktop//tempImgSave//" # "..//DataGen///TempSave//"
#s1,s2, s3, s4 = runAnalysisJit(path, file, gt_path)
s1,s2, s3 = runAnalysisJit(path, file, gt_path)
#show(column(s1, s2, s3, s4))
show(column(s1, s2, s3))
