import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from bokeh.io import output_file, show
from bokeh.plotting import figure, show, curdoc
from bokeh.models import Slider, CustomJS, ColumnDataSource
from bokeh.layouts import column

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

# def make_dummy_gt(times,cycle_time):
#
#     #output "timetaggs" in picosecons



x = np.load("x_axis.npy")
y = np.load("y_axis.npy")




sequence_data, set_data = import_ground_truth("..//DataGen///TempSave//", 28)
print(sequence_data["times_sequence"])
print(set_data["pulses_per_cycle"])
times = np.array(sequence_data["times"])*1e12 + 10000 # adding on 10ns so that redefined clock is 10ns before first data

print(sequence_data["times"])
histgt,binsgt = np.histogram(times,bins = x)

plt.plot(x[1:],histgt*20000)





# source = ColumnDataSource(data=dict(
#             x=x[:-1],
#             data=y,
#             gt=histgt*20000,
#         ))

TOOLTIPS = [
("index", "$index"),
("(x,data)", "($x, $data)"),
("(x,gt)", "($x, $gt)"),
]
Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"
s1 = figure(plot_width=800, plot_height=400, title="count rate monitor",
                    output_backend="svg", tools=Tools,
                    active_scroll='xwheel_zoom', tooltips=TOOLTIPS)
        # s3.xaxis.axis_label = "time"
        # s3.yaxis.axis_label = "count rate (MCounts/s)"
s1.line(x[1:], y)
p = s1.line(x[1:], histgt*20000,color = 'red')


slider = Slider(title = 'offset', start = -10000, end = 10000, value = 0, step = 10)


def update(attr, old, new):
    s = slider.value  # slope.
    #print(slider_slope.value)
    # i = slider_intercept.value  # intercept
    p.data_source.data['x'] = np.roll(x[1:], -s)
    # p.data_source.data['y'] = p.data_source.data['y']



slider.on_change('value', update)


# output_file("layout.html")
layout = column(s1, slider)
curdoc().add_root(layout)


