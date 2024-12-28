from shutil import copyfile
import numpy as np
import segyio
import time


def readsegy(filename_in, inline_byte, xline_byte):
    time_start = time.time()

    with segyio.open(filename_in, mode='r', iline=inline_byte, xline=xline_byte, strict=True,
                     ignore_geometry=False) as src:
        data = segyio.tools.cube(src)
        data = np.transpose(data, [0, 2, 1])
        time_end = time.time()
        print('Reading completion!')
        print('Reading time-consuming: ', '%.2f' % (time_end - time_start), 's')

    return data
    # output of readsegy: [inlines * samples * xlines]


def writesegy(filename_in, inline_byte, xline_byte, data, filename_out):
    # input of writesegy: [inlines * samples * xlines]
    time_start = time.time()
    data = np.transpose(data, [1, 2, 0])
    # transpose to: [samples * xlines * inlines]
    data = np.reshape(data, [np.size(data, 0), np.size(data, 1)*np.size(data, 2)], 'F')

    copyfile(filename_in, filename_out)
    with segyio.open(filename_out, mode='r+', iline=inline_byte, xline=xline_byte, strict=True,
                     ignore_geometry=False) as dst:
        for i in range(dst.tracecount):
            dst.trace[i] = data[:, i]
    time_end = time.time()
    print('Writing completion!')
    print('Writing time-consuming: ', '%.2f' % (time_end - time_start), 's')
