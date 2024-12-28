import numpy as np
import math
import sys


def get_padding_data_and_sliding_ind_3d(input_data, length_window, length_overlap):
    """
    input_data: DxHxW     D: z  H: y    W: x
    length_window: 窗口长度
    length_overlap: 窗口重叠长度
    stride: 滑动步长
    padding_zn: z负方向补零长度
    padding_zp: z正方向补零长度
    padding_yn: y负方向补零长度
    padding_yp: y正方向补零长度
    padding_xn: x负方向补零长度
    padding_xp: x正方向补零长度
    """

    if length_overlap > np.floor(length_window / 2):
        print('重叠长度大于滑动窗口的一半，会在一个窗口上重叠三次以上，请重新设置重叠长度！')
        sys.exit()

    stride = length_window - length_overlap

    nz = input_data.shape[0]
    ny = input_data.shape[1]
    nx = input_data.shape[2]

    padding_zn, padding_zp, sliding_times_z = get_padding_and_sliding_times(nz, length_window, length_overlap, stride)
    padding_yn, padding_yp, sliding_times_y = get_padding_and_sliding_times(ny, length_window, length_overlap, stride)
    padding_xn, padding_xp, sliding_times_x = get_padding_and_sliding_times(nx, length_window, length_overlap, stride)

    # 补零
    # padding_data = np.pad(input_data, ((padding_zn, padding_zp), (padding_yn, padding_yp), (padding_xn, padding_xp)),
    #                       'constant', constant_values=(0, 0))

    # 对称
    padding_data = np.pad(input_data, ((padding_zn, padding_zp), (padding_yn, padding_yp), (padding_xn, padding_xp)),
                          'reflect', reflect_type='odd')

    z_ind = get_sliding_ind(length_window, stride, sliding_times_z)
    y_ind = get_sliding_ind(length_window, stride, sliding_times_y)
    x_ind = get_sliding_ind(length_window, stride, sliding_times_x)

    if padding_xn == 0:
        overlap_xnw, overlap_xpw = 0, 0
    else:
        overlap_xnw, overlap_xpw = length_overlap, length_overlap

    if padding_yn == 0:
        overlap_ynw, overlap_ypw = 0, 0
    else:
        overlap_ynw, overlap_ypw = length_overlap, length_overlap

    if padding_zn == 0:
        overlap_znw, overlap_zpw = 0, 0
    else:
        overlap_znw, overlap_zpw = length_overlap, length_overlap

    window_coefficient = window_function_overlap_3d(length_window, length_window, length_window,
                                                    overlap_znw, overlap_zpw,
                                                    overlap_ynw, overlap_ypw,
                                                    overlap_xnw, overlap_xpw)
    return padding_data, x_ind, y_ind, z_ind, padding_xn, padding_yn, padding_zn, window_coefficient


def get_padding_data_and_sliding_ind_2d(input_data, length_window, length_overlap):
    """
    input_data: HxW     H: y    W: x
    length_window: 窗口长度
    length_overlap: 窗口重叠长度
    stride: 滑动步长
    padding_yn: y负方向补零长度
    padding_yp: y正方向补零长度
    padding_xn: x负方向补零长度
    padding_xp: x正方向补零长度
    """

    if length_overlap > np.floor(length_window / 2):
        print('重叠长度大于滑动窗口的一半，会在一个窗口上重叠三次以上，请重新设置重叠长度！')
        sys.exit()

    stride = length_window - length_overlap

    ny = input_data.shape[0]
    nx = input_data.shape[1]

    padding_yn, padding_yp, sliding_times_y = get_padding_and_sliding_times(ny, length_window, length_overlap, stride)
    padding_xn, padding_xp, sliding_times_x = get_padding_and_sliding_times(nx, length_window, length_overlap, stride)

    # 补零
    # padding_data = np.pad(input_data, ((padding_yn, padding_yp), (padding_xn, padding_xp)),
    #                       'constant', constant_values=(0, 0))

    # 对称
    padding_data = np.pad(input_data, ((padding_yn, padding_yp), (padding_xn, padding_xp)),
                          'reflect', reflect_type='odd')

    y_ind = get_sliding_ind(length_window, stride, sliding_times_y)
    x_ind = get_sliding_ind(length_window, stride, sliding_times_x)

    if padding_xn == 0:
        overlap_xnw, overlap_xpw = 0, 0
    else:
        overlap_xnw, overlap_xpw = length_overlap, length_overlap

    if padding_yn == 0:
        overlap_ynw, overlap_ypw = 0, 0
    else:
        overlap_ynw, overlap_ypw = length_overlap, length_overlap

    window_coefficient = window_function_overlap_2d(length_window, length_window,
                                                    overlap_ynw, overlap_ypw,
                                                    overlap_xnw, overlap_xpw)
    return padding_data, x_ind, y_ind, padding_xn, padding_yn, window_coefficient


def window_function_overlap_3d(nz, ny, nx, length_znw, length_zpw, length_ynw, length_ypw, length_xnw, length_xpw):
    """
    计算三维窗函数
    维度: DxHxW     D: z  H: y    W: x
    nz: z方向总点数
    ny: y方向总点数
    nx: x方向总点数
    length_znw: z负方向重叠窗口长度
    length_zpw: z正方向重叠窗口长度
    length_ynw: y负方向重叠窗口长度
    length_ypw: y正方向重叠窗口长度
    length_xnw: x负方向重叠窗口长度
    length_xpw: x正方向重叠窗口长度
    """

    kernel = np.zeros([nz, ny, nx])

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # x < xnw, y < ynw, z < znw
                if ((ix + 1) <= length_xnw) and ((iy + 1) <= length_ynw) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x < xnw, y < ynw, znw < z < nz - zpw
                elif ((ix + 1) <= length_xnw) and ((iy + 1) <= length_ynw) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         1

                # x < xnw, y < ynw, z > nz - zpw
                elif ((ix + 1) <= length_xnw) and ((iy + 1) <= length_ynw) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # x < xnw, ynw < y < ny - ypw, z < znw
                elif ((ix + 1) <= length_xnw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         1 * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x < xnw, ynw < y < ny - ypw, znw < z < nz - zpw
                elif ((ix + 1) <= length_xnw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         1 * \
                                         1

                # x < xnw, ynw < y < ny - ypw, z > nz - zpw
                elif ((ix + 1) <= length_xnw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         1 * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # x < xnw, y > ny - ypw, z < znw
                elif ((ix + 1) <= length_xnw) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x < xnw, y > ny - ypw, znw < z < nz - zpw
                elif ((ix + 1) <= length_xnw) and ((iy + 1) > (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         1

                # x < xnw, y > ny - ypw, z > nz - zpw
                elif ((ix + 1) <= length_xnw) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # xnw < x < nx - xpw, y < ynw, z < znw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) <= length_ynw) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = 1 * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # xnw < x < nx - xpw, y < ynw, znw < z < nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) <= length_ynw) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         1

                # xnw < x < nx - xpw, y < ynw, z > nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) <= length_ynw) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # xnw < x < nx - xpw, ynw < y < ny - ypw, z < znw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = 1 * \
                                         1 * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # xnw < x < nx - xpw, ynw < y < ny - ypw, znw < z < nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         1 * \
                                         1

                # xnw < x < nx - xpw, ynw < y < ny - ypw, z > nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         1 * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # xnw < x < nx - xpw, y > ny - ypw, z < znw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = 1 * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # xnw < x < nx - xpw, y > ny - ypw, znw < z < nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) > (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         1

                # xnw < x < nx - xpw, y > ny - ypw, z > nz - zpw
                elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = 1 * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # x > nx - xpw, y < ynw, z < znw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) <= length_ynw) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x > nx - xpw, y < ynw, znw < z < nz - zpw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) <= length_ynw) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         1

                # x > nx - xpw, y < ynw, z > nz - zpw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) <= length_ynw) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # x > nx - xpw, ynw < y < ny - ypw, z < znw
                elif (ix + 1) > (nx - length_xpw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         1 * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x > nx - xpw, ynw < y < ny - ypw, znw < z < nz - zpw
                elif (ix + 1) > (nx - length_xpw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         1 * \
                                         1

                # x > nx - xpw, ynw < y < ny - ypw, z > nz - zpw
                elif (ix + 1) > (nx - length_xpw) and (length_ynw < (iy + 1) <= (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         1 * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))

                # x > nx - xpw, y > ny - ypw, z < znw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) <= length_znw):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (1 / (length_znw - 1) * (iz + 1) - 1 / (length_znw - 1))

                # x > nx - xpw, y > ny - ypw, znw < z < nz - zpw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) > (ny - length_ypw)) and (length_znw < (iz + 1) <= (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         1

                # x > nx - xpw, y > ny - ypw, z > nz - zpw
                elif (ix + 1) > (nx - length_xpw) and ((iy + 1) > (ny - length_ypw)) and ((iz + 1) > (nz - length_zpw)):
                    kernel[iz][iy][ix] = (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                         (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw)) * \
                                         (-1 / (length_zpw - 1) * (iz + 1) - nz / (1 - length_zpw))
    return kernel


def window_function_overlap_2d(ny, nx, length_ynw, length_ypw, length_xnw, length_xpw):
    """
    计算二维窗函数
    维度: HxW     H: y    W: x
    ny: y方向总点数
    nx: x方向总点数
    length_ynw: y负方向重叠窗口长度
    length_ypw: y正方向重叠窗口长度
    length_xnw: x负方向重叠窗口长度
    length_xpw: x正方向重叠窗口长度
    """

    kernel = np.zeros([ny, nx])

    for iy in range(ny):
        for ix in range(nx):
            # x < xnw, y < ynw
            if ((ix + 1) <= length_xnw) and ((iy + 1) <= length_ynw):
                kernel[iy][ix] = 1 * (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                 (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1))

            # x < xnw, ynw < y < ny - ypw
            elif ((ix + 1) <= length_xnw) and (length_ynw < (iy + 1) <= (ny - length_ypw)):
                kernel[iy][ix] = 1 * (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1))

            # x < xnw, y > ny - ypw
            elif ((ix + 1) <= length_xnw) and ((iy + 1) > (ny - length_ypw)):
                kernel[iy][ix] = 1 * (1 / (length_xnw - 1) * (ix + 1) - 1 / (length_xnw - 1)) * \
                                 (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw))

            # xnw < x < nx - xpw, y < ynw
            elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) <= length_ynw):
                kernel[iy][ix] = 1 * (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1))

            # xnw < x < nx - xpw, ynw < y < ny - ypw
            elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and (length_ynw < (iy + 1) <= (ny - length_ypw)):
                kernel[iy][ix] = 1

            # xnw < x < nx - xpw, y > ny - ypw
            elif (length_xnw < (ix + 1) <= (nx - length_xpw)) and ((iy + 1) > (ny - length_ypw)):
                kernel[iy][ix] = 1 * (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw))

            # x > nx - xpw, y < ynw
            elif (ix + 1) > (nx - length_xpw) and ((iy + 1) <= length_ynw):
                kernel[iy][ix] = 1 * (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                 (1 / (length_ynw - 1) * (iy + 1) - 1 / (length_ynw - 1))

            # x > nx - xpw, ynw < y < ny - ypw
            elif (ix + 1) > (nx - length_xpw) and (length_ynw < (iy + 1) <= (ny - length_ypw)):
                kernel[iy][ix] = 1 * (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw))

            # x > nx - xpw, y > ny - ypw
            elif (ix + 1) > (nx - length_xpw) and ((iy + 1) > (ny - length_ypw)):
                kernel[iy][ix] = 1 * (-1 / (length_xpw - 1) * (ix + 1) - nx / (1 - length_xpw)) * \
                                 (-1 / (length_ypw - 1) * (iy + 1) - ny / (1 - length_ypw))
    return kernel


def get_sliding_ind(length_window, stride, sliding_times):
    """
    input
    length_window: 窗口长度
    stride: 滑窗步长
    sliding_times: 滑窗次数

    output
    sliding_ind: 每次滑动窗口的起始坐标和结尾坐标
    """

    sliding_ind = np.zeros([sliding_times, 2], dtype=int)

    for i_SlidingTimes in range(sliding_times):
        window_start = stride * i_SlidingTimes
        window_end = length_window + stride * i_SlidingTimes - 1

        sliding_ind[i_SlidingTimes][0] = window_start
        sliding_ind[i_SlidingTimes][1] = window_end
    return sliding_ind


def get_padding_and_sliding_times(n, length_window, length_overlap, stride):
    """
    input
    n: 总点数
    length_window: 窗口长度
    length_overlap: 重叠长度
    stride: 滑动步长

    output
    padding_n: 负方向补零数
    padding_p: 正方向补零数
    sliding_times: 滑窗总次数
    """

    if n < length_window:
        padding_n = 0
        padding_p = length_window - n
        sliding_times = 1
    elif n == length_window:
        padding_n, padding_p = 0, 0
        sliding_times = 1
    else:
        padding_n = length_overlap
        sliding_times = math.ceil((padding_n + n - length_overlap) / stride) + 1        # 最后一次窗口的起始坐标要在正方向的overlap之间，才能保证数据正方向的边缘不会被衰减
        padding_p = length_window - 1 + stride * (sliding_times - 1) - (padding_n + n - 1)
    return padding_n, padding_p, sliding_times

