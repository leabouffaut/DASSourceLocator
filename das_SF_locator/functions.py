import scipy.signal as sp
from tqdm import tqdm
from . import simpleDASreader as sd

import csv
import das4whales as dw
import h5py
from datetime import datetime

def load_ASN_DAS_file(filepath):
    # Load file with the correct sensitivity values
    dfdas = sd.load_DAS_files(filepath, chIndex=None, samples=None, sensitivitySelect=-3,
                              userSensitivity={'sensitivity': 9.36221e6, 'sensitivityUnit': 'rad/(m*strain)'},
                              integrate=True, unwr=True)
    tr = dfdas.values.T

    # Get the metadata
    fs = 1 / dfdas.meta['dt']  # sampling rate in Hz
    ns = dfdas.meta['dimensionRanges']['dimension0']['size']  # number time samples
    nx = dfdas.meta['dimensionRanges']['dimension1']['size'][0]  # number of channels
    dx = 4 * dfdas.meta['dx']  # channel spacing in m
    GL = dx  # gauge length in m

    metadata = {'fs': fs, 'dx': dx, 'ns': ns, 'GL': GL, 'nx': nx}

    # For future save
    fileBeginTimeUTC = dfdas.meta['time']

    return tr, fileBeginTimeUTC, metadata
def get_svalbard_acquisition_parameters(filename):
    """
    Gets DAS acquisition parameters

    Inputs:
    :param filename: a string containing the full path to the data to load

    Outputs:
    :return: fs: the sampling frequency (Hz)
    :return: dx: interval between two virtual sensing points also called channel spacing (m)
    :return: nx: the number of spatial samples also called channels
    :return: ns: the number of time samples
    :return: gauge_length: the gauge length (m)
    :return: scale_factor: the value to convert DAS data from strain rate to strain

    """

    fp = h5py.File(filename, 'r')

    # OptoDAS
    fs = 1/fp['header']['dt'][()]  # sampling rate in Hz
    dx = fp['header']['dx'][()]*fp['demodSpec']['roiDec'][()]  # channel spacing in m
    dx = dx[0]
    nx = fp['header']['dimensionRanges']['dimension1']['size'][()] # number of channels
    nx = nx[0]
    ns = fp['header']['dimensionRanges']['dimension0']['size'][()]  # number of samples
    gauge_length = fp['header']['gaugeLength'][()]  # gauge length in m
    n = fp['cableSpec']['refractiveIndex'][()]  # refractive index of the fiber
    scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * gauge_length)

    metadata = {'fs': fs, 'dx': dx, 'ns': ns, 'GL': gauge_length, 'nx': nx, 'scale_factor': scale_factor}


    return metadata
def load_svalbard_das_data(filename, selected_channels, metadata):
    """
    Load the DAS data corresponding to the input file name as strain according to the selected channels.

    Inputs:
    :param filename: a string containing the full path to the data to load
    :param fs: the sampling frequency (Hz)
    :param dx: interval between two virtual sensing points also called channel spacing (m)
    :param selected_channels:
    :param scale_factor: the value to convert DAS data from strain rate to strain

    Outputs:
    :return: trace: a [channel x sample] nparray containing the strain data
    :return: tx: the corresponding time axis (s)
    :return: dist: the corresponding distance along the FO cable axis (m)
    :return: file_begin_time_utc: the beginning time of the file, can be printed using
    file_begin_time_utc.strftime("%Y-%m-%d %H:%M:%S")
    """
    # Load file with the correct sensitivity values
    dfdas = sd.load_DAS_files(filename, chIndex=None, samples=None,  # sensitivitySelect=-3,
                              # userSensitivity={'sensitivity': 9.36221e6, 'sensitivityUnit': 'rad/(m*strain)'},
                              integrate=True, unwr=True)
    trace = dfdas.values.T
    trace = trace[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)

    # For future save
    file_begin_time_utc = dfdas.meta['time']

    # Store the following as the dimensions of our data block
    nnx = trace.shape[0]
    nns = trace.shape[1]

    # Define new time and distance axes
    tx = np.arange(nns) / metadata['fs']
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata['dx']

    return trace, tx, dist, file_begin_time_utc

def get_metadata(interrogator, file):
    if interrogator == 'optasense':
        # 1) Get metadata
        metadata = dw.data_handle.get_acquisition_parameters(file, interrogator='optasense')

    elif interrogator == 'optodas':
        # 1) Get metadata
        metadata = get_svalbard_acquisition_parameters(file)

    elif interrogator == 'optodas_old':
        # 1) Metadata and file
        # Load file, time and metadata
        tr, fileBeginTimeUTC, metadata = load_ASN_DAS_file(file)
    return metadata


def load_data(interrogator, file, selected_channels, metadata):

    if interrogator == 'optasense':
        # Loads the data using the pre-defined selected channels.
        tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(file, selected_channels, metadata)

    elif interrogator == 'optodas':
        # Loads the data using the pre-defined selected channels.
        tr, time, dist, fileBeginTimeUTC = load_svalbard_das_data(file, selected_channels, metadata)

    elif interrogator == 'optodas_old':
        # 1) Metadata and file
        # Load file, time and metadata
        tr, fileBeginTimeUTC, metadata = load_ASN_DAS_file(file)

        # 3) Remove unnecessary channels and get axis
        tr = tr[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)

        # Store the following as the dimensions of our data block
        nnx = tr.shape[0]
        nns = tr.shape[1]

        # Define new time and distance axes
        time = np.arange(nns) / metadata["fs"]
        dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]

    return tr, time, dist, fileBeginTimeUTC


# Function to get the position of DAS and integrate it to the metadata
def get_lat_lon_depth(file, selected_channels, metadata):
    """
    Function to extract latitude, longitude, and depth information from a CSV or TXT file and integrate it into metadata.

    Args:
        file (str): The path to the CSV or TXT file containing latitude, longitude, and depth information.
        selected_channels (tuple): A tuple containing three integers representing the start, stop, and step values for selecting channels.
        metadata (dict): A dictionary containing metadata information, including the number of channels (nx) and channel positions.

    Returns:
        dict: A dictionary containing the extracted latitude, longitude, and depth information, integrated with the metadata.

    """
    # Prepare lists
    channel = []
    lat = []
    lon = []
    depth = []

    # Read the latitude, longitude, and depth data from the CSV/TXT file
    with open(file, mode='r') as file:
        csv_file = csv.reader(file, delimiter=' ')
        for lines in csv_file:
            # Filter out empty strings
            lines = [line for line in lines if line]
            channel.append(float(lines[0]))
            lat.append(float(lines[1]))
            lon.append(float(lines[2]))
            depth.append(abs(float(lines[3])))

    # Select the specified channels
    channel = channel[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    lat = lat[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    lon = lon[selected_channels[0]:selected_channels[1]:selected_channels[2]]
    depth = depth[selected_channels[0]:selected_channels[1]:selected_channels[2]]

    # Store latitude, longitude, and depth in a dictionary
    position = {
        'Original channel': channel,
        'Lat.': lat,
        'Lon.': lon,
        'Depth': depth,
    }

    return position


def to_rad(degree):
    """
    Convert degrees to radians.

    Args:
        degree (float): The value in degrees.

    Returns:
        float: The converted value in radians.

    """
    return degree * np.pi / 180


def get_dist_lat_lon(position_1, position_2):
    """
    Calculate the distance between two sets of latitude and longitude positions using the Haversine formula.

    Args:
        position_1 (dict): A dictionary containing latitude and longitude positions for the first set of locations.
        position_2 (dict): A dictionary containing latitude and longitude positions for the second set of locations. - may contin a list of values

    Returns:
        numpy.ndarray: An array containing the distances (in meters) between corresponding positions in position_1 and position_2.

    """
    R = 6373.0  # Approximate radius of the Earth in kilometers
    arc = np.zeros(len(position_2['Lat.']))

    for dd in range(len(position_2['Lat.'])):
        dlon = to_rad(position_2['Lon.'][dd]) - to_rad(position_1['Lon.'])
        dlat = to_rad(position_2['Lat.'][dd]) - to_rad(position_1['Lat.'])
        a = (np.sin(dlat / 2)) ** 2 + np.cos(to_rad(position_1['Lat.'])) * np.cos(to_rad(position_2['Lat.'][dd])) * (
            np.sin(dlon / 2)) ** 2
        arc[dd] = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * arc * 1000  # Convert to meters

    return distance


def calculate_DAS_section_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The bearing in degrees from the first point to the second point (relative to true north).
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate the difference in longitude
    delta_lon = lon2_rad - lon1_rad

    # Calculate the bearing
    x = np.sin(delta_lon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
    initial_bearing = np.arctan2(x, y)

    # Convert from radians to degrees
    initial_bearing_deg = np.degrees(initial_bearing)

    # Normalize the bearing to 0 - 360 degrees
    DAS_bearing = (initial_bearing_deg + 360) % 360

    return DAS_bearing


import numpy as np


def get_whale_position_lat_lon(lat_ref, lon_ref, distance_m, bearing, side):
    """
    Calculate the latitude and longitude of a point that is perpendicular to a given bearing.

    Args:
        lat_ref (float): Latitude of the reference point in degrees.
        lon_ref (float): Longitude of the reference point in degrees.
        distance_m (float): Distance to the perpendicular point in meters.
        bearing (float): The bearing of the line in degrees.
        side (str): 'right' or 'left', indicating the side of the line where the point lies.
        R (float, optional): Earth's radius in meters. Defaults to 6371000 meters (mean Earth radius).

    Returns:
        tuple: The latitude and longitude of the perpendicular point in degrees.
    """
    R = 6371000
    # Convert the reference latitude, longitude, and bearing to radians
    lat1 = np.radians(lat_ref)
    lon1 = np.radians(lon_ref)

    # Adjust the bearing by 90 degrees depending on the side (right or left)
    if side == 'right':
        bearing_perp = (bearing + 90) % 360
    elif side == 'left':
        bearing_perp = (bearing - 90) % 360
    else:
        raise ValueError("Side must be either 'right' or 'left'.")

    # Convert the perpendicular bearing to radians
    theta = np.radians(bearing_perp)

    # Convert the distance to radians
    d_by_R = distance_m / R

    # Calculate the new latitude using the Haversine formula
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_by_R) + np.cos(lat1) * np.sin(d_by_R) * np.cos(theta))

    # Calculate the new longitude
    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(d_by_R) * np.cos(lat1),
                             np.cos(d_by_R) - np.sin(lat1) * np.sin(lat2))

    # Convert the results from radians back to degrees
    lat2_deg = np.degrees(lat2)
    lon2_deg = np.degrees(lon2)

    return lat2_deg, lon2_deg


def get_theory_TDOA(DAS_position, whale_apex_m, whale_offset_m, dist, side='right', whale_depth_m=30, c=1490):
    """
    Calculate theoretical Time Difference of Arrival (TDOA) for a whale call based on the whale's position, cable (DAS) position,
    and known sound speed in water.

    Args:
        DAS_position (dict): A dictionary containing latitude, longitude, and depth information of the cable (DAS) positions.
        whale_apex_m (float): The distance from the start of the cable to the apex (closest point) of the whale's trajectory.
        whale_offset_m (float): The lateral offset of the whale from the cable (positive for right, negative for left).
        dist (numpy.ndarray): An array containing distances along the cable.
        whale_position (dict): A dictionary containing the latitude, longitude, and depth of the whale's position.
        side (str, optional): Indicates if the whale is on the 'right' or 'left' side of the cable. Defaults to 'right'.
        whale_depth_m (float, optional): The depth of the whale. Defaults to 30 meters.
        c (float, optional): The speed of sound in water in meters per second. Defaults to 1490 m/s.

    Returns:
        numpy.ndarray: An array containing the theoretical TDOAs for each position along the cable.
    """
    # Adjust lateral offset based on the whale's position (right or left of the DAS cable)
    # if side == 'left':
    #    whale_offset_m = -abs(whale_offset_m)  # Negative for left side
    # elif side == 'right':
    #    whale_offset_m = abs(whale_offset_m)   # Positive for right side
    # else:
    #    raise ValueError("Side must be either 'right' or 'left'.")

    # Find the index of whale_apex_m in dist
    ind_whale_apex = np.where(dist >= whale_apex_m)[0][0]
    whale_apex_lat_lon = {
        'Lat.': DAS_position['Lat.'][ind_whale_apex],
        'Lon.': DAS_position['Lon.'][ind_whale_apex],
    }

    # Get the bearing of the DAS cable around whale position
    step = 3
    DAS_bearing = calculate_DAS_section_bearing(
        DAS_position['Lat.'][ind_whale_apex - step],
        DAS_position['Lon.'][ind_whale_apex - step],
        DAS_position['Lat.'][ind_whale_apex + step],
        DAS_position['Lon.'][ind_whale_apex + step])

    # Get the whale position
    whale_position = {}
    whale_position['Lat.'], whale_position['Lon.'] = get_whale_position_lat_lon(
        DAS_position['Lat.'][ind_whale_apex],
        DAS_position['Lon.'][ind_whale_apex],
        whale_offset_m,
        DAS_bearing,
        side)

    # Create an updated whale position dictionary
    print(f"whale position {whale_position}")

    # Calculate 3D distance for each element in DAS_position
    distances = get_dist_lat_lon(whale_position, DAS_position)

    # Calculate depth differences
    depth_diff = np.array(DAS_position['Depth']) - whale_depth_m

    # Calculate total distance (3D)
    total_distance = np.sqrt(distances ** 2 + depth_diff ** 2)

    # Calculate theoretical TDOAs
    TDOA = (total_distance - np.min(total_distance)) / c

    return TDOA


def get_peak_indexes_subset(peaks_indexes, start_time_s, analysis_win_s, time, dist, fs):
    """
    Extract a subset of peak indexes and corresponding time values within a specified time window.

    Args:
        peaks_indexes (numpy.ndarray): A 2D array containing peak indexes.
        start_time_s (float): The start time (in seconds) of the analysis window.
        analysis_win_s (float): The duration (in seconds) of the analysis window.
        time (numpy.ndarray): An array containing time values.
        dist (numpy.ndarray): An array containing distances along the cable.
        fs (float): The sampling frequency (in Hz).

    Returns:
        tuple: A tuple containing:
            - peaks_indexes_call (numpy.ndarray): A 2D array containing peak indexes within the specified time window.
            - time_call (numpy.ndarray): An array containing time values corresponding to the selected peak indexes.

    """
    # Sort the indexes based on the times
    ind_sort = np.argsort(time[peaks_indexes[1]])
    peaks_indexes = peaks_indexes[:, ind_sort]

    # Find the indices corresponding to start and stop times
    ind_start = np.argmax(time[peaks_indexes[1]] >= start_time_s)
    ind_stop = np.argmax(time[peaks_indexes[1]] >= start_time_s + analysis_win_s)

    # Take the subset
    peaks_indexes_call = np.array([
        peaks_indexes[0, ind_start:ind_stop],
        peaks_indexes[1, ind_start:ind_stop] - int(start_time_s * fs)])

    # Extract corresponding time values
    time_call = time[peaks_indexes_call[1, 0]:peaks_indexes_call[1, -1] + 1]

    return peaks_indexes_call, time_call


def shift_xcorr(x, y):
    """compute the shifted (positive lags only) cross correlation between two 1D arrays

    Parameters
    ----------
    x : numpy.ndarray
        1D array containing signal
    y : numpy.ndarray
        1D array containing signal

    Returns
    -------
    numpy.ndarray
        1D array cross-correlation betweem x and y, only for positive lags
    """
    corr = sp.correlate(x, y, mode='full', method='fft')
    return corr[len(x) - 1:]


def compute_cross_correlogram(data, template):
    """
    Compute the cross correlogram between the given data and template.

    Parameters
    ----------
    data : numpy.ndarray
        The input data array.
    template : numpy.ndarray
        The template array.

    Returns
    -------
    numpy.ndarray
        The cross correlogram array.
    """
    # Normalize data along axis 1 by its maximum (peak normalization)
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.max(np.abs(data), axis=1, keepdims=True)
    template = (template - np.mean(template)) / np.max(np.abs(template))

    # Compute correlation along axis 1
    cross_correlogram = np.empty_like(data)

    for i in tqdm(range(data.shape[0])):
        cross_correlogram[i, :] = shift_xcorr(norm_data[i, :], template)

    # Normalize - so values between 0-1 and where 1 <=> template autocorrelation
    corr_norm = np.max(sp.correlate(template, template, mode='full', method='fft'))

    cross_correlogram = cross_correlogram / corr_norm

    return cross_correlogram
