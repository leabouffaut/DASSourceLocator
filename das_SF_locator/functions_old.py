import scipy.signal as sp
import numpy as np
import csv

# Use relative imports
from . import h5pydict
from . import simpleDASreader as sd


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

    # There is a discrepency between the number of channels in the original data and here (less here)
    # Add leading zeros to align the number of channels with metadata
    # Difference between original number of channels and position = metadata["nx"]-len(metadata["Position"]["Depth"])
    channel = list(range((metadata["nx"] - len(depth)))) + channel
    lat = [0] * (metadata["nx"] - len(depth)) + lat
    lon = [0] * (metadata["nx"] - len(depth)) + lon
    depth = [0] * (metadata["nx"] - len(depth)) + depth

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


def get_distance(position_1, position_2):
    """
    Calculate the distance between two sets of latitude and longitude positions using the Haversine formula.

    Args:
        position_1 (dict): A dictionary containing latitude and longitude positions for the first set of locations.
        position_2 (dict): A dictionary containing latitude and longitude positions for the second set of locations.
        R (float, optional): The radius of the Earth in kilometers. Defaults to 6373.0.

    Returns:
        numpy.ndarray: An array containing the distances (in meters) between corresponding positions in position_1 and position_2.
    """
    R = 6373.0  # Approximate radius of the Earth in kilometers

    lat1_rad = np.radians(position_1['Lat.'])
    lon1_rad = np.radians(position_1['Lon.'])
    lat2_rad = np.radians(position_2['Lat.'])
    lon2_rad = np.radians(position_2['Lon.'])

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c * 1000  # Convert to meters
    return distance


def get_theory_TDOA(DAS_position, whale_apex_m, whale_offset_m, dist, whale_depth_m=30, c=1490):
    """
    Calculate theoretical Time Difference of Arrival (TDOA) for a whale call based on the whale's position, cable (DAS) position,
    and known sound speed in water.

    Args:
        DAS_position (dict): A dictionary containing latitude, longitude, and depth information of the cable (DAS) positions.
        whale_apex_m (float): The distance from the start of the cable to the apex (closest point) of the whale's trajectory.
        whale_offset_m (float): The lateral offset of the whale from the cable.
        dist (numpy.ndarray): An array containing distances along the cable.
        whale_depth_m (float, optional): The depth of the whale. Defaults to 30 meters.
        c (float, optional): The speed of sound in water in meters per second. Defaults to 1500 m/s.

    Returns:
        numpy.ndarray: An array containing the theoretical TDOAs for each position along the cable.
    """
    # Get the variables
    z = np.array(DAS_position['Depth']) - whale_depth_m
    hy = whale_offset_m

    # Find the index of whale_apex_m in dist
    ind_whale_apex = np.where(dist >= whale_apex_m)[0][0]
    whale_apex_lat_lon = {
        'Lat.': DAS_position['Lat.'][ind_whale_apex],
        'Lon.': DAS_position['Lon.'][ind_whale_apex],
    }

    # Calculate hx using the coordinates
    hx = get_distance(whale_apex_lat_lon, DAS_position)

    # Calculate r (distance from whale apex to cable positions)
    r = np.sqrt(hx ** 2 + hy ** 2 + z ** 2)

    # Calculate theoretical TDOAs
    TDOA = (r - np.min(r)) / c

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


from tqdm import tqdm


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
