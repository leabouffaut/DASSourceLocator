# Annotating tool for DAS data
#
# Nov. 2024, V0. Léa Bouffaut, Ph.D., lea.bouffaut@cornell@edu
#
# It pre-processed data and then enables interactive labeling to match theoretical times of arrivals (TOA) to the
# recorded data in the spatiotemporal domain. The application displays a scatter plot of the output of a cross
# correlation between the pre-processed data and a synthetical down sweep where the output is normalized by the
# autocorrelation of the synthetic signal, and thresholded. The matched filter steps is aimed to maximize the
# signal to noise ratio through pulse compression and, improve the spatiotemporal resolution of the analysis.

# Using the spatiotemporal representation, the user can define and adjust values for whale apex (minimum whale-DAS
# channel), whale offset (distance between the whale and the DAS at the apex) and first time of arrival (Start time)
# based on the match between the curves. An option is available to pick the side of the source (left/right) on the
# interrogator to outwards channel axis, when the cable layout enables to break the array's symmetry. The output
# annotation file is a csv file reporting for each labeled Fin whale 20 Hz call, the corresponding DAS file and the user
# inputs. Note that the most challenging measurement is the whale offset, that lacks precision for offsets <= the local
# bathymetry, and minimum ranges are set to that value.

# This code has been tested on 3 different DAS', on fin whale 20 Hz calls and uses streamlit, runs in Python 3.11.9

# RUN: To run the code, write in the terminal: streamlit run source_locator_app.py
# It should open a new browser tab

# STOP: To stop the code, ctrl+c in the terminal (Macs)

# !! Warning!!:
# When you re-run the app, it will save over any existing file in your repository called whale_analysis_data.csv


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import das4whales as dw
from pathlib import Path
from das_SF_locator import functions as fct
import time as python_time

def set_dataset_paths(dataset):
    if dataset == 'MedSea':
        position_file_default = '/Volumes/CCB/projects/2022_CLOCCB_OR_S1113/MedSea/DASPosition/MEUST_WGS84_latlondepth_corrected.txt'
        selected_channels_m = [3050, 53050, 8]
        interrogator = 'optodas_old'
        list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_MedSea/annotations/list_files_to_process/MedSea_listfiles_to_analyze.txt'


    elif dataset == 'OOINorthC2':
        position_file_default = '/Volumes/CCB/projects/2022_CLOCCB_OR_S1113/OOI/DASPosition/north_DAS_latlondepth.txt'
        selected_channels_m = [15000, 65000, 8]
        interrogator = 'optasense'
        list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_OOI/annotations/list_files_to_process/OOI_GL_30m_NorthC2.txt'

    elif dataset == 'OOINorthC3':
        position_file_default = '/Volumes/CCB/projects/2022_CLOCCB_OR_S1113/OOI/DASPosition/north_DAS_latlondepth.txt'
        selected_channels_m = [15000, 65000, 8]
        interrogator = 'optasense'
        list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_OOI/annotations/list_files_to_process/OOI_GL_30m_NorthC3.txt'

    elif dataset == 'OOISouthC1':
        position_file_default = '/Volumes/CCB/projects/2022_CLOCCB_OR_S1113/OOI/DASPosition/south_DAS_latlondepth.txt'
        selected_channels_m = [15000, 65000, 8]
        interrogator = 'optasense'
        list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_OOI/annotations/list_files_to_process/OOI_GL_50m_SouthC1.txt'

    elif dataset == 'Svalbard':
        # annotation_file = ''
        position_file_default = '/Volumes/CCB/projects/2022_CLOCCB_OR_S1113/Svalbard/DASPosition/Svalbard_DAS_latlondepth.txt'
        selected_channels_m = [40000, 100000, 8]
        interrogator = 'optodas'
        list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_Svalbard/annotations/list_files_to_process/svalbard_to_annotate.txt'
    return position_file_default, selected_channels_m, interrogator, list_file_path_default

# Functions to manage Streamlit states
def set_state_stage(i):
    st.session_state.stage = i

def set_state_file_index(i):
    st.session_state.file_index = i

def set_state_save(i):
    st.session_state.save = i

# User parameters in code --------- (Note, these are set for fin whales)
datasets_to_process = ['Svalbard']  # Replace by the name of the dataset of interest (see set_dataset_paths)

whale_depth_m = 20 # Vocalizing depth of the whale (m)
c = 1490 # Considered soundspeed to calculate theoretical TOA (m/s)

thres = 0.5  # Find the local maximas using find peaks and a threshold ASN data 0.5 (Svalbard, MedSea)/OOI = 0.2

fk_filter_bounds_m_s = [1300, 1450, 3300, 3450] # fk filter bounds [cs_min, cp_min, cp_max, cs_max] in m/s
bandpass_filter_bounds_Hz = [14, 30] # bandpass filter frequencies [fmin, fmax] in Hz


# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 0

if 'file_index' not in st.session_state:
    st.session_state.file_index = 0

if 'save' not in st.session_state:
    st.session_state.save = 0

if 'save_data' not in st.session_state:
    st.session_state.save_data = []

# Create a placeholder for file loading
if 'peaks_indexes_tp_HF' not in st.session_state:
    st.session_state.peaks_indexes_tp_HF = None

if st.session_state.stage >= 0:
    # Set up the Streamlit page
    st.set_page_config(page_title='DAS Source Locator')
    st.title('DAS Source Locator')

    # Go through each dataset
    for ds in datasets_to_process:
        # File selection
        # Get dataset-related file paths and info
        position_file_default, selected_channels_m, interrogator, list_file_path_default = set_dataset_paths(ds)

        list_file_path = st.text_input(
            "List of files to analyze",
            value=list_file_path_default)

        # Load the list of files
        with open(list_file_path, 'r') as file:
            list_file = [line.strip() for line in file.readlines()]

        # Get the number of files to analyze
        nb_files = len(list_file)

        # Create the save data dataframe
        save_data_df = pd.DataFrame(
            columns=['File','Whale apex (m)','Whale offset (m)','Start time (s)', 'Whale side'],
            index=np.arange(len(list_file)))

        # Show the files to process
        last_folder_and_file = [Path(path).parent.name + "/" + Path(path).name for path in list_file]
        st.table(pd.DataFrame(last_folder_and_file, columns=['Files to process']))

        # Metadata and channel selection
        metadata = fct.get_metadata(interrogator, list_file[0])

        fs, dx, nx, ns, gauge_length = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"]
        st.write(f'Sampling frequency: {metadata["fs"]} Hz')
        st.write(f'Channel spacing: {metadata["dx"]} m')
        st.write(f'Gauge length: {metadata["GL"]} m')
        st.write(f'File duration: {metadata["ns"] / metadata["fs"]} s')
        st.write(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
        st.write(f'Number of channels: {metadata["nx"]}')
        st.write(f'Number of time samples: {metadata["ns"]}')

        # Channel selection
        st.header('Channel selection for the analysis')
        col1, col2, col3 = st.columns(3)
        channel_min = int(col1.text_input(
            "Channel min (m)",
            value=str(selected_channels_m[0])))
        channel_max = int(col2.text_input(
            "Channel max (m)",
            value=str(selected_channels_m[1])))
        channel_step = col3.select_slider(
            "Channel step (m)",
            value = int(8),
            options=[int(np.floor(metadata["dx"])) * i for i in range(1, 5)])

        channel_step = min([metadata["dx"] * i for i in range(1, 5)], key=lambda x: abs(x - channel_step))
        selected_channels_m = [channel_min, channel_max, channel_step]
        selected_channels = [int(selected_channels_m // dx) for selected_channels_m in selected_channels_m]

        # Position file and display
        position_file = st.text_input(
            "DAS position file",
            value=position_file_default)
        position = fct.get_lat_lon_depth(position_file, selected_channels, metadata)
        df_position = pd.DataFrame(position)
        df_position = df_position.rename(columns={'Lat.': 'LAT', 'Lon.': 'LON'})
        st.map(pd.DataFrame(df_position))

        # Create the dataframe where we are going to store the Analysis data
        # ['File', 'Whale apex (m)', 'Whale offset (m)', 'Start time (s)'])
        save_data = []

        st.button('Start data analysis', on_click=set_state_stage, args=[1])

    if st.session_state.stage >= 1 and st.session_state.file_index <= nb_files - 1:
        file_index = st.session_state.file_index

        # Load DAS data only once
        if st.session_state.peaks_indexes_tp_HF is None:
            st.write('Loading & processing data...')

            tr, time, dist, fileBeginTimeUTC = fct.load_data(interrogator, list_file[file_index], selected_channels, metadata)
            st.session_state.time = time
            st.session_state.dist = dist
            st.session_state.fileBeginTimeUTC = fileBeginTimeUTC

            # Create the f-k filter -
            # the reason the filter is created within the loop is that some of the datasets are heterogeneous in terms
            # of sampling frequencies and channel spacing. If working with a homogeneous dataset, move this after the
            # position file and display part, after loading one file to get the sizes

            fk_filter = dw.dsp.hybrid_ninf_filter_design(
                (tr.shape[0], tr.shape[1]),
                selected_channels,
                dx, fs,
                cs_min=fk_filter_bounds_m_s[0], cp_min=fk_filter_bounds_m_s[1],
                cp_max=fk_filter_bounds_m_s[2], cs_max=fk_filter_bounds_m_s[3],
                fmin=bandpass_filter_bounds_Hz[0], fmax=bandpass_filter_bounds_Hz[1],
                display_filter=False
            )
            # Bandpass added for OOI NorthC2 - It was not picking up signals closer to shore
            # tr = dw.dsp.bp_filt(tr, fs, 14, 30)

            # Apply the f-k filter to the data, returns spatio-temporal strain matrix
            tr_filtered = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=False)

            # X-corr
            # the reason the X-corr is created within the loop is that some of the datasets are heterogeneous in terms
            # of sampling frequencies and channel spacing. If working with a homogeneous dataset, move this after the
            # position file and display part (and calculate the time axis from the metadata)
            HF_note = dw.detect.gen_template_fincall(time, fs, fmin=14, fmax=24, duration=0.70)
            corr_m_HF = fct.compute_cross_correlogram(tr_filtered, HF_note) # Normalized xcorr

            # Find the arrival times and store them in a list of arrays format
            peaks_indexes_m_HF = dw.detect.pick_times(corr_m_HF, threshold=thres)
            st.session_state.peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_m_HF)

        # Apply filtering and compute peaks
        time = st.session_state.time
        dist = st.session_state.dist
        peaks_indexes_tp_HF = st.session_state.peaks_indexes_tp_HF

        # Whale position inputs
        st.header('Find whale position')
        estimated_whale_apex_m = st.number_input(
            "Whale apex (m)",
            min_value=int(dist[0]),
            max_value=int(dist[-1]),
            step=50,
            value=int((selected_channels_m[1]-selected_channels_m[0])/2+selected_channels_m[0]),
            key=f'whale_apex_0'
        )

        estimated_whale_offset_m = st.number_input(
            "Whale offset (m)",
            min_value=0,
            max_value=20000,
            step=100,
            value=2000,
            key=f'whale_offset_0'
        )

        estimates_start_time_s = st.number_input(
            "Start time (s)",
            min_value=time[0],
            max_value=time[-1],
            step=0.1,
            value=2.0,
            key=f'start_time_0'
        )



        TDOA_th_right = fct.get_theory_TDOA(
            position, estimated_whale_apex_m, estimated_whale_offset_m,
            dist, side='right', whale_depth_m=whale_depth_m, c=c)

        TDOA_th_left = fct.get_theory_TDOA(
            position, estimated_whale_apex_m, estimated_whale_offset_m,
            dist, side='left', whale_depth_m=whale_depth_m, c=c)

        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time[peaks_indexes_tp_HF[1]],
            y=dist[peaks_indexes_tp_HF[0]] / 1000,
            mode='markers',
            marker=dict(color='white', opacity=0.2, size=5),
            name='Peaks',
        ))

        # Add theoretical TDOA lines
        fig.add_trace(go.Scatter(
            x=TDOA_th_right + estimates_start_time_s,
            y=dist / 1000,
            mode='lines',
            line=dict(color='red', width=2),
            name='Theoretical TOA right'
        ))

        fig.add_trace(go.Scatter(
            x=TDOA_th_left + estimates_start_time_s,
            y=dist / 1000,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Theoretical TOA left'
        ))

        # Update layout with fixed axis labels and limits
        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Distance (km)',
            xaxis=dict(range=[time[0], time[-1]]),
            yaxis=dict(range=[dist[0] / 1000, dist[-1] / 1000]),
            title=st.session_state.fileBeginTimeUTC.strftime("%Y-%m-%d %H:%M:%S"),
            width=864,  # Fixed figure width in pixels
            height=720  # Fixed figure height in pixels
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig)

        # Buttons to save and move to the next file
        col1, col2 = st.columns(2)
        # Radio button to choose what side the whale is on
        whale_side = col1.radio(
            "Whale side",
            ["right", "left", "either", "N/A"],
            index=None,
        )

        col2.button('Save and Analyze Next file', on_click=set_state_save, args=[1],  key=f'next_button_{file_index}')

        if st.session_state.save == 1:
            # Save current file data
            st.session_state.save_data.append(
                [list_file[file_index], estimated_whale_apex_m, estimated_whale_offset_m, estimates_start_time_s, whale_side])

            # Create a pandas DataFrame from the save_data session state
            save_data_df = pd.DataFrame(
                st.session_state.save_data,
                columns=['File', 'Whale apex (m)', 'Whale offset (m)', 'Start time (s)', 'Whale side']
            )
            st.dataframe(save_data_df)

            # Save DataFrame as CSV
            csv = save_data_df.to_csv('whale_analysis_data.csv', ',',index=False)
            csv_dl_button = save_data_df.to_csv(index=False)

            # Provide a download button
            col2.download_button(
                label="Download data as CSV",
                data=csv_dl_button,
                file_name='whale_analysis_data.csv',
                mime='text/csv',
            )

            python_time.sleep(1)
            st.session_state.save = 0
            st.session_state.peaks_indexes_tp_HF = None

            # Move to the next file
            st.session_state.file_index += 1
            if st.session_state.file_index >= nb_files-1:
                st.write("All files processed!")
                st.stop()
            else:
                st.rerun()