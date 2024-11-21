if st.session_state.save == 1:
    # Save current file data
    st.session_state.save_data.append(
        [list_file[file_index], estimated_whale_apex_m, estimated_whale_offset_m, estimates_start_time_s])
    # Move to the next file
    st.session_state.file_index += 1
    if st.session_state.file_index >= len(list_file):
        st.write("All files processed!")
        st.stop()
    st.experimental_rerun()

if st.session_state.save == 2:
    # Save and exit
    st.session_state.save_data.append(
        [list_file[file_index], estimated_whale_apex_m, estimated_whale_offset_m, estimates_start_time_s])
    st.write("Data saved and exiting.")
    st.stop()









import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import das4whales as dw
import scipy.signal as sp
from pathlib import Path
from das_SF_locator import functions_old as fct

# Create a "staged" version of the program so not all shows up at once
def set_state_stage(i):
    st.session_state.stage = i
def set_state_file_index(i):
    st.session_state.file_index = i
def set_state_save(i):
    st.session_state.save = i

if 'stage' not in st.session_state:
    st.session_state.stage = 0

if st.session_state.stage >= 0:
    # Set up the Streamlit page
    st.set_page_config(page_title='DAS Source Locator')
    st.title('DAS Source Locator')

    # File selection
    list_file_path_default = '/Users/lb736/Python/DAS_GL/DAS_MedSea/annotations/MedSea_listfiles_to_analyze.txt'
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
        columns=['File','Whale apex (m)','Whale offset (m)','Start time (s)'],
        index=np.arange(len(list_file)))

    # Show the files to process
    last_folder_and_file = [Path(path).parent.name + "/" + Path(path).name for path in list_file]
    st.table(pd.DataFrame(last_folder_and_file, columns=['Files to process']))

    # Metadata and channel selection
    tr, fileBeginTimeUTC, metadata = fct.load_ASN_DAS_file(list_file[0])
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
        value="3050"))
    channel_max = int(col2.text_input(
        "Channel max (m)",
        value="53050"))
    channel_step = col3.select_slider(
        "Channel step (m)",
        value = int(8),
        options=[int(np.floor(metadata["dx"])) * i for i in range(1, 5)])

    channel_step = min([metadata["dx"] * i for i in range(1, 5)], key=lambda x: abs(x - channel_step))
    selected_channels_m = [channel_min, channel_max, channel_step]
    selected_channels = [int(selected_channels_m // dx) for selected_channels_m in selected_channels_m]

    # Position file and display
    position_file_default = '/Users/lb736/Python/DAS_GL/DAS_MedSea/MEUST_WGS84_latlondepth_corrected.txt'
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


if st.session_state.stage >= 1:
    # Set the session states for going through files & save
    if 'file_index' not in st.session_state:
        st.session_state.file_index = 0
    st.session_state.save = 0

    # Process current file
    file_index = st.session_state.file_index

    tr, fileBeginTimeUTC, m = fct.load_ASN_DAS_file(list_file[file_index])
    tr = tr[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)
    del m
    nnx = tr.shape[0]
    nns = tr.shape[1]
    time = np.arange(nns) / metadata["fs"]
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]

    # Create the f-k filter
    fk_filter = dw.dsp.hybrid_ninf_filter_design((tr.shape[0], tr.shape[1]), selected_channels, dx, fs,
                                                 cs_min=1350, cp_min=1450, cp_max=3300, cs_max=3450, fmin=14,
                                                 fmax=30,
                                                 display_filter=False)
    # Apply the bandpass
    tr = dw.dsp.bp_filt(tr, fs, 14, 30)

    # Apply the f-k filter to the data, returns spatio-temporal strain matrix
    trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=False)

    # Detection
    HF_note = dw.detect.gen_template_fincall(time, fs, fmin=14, fmax=24, duration=0.70)
    corr_m_HF = fct.compute_cross_correlogram(trf_fk, HF_note)

    # Find the local maximas using find peaks and a threshold
    thres = 0.5

    # Find the arrival times and store them in a list of arrays format
    peaks_indexes_m_HF = dw.detect.pick_times(corr_m_HF, threshold=thres)

    # Convert the list of array to tuple format
    peaks_indexes_tp_HF = dw.detect.convert_pick_times(peaks_indexes_m_HF)

    # Add peaks
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time[peaks_indexes_tp_HF[1]],
        y=dist[peaks_indexes_tp_HF[0]] / 1000,
        mode='markers',
        marker=dict(color='white', opacity=0.2, size=5),
        name='Peaks',
    ))

    # Get user input for whale position and other parameters
    st.header('Find whale position')
    estimated_whale_apex_m = st.number_input(
        "Whale apex (m)",
        min_value=int(dist[0]),
        max_value=int(dist[-1]),
        step=50,
        value=22600,
        key=f'whale_apex_{file_index}'  # Unique key for whale apex
    )

    estimated_whale_offset_m = st.number_input(
        "Whale offset (m)",
        min_value=0,
        max_value=20000,
        step=100,
        value=2000,
        key=f'whale_offset_{file_index}'  # Unique key for whale offset
    )

    estimates_start_time_s = st.number_input(
        "Start time (s)",
        min_value=time[0],
        max_value=time[-1],
        step=0.1,
        value=2.0,
        key=f'start_time_{file_index}'  # Unique key for start time
    )

    whale_depth_m = 30
    c = 1490

    # Compute theoretical TDOA and plot
    TDOA_th = fct.get_theory_TDOA(
        position,
        estimated_whale_apex_m,
        estimated_whale_offset_m,
        dist,
        whale_depth_m=whale_depth_m, c=c)

    # Add theoretical TDOA line
    fig.add_trace(go.Scatter(
        x=TDOA_th + estimates_start_time_s,
        y=dist / 1000,
        mode='lines',
        line=dict(color='red', width=2),
        name='Theoretical TOA'
    ))

    # Update layout with fixed axis labels and limits
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Distance (km)',
        xaxis=dict(range=[time[0], time[-1]]),  # Fixed x-axis range
        yaxis=dict(range=[dist[0] / 1000, dist[-1] / 1000]),  # Consistent y-axis range
        title=fileBeginTimeUTC.strftime("%Y-%m-%d %H:%M:%S"),
        width=864,  # Fixed figure width in pixels
        height=720  # Fixed figure height in pixels
    )
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

    # Buttons to save and move to the next file
    col1, col2 = st.columns(2)
    col1.button('Save and Analyze Next file', on_click=set_state_save, args=[1],  key=f'next_button_{file_index}')
    col2.button('Save and Exit', on_click=set_state_save, args=[2], key=f'exit_button_{file_index}')

    if st.session_state.save == 1:
        # Save data in list
        save_data.append([
            list_file[file_index],
            estimated_whale_apex_m,
            estimated_whale_offset_m,
            estimates_start_time_s,
        ])
        st.write(save_data)
        st.session_state.file_index = (st.session_state.file_index + 1) % len(list_file)
        st.experimental_rerun()

    if st.session_state.save == 2:
        # Save data in list
        save_data.append([
            list_file[file_index],
            estimated_whale_apex_m,
            estimated_whale_offset_m,
            estimates_start_time_s,
        ])
        st.write(save_data)
        # Stop prog
        st.stop()