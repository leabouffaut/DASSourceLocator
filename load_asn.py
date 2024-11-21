from das_SF_locator import functions as fct

# 1) Metadata
# Load file, time and metadata
tr, fileBeginTimeUTC, metadata = fct.load_ASN_DAS_file(list_file[0])
# Metadata in code
fs, dx, nx, ns, gauge_length = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"]

st.write(f'Sampling frequency: {metadata["fs"]} Hz')
st.write(f'Channel spacing: {metadata["dx"]} m')
st.write(f'Gauge length: {metadata["GL"]} m')
st.write(f'File duration: {metadata["ns"] / metadata["fs"]} s')
st.write(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
st.write(f'Number of channels: {metadata["nx"]}')
st.write(f'Number of time samples: {metadata["ns"]}')



# 2) Load data
# Load data
tr, fileBeginTimeUTC,m = fct.load_ASN_DAS_file(file)
tr = tr[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64)
del m
# Store the following as the dimensions of our data block
nnx = tr.shape[0]
nns = tr.shape[1]

# Define new time and distance axes
time = np.arange(nns) / metadata["fs"]
dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata["dx"]








