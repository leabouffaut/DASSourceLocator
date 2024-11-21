# DAS Data Annotation Tool

## Overview

This tool is designed for annotating Distributed Acoustic Sensing (DAS) data, specifically for matching theoretical times of arrivals (TOA) to recorded data in the spatiotemporal domain. 
It was developed by Léa Bouffaut, Ph.D.

## Features
- Pre-processes DAS data
- Enables interactive labeling
- Displays a scatter plot of cross-correlation output
- Allows user to define and adjust values for:
  - Whale apex (minimum whale-DAS channel)
  - Whale offset (distance between whale and DAS at apex)
  - First time of arrival (Start time)
- Option to pick the side of the source (left/right) on the interrogator
- Outputs annotations to a CSV file
  
[Check out the provided demo video](https://github.com/leabouffaut/DASSourceLocator/blob/main/DASSourceLocatorDemo.mp4)



## Technical Details
- Tested on 3 different DAS systems
- Optimized for Fin whale 20 Hz calls
- Built using Streamlit
- Runs on Python 3.11.9

## Usage

To run the application:

```bash
streamlit run source_locator_app.py
