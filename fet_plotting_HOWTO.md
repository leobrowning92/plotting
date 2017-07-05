# FET analysis

This is for automatically collected data that is ouputted
from the script `pa-control/cntrun.py`
from the Agilent 4156C parameter analyser

## run instructions

### single folder plotting

ensure directory structure is as follows in this example:

    .
    ├── <step directory>
    │   └── data
    ├── postSD-autodata
    │   ├── data
    │   └── plots
    ├── postSD-manualdata
    │   └── data
    ├── postSD-redo-485
    │   ├── data
    │   └── plots
    └── postSU8-autodata
        ├── data
        └── plots
usually this just involves moving the data csv files to to a data subdirectory.


first run:

    number_runs.sh <step directory/data>

this will rename the runs with run numbers for each type of sweep in the standard sweep library from `pa-control/cntrun.py`

To then plot a series of curves for each chp in the plots subdir (this will be made if it doesn't exist) run:

    fet_curveplotting.py <step directory> <chip index start> <finish>

where <chip index start/finish> is the start end chip indices in the filename
of "COL<chip index>\_blahdeblah\_.csv"

    python3 fet_analysis.py <step directory> <search term>

where the search term filters the data to be analysed to only files that contain that string in the filename

At this point there should be plots of all the data from the directory in the plots folder, with a figure for each chip, containing all data taken.
