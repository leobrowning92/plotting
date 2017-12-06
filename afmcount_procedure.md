## Process AFM images
first open up gwyddion and use pygwy_processFolder.py to generate images
from all files.

    Data Process > PyGwy console > open

then replace the directory variable with the absolute path to the folder containing all of the `.nid` files that you want processed. and hit execute.
## Image Resizing
resize images to 1024x1024 pixels and put them in a counting directory

    mkdir counting
    cp *.png counting/
    mogrify -resize 1024x1024 counting/*.png

## Counting in ImageJ
open an image in image J then go to `Analyze > Set Measurement`
and make sure only _mean grey area_ is selected if you are doing junction measurements.


#### protip:
remap a key to click so you dont get rsi counting points.
on ubuntu go to `System Settings > Universal Access > Pointing and Clicking` and turn on `Mouse Keys` This remaps `numpad5` as a left click and will save you pain.

select all points, then get the measurements using `ctrl+m`. then use `ctrl+s` to save with the image filename as `fname_junctions.txt`

## analyzing junction density

    afmdensityplot -s $(ls *.txt)

Other options include --show to see the plots as you go.

## for measuring tubes

open image in ImageJ
then use `analyze > tools > roi manager` and tick show all

then use line tool to select line, and then add to roi manager using `t`.


make sure the _centroid_ and _mean grey area_ measurements are selected
when you have finished all of the lines use `ctrl+m` to measure and then save as with junctions.
