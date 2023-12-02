#!/bin/bash


cd ../ExpOptimDuffing
for dir in `ls -d */`; do
    cd $dir
    convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 ContourCEI_*.pdf anim_contourEI.gif
    convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 ContourObj_*.pdf anim_contourObj.gif
    convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 SurfaceCEI_*.pdf anim_surfaceEI.gif
    convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 SurfaceObj_*.pdf anim_surfaceObj.gif
    cd ..
done
