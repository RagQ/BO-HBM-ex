#!/bin/bash

BUILD_GIF='convert -alpha deactivate -delay 50 -loop 0 -density 300'
OPTIM_GIF='gifsicle -O3 --colors 256 --scale=0.25'

cd ../ExpOptimDuffing
for dir in `ls -d */`; do
    cd $dir
    echo "$dir"
     ${BUILD_GIF} ContourCEI_*.pdf anim_contourEI.gif &
     ${BUILD_GIF} ContourObj_*.pdf anim_contourObj.gif &
     ${BUILD_GIF} SurfaceCEI_*.pdf anim_surfaceEI.gif &
     ${BUILD_GIF} SurfaceObj_*.pdf anim_surfaceObj.gif
     #
     ${OPTIM_GIF} anim_contourEI.gif -o anim_contourEI-optim.gif 
     ${OPTIM_GIF} anim_contourObj.gif -o anim_contourObj-optim.gif &
     ${OPTIM_GIF} anim_surfaceEI.gif -o anim_surfaceEI-optim.gif &
     ${OPTIM_GIF} anim_surfaceObj.gif -o anim_surfaceObj-optim.gif

     echo "$dir done"
    cd ..
done
