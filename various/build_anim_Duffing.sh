#!/bin/bash

python build_pic_Duffing.py

BUILD_GIF='convert -alpha deactivate -delay 50 -loop 0 -density 300'
OPTIM_GIF='gifsicle -O3 --colors 256 --scale=0.25'

cd figs
    ${BUILD_GIF} Drms_*.pdf anim_Drms.gif
    ${BUILD_GIF} Arms_*.pdf anim_Arms.gif
    ${OPTIM_GIF} anim_Drms.gif -o anim_Drms-optim.gif
    ${OPTIM_GIF} anim_Arms.gif -o anim_Arms-optim.gif
cd ..