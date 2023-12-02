#!/bin/bash

python build_pic_Duffing.py

cd figs
convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 Drms_*.pdf anim_Drms.gif
convert -alpha deactivate -verbose -delay 50 -loop 0 -density 300 Arms_*.pdf anim_Arms.gif
cd ..