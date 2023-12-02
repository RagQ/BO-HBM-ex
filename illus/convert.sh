#!/bin/bash

for file in `ls *.pdf`
do
    convert -alpha deactivate -density 300 $file ${file%.pdf}.png
done
