#!/bin/bash 
# add label to mypaint brush by Deevad (need Imagemagick installed)

for brushpng in *.png ; do
# get the name from the file name
label=${brushpng/_prev.png/}
cleanlabel=${label//[_]/-}
	# Communications
	echo '====='$brushpng'====='
	echo '> adding label: '$cleanlabel''
	# Imagemagick ( label auto sized + grey border )
	convert $brushpng -fill black -font fixed -gravity South -background '#FFF8' -size 124x26 caption:''$cleanlabel'' -composite -bordercolor '#CCC9' -border 2 -bordercolor '#9999' -border 1 -resize '128x128' -unsharp 0x.5 -flatten +matte $brushpng

done

echo 'Done'




