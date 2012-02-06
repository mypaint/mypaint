#!/bin/sh
# by Deevad
# Mypaint icon exporter 2012-01-30
# --------------------------------
# what it does : export *.svgz Mypaint icons templates  ( from mypaint/svg folder ) to the right folders automagically with right names. Finish long export time with Inkscape !
# what it needs : Inkscape installed.
# how to use : save your *.svgz file ( done from template ) in mypaint/svg with the background layer hidden, then call the script. 
# ./0_icon-exporter.sh %f
# --------------------------------
# Note : the script don't do the 'scalable' version yet. 
# Note2 : easy to set as a nautilus script / KDE services / XFCE action ( right click menu ). 


if [ $# -eq 0 ]; then
        # error no file selected
        exit 
fi
IFS='
'

for arg in "$@"
do
targetpngfilename=$(echo $arg|sed 's/\(.*\)\..\+/\1/')".png"
targetsvgfilename=$(echo $arg|sed 's/\(.*\)\..\+/\1/')".svg"

 	# export scalable SVG ( in progress ): 
 	# note; go after to /desktop/icons/hicolor/scalable/actions/ , and crop the file manually
 	# inkscape exporter have a bug and area don't work on svg export... 
 	# inkscape -f "$arg" -a 16:65:64:17 -w 48 -h 48 --vacuum-defs -y 0 -z -l "$targetsvgfilename"
 	# mv "$targetsvgfilename" ../desktop/icons/hicolor/scalable/actions/"$targetsvgfilename"
 	
 	# export png 48x48px
 	inkscape -f "$arg" -a 16:64:64:16 -y 0 -z -e "$targetpngfilename"
 	mv "$targetpngfilename" ../desktop/icons/hicolor/48x48/actions/"$targetpngfilename"
 	
 	# export png 32x32px
 	inkscape -f "$arg" -a 79:56:111:24 -y 0 -z -e "$targetpngfilename"
 	mv "$targetpngfilename" ../desktop/icons/hicolor/32x32/actions/"$targetpngfilename"
 	
 	# export png 24x24px
 	inkscape -f "$arg" -a 123:52:147:28 -y 0 -z -e "$targetpngfilename"
 	mv "$targetpngfilename" ../desktop/icons/hicolor/24x24/actions/"$targetpngfilename"
 	
 	# export png 22x22px
 	inkscape -f "$arg" -a 154:51:176:29 -y 0 -z -e "$targetpngfilename"
 	mv "$targetpngfilename" ../desktop/icons/hicolor/22x22/actions/"$targetpngfilename"
 	
 	# export png 16x16px
 	inkscape -f "$arg" -a 192:48:208:32 -y 0 -z -e "$targetpngfilename"
 	mv "$targetpngfilename" ../desktop/icons/hicolor/16x16/actions/"$targetpngfilename"
 	
done

# script finished

