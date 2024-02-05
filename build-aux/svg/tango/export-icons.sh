#!/bin/sh
# by Deevad
# Mypaint icon exporter 2012-01-30
# --------------------------------
# what it does : export *.svgz Mypaint icons templates  ( from mypaint/svg folder ) to the right folders automagically with right names. Finish long export time with Inkscape !
# what it needs : Inkscape installed.
# how to use : save your *.svgz file ( done from template ) in mypaint/svg with the background layer hidden, then call the script. 
# ./0_icon-exporter.sh %f
# --------------------------------


if [ $# -eq 0 ]; then
    echo "usage: icon-exporter [SVGZFILE [...]]"
    echo 
    echo "SVG files must have a hidden layer with rects having the IDs"
    echo "16x16, 22x22, 24x24, 32x32, and 48x48. SVG export involves"
    echo "manual steps; see the instructions in the terminal output."
    exit 0
fi

OUTPUT_DIR=../desktop/icons/hicolor


for arg in "$@"; do
    targetpngfilename=$(echo $arg|sed 's/\(.*\)\..\+/\1/')".png"
    targetsvgfilename=$(echo $arg|sed 's/\(.*\)\..\+/\1/')".svg"
    targetsvgzfilename="${targetsvgfilename}.tmp.svgz"

    for s in 16x16 22x22 24x24 32x32 48x48; do
        echo "Exporting $s slice to PNG..."
 	    inkscape -f "$arg" -i $s -y 0 -z -e "$targetpngfilename"
 	    mv -v "$targetpngfilename" "$OUTPUT_DIR/$s/actions/$targetpngfilename"
    done
 	
    # Scalable SVG from the 48x48 slice
    # Begin by cropping the page to the slice rect
    cp "$arg" "$targetsvgzfilename"
    echo "Scripted page resize to the 48x48 slice..."
    inkscape --select=48x48 --verb FitCanvasToSelectionOrDrawing \
        --verb EditSelectAll --verb EditDelete \
        --verb FileSave --verb FileClose \
        "$targetsvgzfilename"
    # Make plain SVG: simpler and quicker to edit.
    inkscape --export-plain-svg="$targetsvgfilename" -z "$targetsvgzfilename"
    rm -v -f "$targetsvgzfilename"
    # Final cleanup step has to be done manually.
    echo "*** Now manually remove unwanted elements, Vacuum Defs, and save"
    inkscape --verb ZoomDrawing --verb EditSelectAllInAllLayers \
            --verb SelectionUnGroup --verb EditDeselect \
            $targetsvgfilename
 	mv -v "$targetsvgfilename" "$OUTPUT_DIR/scalable/actions/$targetsvgfilename"
done
