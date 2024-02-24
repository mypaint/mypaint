#!/bin/bash

########################################################################
# Package the binaries built on Travis-CI as an AppImage
# By Simon Peter 2016
# For more information, see http://appimage.org/
########################################################################

export ARCH=$(arch)
eval `python2 lib/meta.py`

APP=MyPaint
LOWERAPP=${APP,,}

mkdir -p $HOME/$APP/$APP.AppDir/usr/

cd $HOME/$APP/

wget -q https://github.com/probonopd/AppImages/raw/master/functions.sh -O ./functions.sh
. ./functions.sh

cd $APP.AppDir

sudo chown -R $USER /app/

cp -r /app/* ./usr/

########################################################################
# Copy desktop and icon file to AppDir for AppRun to pick them up
########################################################################

get_apprun
get_desktop
get_icon

########################################################################
# Other appliaction-specific finishing touches
########################################################################

# Bundle Python and all the plugins needed

cd ..

generate_status

echo "deb http://archive.ubuntu.com/ubuntu/ trusty main universe" > sources.list
apt-get $OPTIONS update
URLS=$(apt-get $OPTIONS -y install --print-uris python-gi gir1.2-gtk-3.0 python-gi-cairo libgtk-3-0 python-numpy unity-gtk3-module libcanberra-gtk3-module | cut -d "'" -f 2 | grep -e "^http")
wget -c $URLS

cd ./$APP.AppDir/

find ../*.deb -exec dpkg -x {} . \; || true

# Workaround for:
# python2.7: symbol lookup error: /usr/lib/x86_64-linux-gnu/libgtk-3.so.0: undefined symbol: gdk__private__

cp /usr/lib/x86_64-linux-gnu/libg*k-3.so.0 usr/lib/x86_64-linux-gnu/

# Compile Glib schemas
( mkdir -p usr/share/glib-2.0/schemas/ ; cd usr/share/glib-2.0/schemas/ ; glib-compile-schemas . )

########################################################################
# Copy in the dependencies that cannot be assumed to be available
# on all target systems
########################################################################

copy_deps

########################################################################
# Delete stuff that should not go into the AppImage
########################################################################

# Delete dangerous libraries; see
# https://github.com/probonopd/AppImages/blob/master/excludelist
delete_blacklisted
find . -name *harfbuzz* -delete

########################################################################
# desktopintegration asks the user on first run to install a menu item
########################################################################

get_desktopintegration $LOWERAPP

########################################################################
# Determine the version of the app; also include needed glibc version
########################################################################

VERSION=$(echo "$MYPAINT_VERSION_CEREMONIAL" | sed -e 's/alpha+gitexport/git/')

########################################################################
# Patch away absolute paths; it would be nice if they were relative
########################################################################

find usr/ -type f -exec sed -i -e 's|/usr|././|g' {} \;
find usr/ -type f -exec sed -i -e 's@././/bin/env@/usr/bin/env@g' {} \;

########################################################################
# AppDir complete
# Now packaging it as an AppImage
########################################################################

cd .. # Go out of AppImage

mkdir -p ../out/
generate_type2_appimage

########################################################################
# Upload the AppDir
########################################################################

transfer ../out/*
echo "AppImage has been uploaded to the URL above; use something like GitHub Releases for permanent storage"
