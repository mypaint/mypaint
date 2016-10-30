#!/bin/bash

########################################################################
# Package the binaries built on Travis-CI as an AppImage
# By Simon Peter 2016
# For more information, see http://appimage.org/
########################################################################

export ARCH=$(arch)

APP=MyPaint
LOWERAPP=${APP,,}

GIT_REV=$(git rev-parse --short HEAD)
echo $GIT_REV

mkdir -p $HOME/$APP/$APP.AppDir/usr/

cd $HOME/$APP/

wget -q https://github.com/probonopd/AppImages/raw/master/functions.sh -O ./functions.sh
. ./functions.sh

cd $APP.AppDir

sudo chown -R $USER /app/

cp -r /app/* ./usr/
BINARY=$(find ./usr/bin/ -name mypaint -type f -executable | head -n 1)

########################################################################
# Copy desktop and icon file to AppDir for AppRun to pick them up
########################################################################

get_apprun
get_desktop
get_icon

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

########################################################################
# desktopintegration asks the user on first run to install a menu item
########################################################################

get_desktopintegration $LOWERAPP

########################################################################
# Determine the version of the app; also include needed glibc version
########################################################################

GLIBC_NEEDED=$(glibc_needed)
VERSION=${RELEASE_VERSION}-glibc$GLIBC_NEEDED

########################################################################
# Other appliaction-specific finishing touches
########################################################################

# TODO: Bundle Python and all the plugins needed

mkdir -p ./$APP/$APP.AppDir/usr/lib

cd ./$APP/

wget -q https://github.com/probonopd/AppImages/raw/master/functions.sh -O ./functions.sh
. ./functions.sh

generate_status

echo "deb http://archive.ubuntu.com/ubuntu/ trusty main universe" > sources.list
apt-get $OPTIONS update
URLS=$(apt-get $OPTIONS -y install --print-uris python-gi gir1.2-gtk-3.0 python-gi-cairo | cut -d "'" -f 2 | grep -e "^http")
wget -c $URLS

cd ./$APP.AppDir/

find ../*.deb -exec dpkg -x {} . \; || true

########################################################################
# Patch away absolute paths; it would be nice if they were relative
########################################################################

patch_usr
# sed -i -e 's|././|usr|g' $BINARY # unpatch

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
