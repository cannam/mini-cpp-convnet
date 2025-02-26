#!/bin/bash

set -eu

mydir=$(dirname "$0")
cd "$mydir"

if [ -d data ]; then
    echo "*** Data directory exists, not recreating it"
else
    if [ -d flower_photos ]; then
        echo "*** Source image directory flower_photos exists, not re-downloading"
    else
        echo "*** Downloading source images..."
        if ! ( wget --help 2>/dev/null | grep -q Wget ) ; then
            echo "*** ERROR: Wget utility required"
            exit 2
        fi
        wget http://download.tensorflow.org/example_images/flower_photos.tgz
        echo "*** Unpacking source images..."
        tar xvf flower_photos.tgz
    fi
    echo "*** Converting and scaling datasets..."
    if ! ( magick --help 2>/dev/null | grep -q ImageMagick ) ; then
        echo "*** ERROR: ImageMagick \"convert\" utility required"
        exit 2
    fi
    mkdir -p data/train data/test
    ( cd flower_photos
      for infile in */[0-9]*.jpg ; do
          echo -n "."
          group=""
	  case "$infile" in
              */[0-5]*) group=train;;
              */[6-7]*) group=validate;;
              */[8-9]*) group=test;;
          esac
	  if [ -z "$group" ]; then
	      echo "*** ERROR: Unexpected filename \"$infile\""
	      exit 2
	  fi
          category=${infile%%/*}
          base=$(basename "$infile" .jpg)
          mkdir -p "../data/$group/$category"
          magick "$infile" -strip -resize 128x128 -gravity center -extent 128x128 "../data/$group/$category/$base.png"
      done
    )
    echo
    echo "*** Conversion finished"
fi
