#!/bin/bash

#++++++++++++++++++++++++++++++ FileHeaderBegin +++++++++++++++++++++++++++++++
#
# CONFIDENTIAL AND PROPRIETARY INFORMATION
# Copyright 2004 - 2021 videantis GmbH
# All Rights Reserved
#
# This document contains confidential and proprietary information of videantis
# GmbH and is protected by copyright, trade secret and other local, state,
# federal, and international laws. Its receipt or possession does not convey
# any rights to reproduce, transfer, disclose or publish its contents, or to
# manufacture, commercially or non-commercially use or sell anything it may
# describe or contain. Reproduction, disclosure or any use without specific
# written authorization of videantis GmbH or an individual license agreement
# with videantis GmbH is strictly forbidden.
#
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
#  FILENAME:      finalize_vcnn.sh
# 
#  DESCRIPTION:   bash script to finalize VCNN file:
#                 - add vmp binary
#                 - set target addresses according to vmpcode.tab
# 
# ++++++++++++++++++++++++++++++ FileHeaderEnd +++++++++++++++++++++++++++++++


CNNINFO_TOOL=../../vid_cnninfo/cnninfo_tool

VCNN_FILE=$1

if [ "$VCNN_FILE" == "" ] ; then
    echo "Usage: finalize_vcnn.sh <vcnn_file>"
    exit 1
fi

if [ ! -e "$VCNN_FILE" ] ; then
    echo "Cannot open $VCNN_FILE"
    exit 1
fi

echo Finalizing $VCNN_FILE

SRCDIR=./src
OBJDIR=./obj
BINDIR=./bin
GENSRCDIR=$SRCDIR/generated

CODETAB_FILE=$GENSRCDIR/vmpcode.tab
VMPMAP_FILE=$OBJDIR/vmp.map
VMPBIN_FILE=$BINDIR/vmp.bin
BUILDTS_FILE=$GENSRCDIR/build_timestamp.inc

check_overwrite() {
    if [ -e "$1" ] ; then
	read -p "About to overwrite $1 - Continue (y/n)? " CONT
	if [[ ! $CONT =~ ^[Yy]$ ]] ; then
	    exit 1
	fi
    fi
}

get_vmp_code() {
    cat $VMPMAP_FILE | grep "^imem:" | grep "routine_$(printf %02d $1):start:0x.*$" | sed "s/.*://"
}

get_spad_code() {
    if [ "$2" == "1" ] ; then
	cat $VMPMAP_FILE | grep "^imem:" | grep "spad_$(printf %02d $1)_first:0x.*$" | sed "s/.*://"
    elif [ "$2" == "2" ] ; then
	cat $VMPMAP_FILE | grep "^imem:" | grep "spad_$(printf %02d $1)_rest:0x.*$" | sed "s/.*://"
    else 
	echo "Usage: get_spad_code [idx] [1|2]"
	exit 1
    fi
}

echo
echo "# Import $VMPBIN_FILE into $VCNN_FILE"
CMD="$CNNINFO_TOOL -a -q $VCNN_FILE cnn.vmpbin=$VMPBIN_FILE"
echo \$ $CMD
$CMD

echo
echo "# Import $VMPMAP_FILE into $VCNN_FILE"
CMD="$CNNINFO_TOOL -a -q $VCNN_FILE cnn.vmpmap=$VMPMAP_FILE"
echo \$ $CMD
$CMD

echo
echo "# Set vmp_codes using $CODETAB_FILE and $VMPMAP_FILE"
PARAMS=""
LC=0
while read -r line ; do
    LC=$((LC+1))
    # ignore #-prefixed comments and blank lines:
    if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ ! $line =~ ^[[:space:]]*$ ]] ; then
	LAYER=$(echo $line | cut -f1 -d' ')
	PART=$(echo $line | cut -f2 -d' ')
	IDX=$(echo $line | cut -f3 -d' ')
	SPAD=$(echo $line | cut -f4 -d' ')

	if [[ ! "$LAYER" =~ [0-9]+ ]] || [[ ! "$PART" =~ [0-9]+ ]] || [[ ! "$IDX" =~ [0-9]+ ]] || [[ ! "$SPAD" =~ [0-9]+ ]] ; then
	    echo Syntax error in $CODETAB_FILE:
	    echo Line $LC: $line
	    exit 1
	fi

	VMP_CODE=$(get_vmp_code $IDX)
	if [[ ! "$VMP_CODE" =~ ^0x.* ]] ; then
	    echo Failed to get vmp_code for layer=$LAYER part=$PART idx=$IDX from $VMPMAP_FILE
	    exit 1
	fi
	PARAMS=${PARAMS:+$PARAMS' '}"cnn.layer[$LAYER].part[$PART].vmpCode=$VMP_CODE"

	if [[ ! "$SPAD" == "0" ]] ; then
	    SPAD1_CODE=$(get_spad_code $SPAD 1)
	    if [[ ! "$SPAD1_CODE" =~ ^0x.* ]] ; then
		echo Failed to get spad1_code for layer=$LAYER part=$PART idx=$IDX from $VMPMAP_FILE
		exit 1
	    fi
	    SPAD2_CODE=$(get_spad_code $SPAD 2)
	    if [[ ! "$SPAD2_CODE" =~ ^0x.* ]] ; then
		echo Failed to get spad2_code for layer=$LAYER part=$PART idx=$IDX from $VMPMAP_FILE
		exit 1
	    fi
	    PARAMS=${PARAMS:+$PARAMS' '}"cnn.layer[$LAYER].part[$PART].spad1Code=$SPAD1_CODE"
	    PARAMS=${PARAMS:+$PARAMS' '}"cnn.layer[$LAYER].part[$PART].spad2Code=$SPAD2_CODE"
	fi

    fi
done < $CODETAB_FILE

echo "\$ $CNNINFO_TOOL -q $VCNN_FILE \\"
for x in $PARAMS ; do
    echo "                  $x \\"
done
echo
$CNNINFO_TOOL -q $VCNN_FILE $PARAMS

echo
echo "# Set build timestamp and mark structure as ready to run"
BUILD_TIMESTAMP=$(grep "#define BUILD_TIMESTAMP" $BUILDTS_FILE | sed "s/^.*BUILD_TIMESTAMP //")
if [ "$BUILD_TIMESTAMP" == "" ] ; then
    echo Failed to get build timestamp from $BUILDTS_FILE
    exit 1
fi
CMD="$CNNINFO_TOOL -q $VCNN_FILE cnn.vmpbinBuild=$BUILD_TIMESTAMP cnn.flags=0x00000001"
echo \$ $CMD
$CMD

echo
echo "Successfully finalized $VCNN_FILE"
echo

