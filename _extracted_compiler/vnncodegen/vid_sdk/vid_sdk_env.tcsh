#
set sourced=($_)
set dname=`dirname $sourced[2]`
pushd $dname > /dev/null
set abspath=`pwd`
setenv VID_SDK_ROOT $abspath
setenv VIDEANTIS_LICENSE_PATH $VID_SDK_ROOT/license
setenv VMPCC_ROOT $VID_SDK_ROOT
setenv VSPGCC_ROOT $VID_SDK_ROOT
setenv PKG_CONFIG_PATH $VID_SDK_ROOT/lib/pkgconfig
setenv VIDSDK_DIR $VID_SDK_ROOT
setenv PATH $VID_SDK_ROOT/bin:$PATH
popd > /dev/null
