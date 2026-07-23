dirname=$(dirname $BASH_SOURCE)
pushd $dirname > /dev/null
abspath=$(pwd)
export VID_SDK_ROOT=$abspath
export VIDEANTIS_LICENSE_PATH=$VID_SDK_ROOT/license
export VMPCC_ROOT=$VID_SDK_ROOT
export VSPGCC_ROOT=$VID_SDK_ROOT
export PKG_CONFIG_PATH=$VID_SDK_ROOT/lib/pkgconfig
export VIDSDK_DIR=$VID_SDK_ROOT
export PATH=$VID_SDK_ROOT/bin:$PATH
popd > /dev/null

