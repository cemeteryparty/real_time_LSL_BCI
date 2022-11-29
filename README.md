# Real-Time-CLEEGN

CLEEGN Real Time System

## Install

```bash
pip install git+https://github.com/intheon/stream_viewer.git
conda install -c conda-forge liblsl #linux

pip install prettytable
pip install tensorflow

lsl_viewer
```


## Issue: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found."`

```sh
lsl_viewer
## OUTPUT
# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
# ...
# Aborted (core dumped)
vim ~/.bashrc
# write "export QT_DEBUG_PLUGINS=1  # dump all Qt msg"
source ~/.bashrc

lsl_viewer
## OUTPUT
# ...
# Cannot load library LOCATION/DynamicLibFILE ...
# ...
# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
# ...
# Aborted (core dumped)
cd LOCATION
ldd DynamicLibFILE  # check dependency
## OUTPUT (may be different)
# ...
# 	libxcb-xinerama.so.0 => not found
# ...
sudo apt install libxcb-xinerama0  # install not found package
# remember to resume the ~/.bashrc
```
