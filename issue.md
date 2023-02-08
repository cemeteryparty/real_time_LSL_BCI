# Real-Time-CLEEGN Issue

In this document, we will mention some potential issue you may meet after installing [StreamViewer](https://github.com/intheon/stream_viewer)

## LSL binary library file not found error (unix)

**1. run `stream_viewer`**

```sh
lsl_viewer
```

returned error message:

```txt
RuntimeError: LSL binary library file was not found. Please make sure that the binary file can be found in the package lib folder
...
You can install the LSL library with conda: `conda install -c conda-forge liblsl`
or otherwise download it from the liblsl releases page assets: https://github.com/sccn/liblsl/releases
```

**2. install `liblsl`**

```bash
conda install -c conda-forge liblsl
```

## `xcb` Qt plugin fails to load

**1. run `stream_viewer`**

```sh
lsl_viewer
```

returned error message:

```txt
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
...
Aborted (core dumped)
```

**2. activate the qt debug mode** 

```sh
vim ~/.bashrc
```

write following line to `~/.bashrc`

```txt
export QT_DEBUG_PLUGINS=1  # dump all Qt msg"
```

promit the change

```bash
source ~/.bashrc
```

**3. identify the error**

```sh
lsl_viewer
```

returned debug message:

```txt
...
Cannot load library LOCATION/DynamicLibFILE ...
...
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
...
Aborted (core dumped)
```

check dependency

```sh
cd LOCATION
ldd DynamicLibFILE  # check dependency
```

returned output (example) 

```txt
...
    libxcb-xinerama.so.0 => not found
...
```

install (example)

```sh
sudo apt install libxcb-xinerama0  # install not found package
# remember to resume the ~/.bashrc
```

## visbrain issue

solved? todo?

## libGL Error

update graphic driver
