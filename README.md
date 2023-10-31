## pkg-template

> template for an Uqbar package


A package consists of one or more wasm processes. These are installed through app_tracker, which can give capabilities.


### run

build: `python3 build.py`

start: `python3 start-package.py http://localhost:8080 pkg/`

This template currently boots a package `template:bitful.uq`, it contains 2 processes, `hello1` and `hello2`.

To build all processes in `manifest.json`, run `python3 build`, you can specify a specific rust directory with `python3 build <dir>`too.

If you have an existing wasm process built that you want to boot, put its compiled .wasm in `pkg`, and run start-package with your nodes url, name, and the /pkg directory as arguments!

>structure
```
/
├─ pkg/
│  ├─ manifest.json
│  ├─ metadata.json
│  ├─ hello1.wasm
│  ├─ hello2.wasm
├─ hello1/
│  ├─ src/lib.rs
│  ├─ ...
├─ hello2/
│  ├─ src/lib.rs
│  ├─ ...
```

