# Web Build

```bash
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web -j
python3 -m http.server 8080 -d build-web
```

Open `http://localhost:8080/App.html` in a WebGPU-enabled browser.
