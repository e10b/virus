# virus

WebGPU viral dynamics playground built on **[wgfx](https://github.com/e10b/wgfx)** and **Dear ImGui**.

The app runs a CPU double-buffered cellular automata infection model and renders the grid with WGSL. Presets include COVID, Hantavirus, and Black Plague, with sliders for spread probability, mortality, population density, initial cases, disease duration, simulation speed, zoom, and pan.

## Clone

```bash
git clone --recurse-submodules git@github.com:e10b/virus.git
cd virus
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

```bash
cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out -j
./out/App
```

## Web Build

```bash
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web -j
python3 -m http.server 8080 -d build-web
```

Then open `http://localhost:8080/App.html` in a WebGPU-enabled browser.

## Layout

- `deps/wgfx`: WebGPU, SDL3, GLM, and graphics helpers
- `deps/imgui`: Dear ImGui
- `example/`: app entry point, SDL context, and viral simulation logic
- `res/`: WGSL shader for the viral grid renderer
