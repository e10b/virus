// 3D TDSE volume ray-marcher
// Storage buffer layout: each cell is vec4f (rho, phase01, pot01, reserved)
// Grid is N x N x N, stored in z-major order: idx = iz*N*N + iy*N + ix

struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct Tdse3dUniform {
    invViewProj:    mat4x4f,  // inverse view-projection for ray generation
    camPos:         vec4f,    // xyz=camera world pos, w=time
    params:         vec4f,    // x=gridSize, y=domainHalf, z=intensityScale, w=intensityRange
    render:         vec4f,    // x=colorMode, y=aspect, z=showPotential, w=sliceAxis(-1=vol)
    march:          vec4f,    // x=stepCount, y=alphaScale, z=slicePos, w=reserved
};

@group(0) @binding(0) var<uniform>          u:        Tdse3dUniform;
@group(0) @binding(1) var<storage, read>    waveGrid: array<vec4f>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

// ---------------------------------------------------------------------------
// Grid sampling
// ---------------------------------------------------------------------------
fn gridIndex(ix: i32, iy: i32, iz: i32, N: i32) -> i32 {
    let cx = clamp(ix, 0, N - 1);
    let cy = clamp(iy, 0, N - 1);
    let cz = clamp(iz, 0, N - 1);
    return cz * N * N + cy * N + cx;
}

fn sampleRho(px: f32, py: f32, pz: f32, N: i32, domainHalf: f32) -> f32 {
    let h = domainHalf;
    let nx = (px / h) * 0.5 + 0.5;
    let ny = (py / h) * 0.5 + 0.5;
    let nz = (pz / h) * 0.5 + 0.5;
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0 || nz < 0.0 || nz > 1.0) {
        return 0.0;
    }
    let fx = nx * f32(N - 1);
    let fy = ny * f32(N - 1);
    let fz = nz * f32(N - 1);
    let ix = i32(floor(fx));
    let iy = i32(floor(fy));
    let iz = i32(floor(fz));
    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let tz = fz - f32(iz);

    let c000 = waveGrid[gridIndex(ix,   iy,   iz,   N)].x;
    let c100 = waveGrid[gridIndex(ix+1, iy,   iz,   N)].x;
    let c010 = waveGrid[gridIndex(ix,   iy+1, iz,   N)].x;
    let c110 = waveGrid[gridIndex(ix+1, iy+1, iz,   N)].x;
    let c001 = waveGrid[gridIndex(ix,   iy,   iz+1, N)].x;
    let c101 = waveGrid[gridIndex(ix+1, iy,   iz+1, N)].x;
    let c011 = waveGrid[gridIndex(ix,   iy+1, iz+1, N)].x;
    let c111 = waveGrid[gridIndex(ix+1, iy+1, iz+1, N)].x;

    let c00 = mix(c000, c100, tx);
    let c10 = mix(c010, c110, tx);
    let c01 = mix(c001, c101, tx);
    let c11 = mix(c011, c111, tx);
    let c0  = mix(c00,  c10,  ty);
    let c1  = mix(c01,  c11,  ty);
    return mix(c0, c1, tz);
}

fn samplePhase(px: f32, py: f32, pz: f32, N: i32, domainHalf: f32) -> f32 {
    let h = domainHalf;
    let nx = (px / h) * 0.5 + 0.5;
    let ny = (py / h) * 0.5 + 0.5;
    let nz = (pz / h) * 0.5 + 0.5;
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0 || nz < 0.0 || nz > 1.0) {
        return 0.0;
    }
    let ix = clamp(i32(floor(nx * f32(N - 1))), 0, N - 1);
    let iy = clamp(i32(floor(ny * f32(N - 1))), 0, N - 1);
    let iz = clamp(i32(floor(nz * f32(N - 1))), 0, N - 1);
    return waveGrid[gridIndex(ix, iy, iz, N)].y;
}

fn samplePot(px: f32, py: f32, pz: f32, N: i32, domainHalf: f32) -> f32 {
    let h = domainHalf;
    let nx = (px / h) * 0.5 + 0.5;
    let ny = (py / h) * 0.5 + 0.5;
    let nz = (pz / h) * 0.5 + 0.5;
    if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0 || nz < 0.0 || nz > 1.0) {
        return 0.0;
    }
    let ix = clamp(i32(floor(nx * f32(N - 1))), 0, N - 1);
    let iy = clamp(i32(floor(ny * f32(N - 1))), 0, N - 1);
    let iz = clamp(i32(floor(nz * f32(N - 1))), 0, N - 1);
    return waveGrid[gridIndex(ix, iy, iz, N)].z;
}

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------
fn hsvToRgb(h: f32, s: f32, v: f32) -> vec3f {
    let hue = fract(h) * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs(fract(hue * 0.5) * 2.0 - 1.0));
    let m = v - c;
    if (hue < 1.0) { return vec3f(c + m, x + m, m); }
    if (hue < 2.0) { return vec3f(x + m, c + m, m); }
    if (hue < 3.0) { return vec3f(m, c + m, x + m); }
    if (hue < 4.0) { return vec3f(m, x + m, c + m); }
    if (hue < 5.0) { return vec3f(x + m, m, c + m); }
    return vec3f(c + m, m, x + m);
}

fn samplePalette6(x: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f) -> vec3f {
    let u = clamp(x, 0.0, 1.0) * 5.0;
    if (u < 1.0) { return mix(c0, c1, u); }
    if (u < 2.0) { return mix(c1, c2, u - 1.0); }
    if (u < 3.0) { return mix(c2, c3, u - 2.0); }
    if (u < 4.0) { return mix(c3, c4, u - 3.0); }
    return mix(c4, c5, u - 4.0);
}

fn mapColorMode(t: f32, colorMode: i32, phase01: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    if (colorMode == 0) { // Inferno
        return samplePalette6(x, vec3f(0.0015,0.0005,0.0139), vec3f(0.1462,0.0449,0.3374), vec3f(0.3904,0.1004,0.5019), vec3f(0.6663,0.1819,0.3698), vec3f(0.9018,0.4251,0.1081), vec3f(0.9884,0.9984,0.6449));
    }
    if (colorMode == 1) { // Magma
        return samplePalette6(x, vec3f(0.0015,0.0005,0.0139), vec3f(0.1717,0.0673,0.3708), vec3f(0.4452,0.1227,0.5069), vec3f(0.7164,0.2150,0.4753), vec3f(0.9440,0.3776,0.3651), vec3f(0.9871,0.9914,0.7495));
    }
    if (colorMode == 2) { // Plasma
        return samplePalette6(x, vec3f(0.0504,0.0298,0.5280), vec3f(0.4176,0.0006,0.6584), vec3f(0.6928,0.1651,0.5645), vec3f(0.8814,0.3925,0.3832), vec3f(0.9883,0.6523,0.2114), vec3f(0.9400,0.9752,0.1313));
    }
    if (colorMode == 3) { // Viridis
        return samplePalette6(x, vec3f(0.2670,0.0049,0.3294), vec3f(0.2539,0.2653,0.5300), vec3f(0.1636,0.4711,0.5581), vec3f(0.1347,0.6586,0.5176), vec3f(0.4775,0.8214,0.3182), vec3f(0.9932,0.9062,0.1439));
    }
    if (colorMode == 6) { // Gray
        return vec3f(x, x, x);
    }
    // Phase (colorMode == 9 or default for 3D): HSV from phase angle
    return hsvToRgb(phase01, 0.92, 0.18 + 0.82 * x);
}

// ---------------------------------------------------------------------------
// Box intersection: returns (tNear, tFar), tFar < 0 means miss
// ---------------------------------------------------------------------------
fn boxIntersect(ro: vec3f, rd: vec3f, boxMin: vec3f, boxMax: vec3f) -> vec2f {
    let invD = 1.0 / rd;
    let t0 = (boxMin - ro) * invD;
    let t1 = (boxMax - ro) * invD;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let tNear = max(max(tmin.x, tmin.y), tmin.z);
    let tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(tNear, tFar);
}

// ---------------------------------------------------------------------------
// Fragment: volume ray-march
// ---------------------------------------------------------------------------
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let N           = max(4, i32(round(u.params.x)));
    let domainHalf  = max(u.params.y, 0.5);
    let iScale      = max(u.params.z, 0.0001);
    let iRange      = max(u.params.w, 0.0001);
    let colorMode   = i32(round(u.render.x));
    let showPot     = u.render.z > 0.5;
    let sliceAxis   = i32(round(u.render.w));  // 0=x,1=y,2=z,-1=volume
    let slicePos    = u.march.z;
    let steps       = max(4, i32(round(u.march.x)));
    let alphaScale  = max(u.march.y, 0.001);
    let time        = u.camPos.w;

    // --- Slice mode ---
    if (sliceAxis >= 0) {
        var px = 0.0; var py = 0.0; var pz = 0.0;
        let aspect = max(u.render.y, 0.001);
        var uvs = input.uv;
        uvs.x *= aspect;
        let h = domainHalf;
        if (sliceAxis == 0) {
            px = slicePos; py = uvs.y * h; pz = uvs.x * h;
        } else if (sliceAxis == 1) {
            px = uvs.x * h; py = slicePos; pz = uvs.y * h;
        } else {
            px = uvs.x * h; py = uvs.y * h; pz = slicePos;
        }
        let rho   = sampleRho(px, py, pz, N, domainHalf);
        let phase = samplePhase(px, py, pz, N, domainHalf);
        let pot   = samplePot(px, py, pz, N, domainHalf);
        let scaled = rho * iScale * 200.0;
        let bright = clamp(scaled / (scaled + iRange), 0.0, 1.0);
        var col = mapColorMode(bright, colorMode, phase);
        if (showPot) {
            let potCol = mix(vec3f(0.15,0.45,0.95), vec3f(0.95,0.25,0.10), pot);
            col = mix(col, potCol, 0.35 * (0.2 + 0.8 * pot));
        }
        let bg = vec3f(0.01, 0.01, 0.02);
        return vec4f(mix(bg, col, bright), 1.0);
    }

    // --- Volume ray-march mode ---
    // Reconstruct world-space ray from inverse view-projection
    let ndcPos = vec4f(input.uv.x, input.uv.y, 1.0, 1.0);
    let worldH = u.invViewProj * ndcPos;
    let worldPt = worldH.xyz / worldH.w;
    let camPos = u.camPos.xyz;
    let rd = normalize(worldPt - camPos);
    let ro = camPos;

    let boxMin = vec3f(-domainHalf);
    let boxMax = vec3f( domainHalf);
    let hit = boxIntersect(ro, rd, boxMin, boxMax);
    if (hit.y < hit.x || hit.y < 0.0) {
        return vec4f(0.01, 0.01, 0.02, 1.0);
    }

    let tStart = max(hit.x, 0.0);
    let tEnd   = hit.y;
    let stepSize = (tEnd - tStart) / f32(steps);

    var accColor = vec3f(0.0);
    var accAlpha = 0.0;
    var t = tStart + stepSize * 0.5;

    var i = 0;
    loop {
        if (i >= steps || accAlpha >= 0.99) { break; }

        let p = ro + rd * t;
        let rho   = sampleRho(p.x, p.y, p.z, N, domainHalf);
        let phase = samplePhase(p.x, p.y, p.z, N, domainHalf);
        let pot   = samplePot(p.x, p.y, p.z, N, domainHalf);

        let scaled = rho * iScale * 200.0;
        let bright = clamp(scaled / (scaled + iRange), 0.0, 1.0);

        var col = mapColorMode(bright, colorMode, phase);
        if (showPot) {
            let potCol = mix(vec3f(0.15,0.45,0.95), vec3f(0.95,0.25,0.10), pot);
            col = mix(col, potCol, 0.25 * pot);
        }

        let sampleAlpha = clamp(bright * alphaScale * stepSize, 0.0, 1.0);
        accColor += col * sampleAlpha * (1.0 - accAlpha);
        accAlpha += sampleAlpha * (1.0 - accAlpha);

        t += stepSize;
        i += 1;
    }

    let bg = vec3f(0.01, 0.01, 0.02);
    let finalColor = mix(bg, accColor / max(accAlpha, 0.001), accAlpha);
    return vec4f(finalColor, 1.0);
}
