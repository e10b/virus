struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct ViralUniform {
    params: vec4f, // x=gridSize, y=aspect, z=time
    view: vec4f,   // x=zoom, y=panX, z=panY, w=populationDensity
    covid: vec4f,  // x=spread, y=mortality, z=reserved, w=model(0=covid,1=hanta,2=plague,3=custom)
    hanta: vec4f,  // reserved
};

struct ViralCells {
    data: array<vec4f>,
};

@group(0) @binding(0) var<uniform> u: ViralUniform;
@group(0) @binding(1) var<storage, read> cells: ViralCells;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

fn hash32(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn randCell(i: u32) -> f32 {
    return f32(hash32(i ^ 0x68bc21ebu) & 0x00ffffffu) / 16777216.0;
}

fn sampleCell(pos01: vec2f, n: i32) -> vec4f {
    let p = clamp(pos01, vec2f(0.0), vec2f(1.0));
    let ix = clamp(i32(floor(p.x * f32(n))), 0, n - 1);
    let iy = clamp(i32(floor(p.y * f32(n))), 0, n - 1);
    return cells.data[u32(iy * n + ix)];
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let n = max(2, i32(round(u.params.x)));
    let aspect = max(u.params.y, 0.001);
    var p = input.uv;
    p.x *= aspect;
    p /= max(u.view.x, 0.001);
    p -= u.view.yz;
    let pos01 = p * 0.5 + vec2f(0.5);

    let bg = vec3f(0.006, 0.007, 0.013);
    if (pos01.x < 0.0 || pos01.x > 1.0 || pos01.y < 0.0 || pos01.y > 1.0) {
        return vec4f(bg, 1.0);
    }

    let cell = sampleCell(pos01, n);
    let state = i32(round(cell.x));
    let days01 = clamp(cell.y, 0.0, 1.0);
    let model = u.covid;
    let mortality = clamp(model.y, 0.0, 1.0);

    let caughtAliveColor = vec3f(1.0, 0.82, 0.08);
    let deathColor = vec3f(0.58, 0.02, 0.035);

    var color = bg;
    if (state == 1) {
        color = vec3f(0.070, 0.073, 0.080);
    } else if (state == 2) {
        color = mix(caughtAliveColor * 0.72, caughtAliveColor, 0.35 + 0.40 * days01);
    } else if (state == 3) {
        color = vec3f(0.052, 0.057, 0.064);
    } else if (state == 4) {
        color = deathColor;
    }

    return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
