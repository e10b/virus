struct VertexInput {
    @location(0) position: vec3f,
    @builtin(vertex_index) vertexIndex: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) visible: f32,
};

struct OrbitalUniform {
    viewProj: mat4x4f,
    clipOrigin: vec4f,
    quantum: vec4f, // x:n, y:l, z:m, w:colorMode
    render: vec4f,  // x:removedOctant, y:intensityScale, z:simulationTime, w:seed
};

@group(0) @binding(0) var<uniform> orbital: OrbitalUniform;

const RADIAL_BINS: i32 = 192;
const THETA_BINS: i32 = 160;

fn hash32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    (*state) = hash32((*state) + 0x9e3779b9u);
    let mantissa = (*state) & 0x00ffffffu;
    return f32(mantissa) / 16777216.0;
}

fn associatedLaguerre(k: i32, alpha: i32, x: f32) -> f32 {
    if (k <= 0) {
        return 1.0;
    }
    var lm2 = 1.0;
    var lm1 = 1.0 + f32(alpha) - x;
    if (k == 1) {
        return lm1;
    }

    var l = lm1;
    var j = 2;
    loop {
        if (j > k) {
            break;
        }
        l = ((f32(2 * j - 1 + alpha) - x) * lm1 - f32(j - 1 + alpha) * lm2) / f32(j);
        lm2 = lm1;
        lm1 = l;
        j += 1;
    }
    return l;
}

fn associatedLegendre(l: i32, mAbs: i32, x: f32) -> f32 {
    var pmm = 1.0;
    if (mAbs > 0) {
        let somx2 = sqrt(max(0.0, (1.0 - x) * (1.0 + x)));
        var fact = 1.0;
        var j = 1;
        loop {
            if (j > mAbs) {
                break;
            }
            pmm = pmm * (-fact) * somx2;
            fact += 2.0;
            j += 1;
        }
    }

    if (l == mAbs) {
        return pmm;
    }

    var pm1m = x * f32(2 * mAbs + 1) * pmm;
    if (l == mAbs + 1) {
        return pm1m;
    }

    var pll = pm1m;
    var ll = mAbs + 2;
    loop {
        if (ll > l) {
            break;
        }
        pll = ((f32(2 * ll - 1) * x * pm1m) - (f32(ll + mAbs - 1) * pmm)) / f32(ll - mAbs);
        pmm = pm1m;
        pm1m = pll;
        ll += 1;
    }
    return pll;
}

fn factorialInt(v: i32) -> f32 {
    if (v <= 1) {
        return 1.0;
    }
    var out = 1.0;
    var i = 2;
    loop {
        if (i > v) {
            break;
        }
        out *= f32(i);
        i += 1;
    }
    return out;
}

fn radialPdf(r: f32, n: i32, l: i32) -> f32 {
    let nn = max(f32(n), 1.0);
    let rho = 2.0 * r / nn;
    let k = n - l - 1;
    let alpha = 2 * l + 1;
    let lag = associatedLaguerre(max(k, 0), alpha, rho);
    let num = factorialInt(max(n - l - 1, 0));
    let den = factorialInt(max(n + l, 0));
    let norm = pow(2.0 / nn, 3.0) * num / (2.0 * nn * max(den, 1.0));
    let radial = sqrt(max(norm, 0.0)) * exp(-rho * 0.5) * pow(max(rho, 0.0001), f32(l)) * lag;
    return max(0.0, r * r * radial * radial);
}

fn thetaPdf(theta: f32, l: i32, mAbs: i32) -> f32 {
    let x = cos(theta);
    let plm = associatedLegendre(l, mAbs, x);
    return max(0.0, sin(theta) * plm * plm);
}

fn sampleRFromCDF(u: f32, n: i32, l: i32) -> f32 {
    let rMax = 10.0 * f32(n * n);
    let dr = rMax / f32(RADIAL_BINS - 1);
    var cdf: array<f32, 192>;
    var sum = 0.0;

    var i = 0;
    loop {
        if (i >= RADIAL_BINS) {
            break;
        }
        let r = f32(i) * dr;
        sum += radialPdf(r, n, l);
        cdf[i] = sum;
        i += 1;
    }

    if (sum <= 1e-8) {
        return 0.0;
    }

    let sampleTarget = u * sum;
    var lo = 0;
    var hi = RADIAL_BINS - 1;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2;
        if (cdf[mid] < sampleTarget) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo <= 0) {
        return 0.0;
    }

    let prev = lo - 1;
    let c0 = cdf[prev];
    let c1 = cdf[lo];
    let denom = max(c1 - c0, 1e-8);
    let t = clamp((sampleTarget - c0) / denom, 0.0, 1.0);
    return (f32(prev) + t) * dr;
}

fn sampleThetaFromCDF(u: f32, l: i32, mAbs: i32) -> f32 {
    let dTheta = 3.14159265358979323846 / f32(THETA_BINS - 1);
    var cdf: array<f32, 160>;
    var sum = 0.0;

    var i = 0;
    loop {
        if (i >= THETA_BINS) {
            break;
        }
        let theta = f32(i) * dTheta;
        sum += thetaPdf(theta, l, mAbs);
        cdf[i] = sum;
        i += 1;
    }

    if (sum <= 1e-8) {
        return 0.0;
    }

    let sampleTarget = u * sum;
    var lo = 0;
    var hi = THETA_BINS - 1;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2;
        if (cdf[mid] < sampleTarget) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo <= 0) {
        return 0.0;
    }

    let prev = lo - 1;
    let c0 = cdf[prev];
    let c1 = cdf[lo];
    let denom = max(c1 - c0, 1e-8);
    let t = clamp((sampleTarget - c0) / denom, 0.0, 1.0);
    return (f32(prev) + t) * dTheta;
}

fn sphericalToCartesian(r: f32, theta: f32, phi: f32) -> vec3f {
    return vec3f(
        r * sin(theta) * cos(phi),
        r * cos(theta),
        r * sin(theta) * sin(phi)
    );
}

fn inRemovedOctant(pos: vec3f, origin: vec3f, removedOctant: i32) -> bool {
    let v = pos - origin;
    if (removedOctant == 1) { return v.x >= 0.0 && v.y >= 0.0 && v.z >= 0.0; }
    if (removedOctant == 2) { return v.x < 0.0 && v.y >= 0.0 && v.z >= 0.0; }
    if (removedOctant == 3) { return v.x < 0.0 && v.y < 0.0 && v.z >= 0.0; }
    if (removedOctant == 4) { return v.x >= 0.0 && v.y < 0.0 && v.z >= 0.0; }
    if (removedOctant == 5) { return v.x >= 0.0 && v.y >= 0.0 && v.z < 0.0; }
    if (removedOctant == 6) { return v.x < 0.0 && v.y >= 0.0 && v.z < 0.0; }
    if (removedOctant == 7) { return v.x < 0.0 && v.y < 0.0 && v.z < 0.0; }
    return v.x >= 0.0 && v.y < 0.0 && v.z < 0.0;
}

fn intensityAt(pos: vec3f, n: i32, l: i32, m: i32, intensityScale: f32, intensityRange: f32) -> f32 {
    let r = length(pos);
    let theta = acos(clamp(pos.y / max(r, 0.0001), -1.0, 1.0));

    let rho = 2.0 * r / max(f32(n), 1.0);
    let k = n - l - 1;
    let alpha = 2 * l + 1;
    let laguerre = associatedLaguerre(max(k, 0), alpha, rho);
    let nn = max(f32(n), 1.0);
    let num = factorialInt(max(n - l - 1, 0));
    let den = factorialInt(max(n + l, 0));
    let norm = pow(2.0 / nn, 3.0) * num / (2.0 * nn * max(den, 1.0));
    let radial = sqrt(max(norm, 0.0)) * exp(-rho * 0.5) * pow(max(rho, 0.0001), f32(l)) * laguerre;
    let radialP = radial * radial;

    let x = cos(theta);
    let mAbs = abs(m);
    let plm = associatedLegendre(l, mAbs, x);
    let angular = plm * plm;

    let raw = radialP * angular;
    // Keep brightness roughly consistent as n grows without washing out to white.
    let nNorm = pow(max(f32(n), 1.0), 3.0);
    let scaled = raw * max(intensityScale, 0.0001) * 30.0 * nNorm;
    // intensityRange controls highlight compression; lower values are brighter, higher values preserve contrast.
    return clamp(scaled / (scaled + max(intensityRange, 0.0001)), 0.0, 1.0);
}

fn mapColor(t: f32, colorMode: i32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    if (colorMode == 1) {
        return vec3f(x, x, x);
    }
    if (colorMode == 2) {
        return vec3f(0.2 + 0.8 * x, 1.0 - 0.6 * x, 1.0);
    }

    let c0 = vec3f(0.0, 0.0, 0.0);
    let c1 = vec3f(0.5, 0.0, 0.99);
    let c2 = vec3f(0.8, 0.0, 0.0);
    let c3 = vec3f(1.0, 0.5, 0.0);
    let c4 = vec3f(1.0, 1.0, 0.0);
    let c5 = vec3f(1.0, 1.0, 1.0);

    let u = x * 5.0;
    if (u < 1.0) { return c0 + u * (c1 - c0); }
    if (u < 2.0) { return c1 + (u - 1.0) * (c2 - c1); }
    if (u < 3.0) { return c2 + (u - 2.0) * (c3 - c2); }
    if (u < 4.0) { return c3 + (u - 3.0) * (c4 - c3); }
    return c4 + (u - 4.0) * (c5 - c4);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;

    let n = i32(round(orbital.quantum.x));
    let l = i32(round(orbital.quantum.y));
    let m = i32(round(orbital.quantum.z));
    let colorMode = i32(round(orbital.quantum.w));

    let removedOctant = i32(round(orbital.render.x));
    let intensityScale = orbital.render.y;
    let simulationTime = orbital.render.z;
    let seed = u32(max(1.0, orbital.render.w));
    let intensityRange = orbital.clipOrigin.w;

    var rngState = hash32(input.vertexIndex ^ seed ^ 0x68bc21ebu);
    let r = sampleRFromCDF(rand01(&rngState), n, l);
    let theta = sampleThetaFromCDF(rand01(&rngState), l, abs(m));
    let phi = rand01(&rngState) * 6.28318530717958647692;

    let safeR = max(r, 0.04);
    let safeSin = max(abs(sin(theta)), 0.04);
    let omega = f32(m) / (safeR * safeSin);
    let advectedPhi = phi + simulationTime * omega;
    let samplePos = sphericalToCartesian(r, theta, advectedPhi);

    let hidden = inRemovedOctant(samplePos, orbital.clipOrigin.xyz, removedOctant);
    let t = intensityAt(samplePos, n, l, m, intensityScale, intensityRange);

    output.position = orbital.viewProj * vec4f(samplePos, 1.0);
    output.color = mapColor(t, colorMode);
    output.visible = select(1.0, 0.0, hidden);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.visible < 0.5) {
        discard;
    }
    return vec4f(input.color, 1.0);
}
