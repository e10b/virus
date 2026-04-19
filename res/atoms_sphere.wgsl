struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.position = vec4f(input.position, 1.0);
    output.uv = (input.position.xy + 1.0) * 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Ray origin and screen-space ray direction for a simple pinhole camera.
    let ro = vec3f(0.0, 0.0, 3.0);
    var p = input.uv * 2.0 - vec2f(1.0, 1.0);
    p.x = p.x * (16.0 / 9.0);
    let rd = normalize(vec3f(p, -1.5));

    // Sphere at origin with radius 1.0.
    let center = vec3f(0.0, 0.0, 0.0);
    let radius = 1.0;

    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;

    if (h < 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let t = -b - sqrt(h);
    if (t <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // White shaded sphere on black background.
    let hitPos = ro + t * rd;
    let n = normalize(hitPos - center);
    let lightDir = normalize(vec3f(0.5, 0.7, 1.0));
    let lit = max(dot(n, lightDir), 0.0);
    let shade = 0.2 + 0.8 * lit;

    return vec4f(vec3f(shade), 1.0);
}
