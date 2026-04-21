struct VertexInput {
    @location(0) position: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct TwoDUniform {
    circle: vec4f, // x,y center in NDC, z radius, w edge softness
    render: vec4f, // x time, y aspect ratio, z pulse speed
};

@group(0) @binding(0) var<uniform> u: TwoDUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(input.position, 1.0);
    out.uv = input.position.xy;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let center = u.circle.xy;
    let baseRadius = max(u.circle.z, 0.0001);
    let edge = max(u.circle.w, 0.0001);
    let aspect = max(u.render.y, 0.0001);
    let pulseSpeed = u.render.z;
    let time = u.render.x;

    var p = input.uv - center;
    p.x *= aspect;

    let animatedRadius = baseRadius * (0.9 + 0.1 * sin(time * pulseSpeed));
    let dist = length(p);
    let circleMask = 1.0 - smoothstep(animatedRadius - edge, animatedRadius + edge, dist);

    let bg = vec3f(0.02, 0.02, 0.035);
    let ringColor = vec3f(0.95, 0.35, 0.2);
    let fillColor = vec3f(0.22, 0.7, 1.0);
    let rim = smoothstep(animatedRadius - edge * 2.0, animatedRadius, dist);
    let color = mix(bg, mix(fillColor, ringColor, rim), circleMask);

    return vec4f(color, 1.0);
}
