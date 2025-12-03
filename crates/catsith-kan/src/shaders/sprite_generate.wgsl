// Sprite generation shader
//
// Generates all pixels in parallel from a latent code.
// Each thread computes one pixel's RGBA value.
//
// TODO: This is a placeholder - full implementation needs to
// integrate KAN forward pass into GPU shader.

struct Params {
    width: u32,
    height: u32,
    latent_dim: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> latent: array<f32>;

@group(0) @binding(2)
var<storage, read_write> pixels: array<u32>; // Packed RGBA

// Sigmoid activation
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let pixel_idx = y * params.width + x;

    // Normalize coordinates to [-1, 1]
    let nx = f32(x) / f32(params.width - 1u) * 2.0 - 1.0;
    let ny = f32(y) / f32(params.height - 1u) * 2.0 - 1.0;

    // Simple placeholder: generate pattern based on latent + coords
    // Real implementation would run KAN forward pass here
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    var a: f32 = 1.0;

    // Mix latent code with position for basic pattern
    for (var i: u32 = 0u; i < params.latent_dim; i = i + 1u) {
        let phase = f32(i) * 0.5;
        r += latent[i] * sin(nx * 3.14159 + phase);
        g += latent[i] * sin(ny * 3.14159 + phase);
        b += latent[i] * sin((nx + ny) * 3.14159 + phase);
    }

    r = sigmoid(r);
    g = sigmoid(g);
    b = sigmoid(b);

    // Pack RGBA into u32
    let ri = u32(r * 255.0);
    let gi = u32(g * 255.0);
    let bi = u32(b * 255.0);
    let ai = u32(a * 255.0);

    pixels[pixel_idx] = ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}
