// KAN layer forward pass shader
//
// Computes the projection and accumulates function outputs.
// Note: This is a simplified version that uses projections directly.
// Full implementation would integrate univariate function evaluation.

struct LayerParams {
    input_dim: u32,
    output_dim: u32,
    num_functions: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<uniform> params: LayerParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> projection: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let o = global_id.x;

    if o < params.output_dim {
        var sum: f32 = 0.0;

        // For each univariate function
        for (var f: u32 = 0u; f < params.num_functions; f = f + 1u) {
            var proj: f32 = 0.0;

            // Compute projection: input · projection_column
            for (var i: u32 = 0u; i < params.input_dim; i = i + 1u) {
                let proj_idx = i * params.num_functions + f;
                if proj_idx < arrayLength(&projection) && i < arrayLength(&input) {
                    proj += input[i] * projection[proj_idx];
                }
            }

            // In full implementation, would evaluate univariate function here
            // For now, use projection directly (simplified)
            sum += proj;
        }

        output[o] = sum;
    }
}
