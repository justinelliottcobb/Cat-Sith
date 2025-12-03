// Univariate function evaluation shader
//
// Computes: f(x) = Σ basis[i] * weights[i]

@group(0) @binding(0)
var<storage, read> basis: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: f32;

@compute @workgroup_size(1)
fn main() {
    var result: f32 = 0.0;

    let n = min(arrayLength(&basis), arrayLength(&weights));
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        result += basis[i] * weights[i];
    }

    output = result;
}
