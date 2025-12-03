// B-spline basis function evaluation shader
//
// Uses Cox-de Boor recursion for numerically stable evaluation.

@group(0) @binding(0)
var<storage, read> knots: array<f32>;

@group(0) @binding(1)
var<storage, read_write> basis_output: array<f32>;

struct Params {
    x: f32,
    degree: u32,
}

@group(0) @binding(2)
var<uniform> params: Params;

// Evaluate a single B-spline basis function using Cox-de Boor
fn evaluate_basis(x: f32, degree: u32, target_idx: u32) -> f32 {
    let num_basis = arrayLength(&basis_output);

    if target_idx >= num_basis {
        return 0.0;
    }

    // Find knot span containing x
    var span: u32 = degree;
    for (var i: u32 = degree; i < arrayLength(&knots) - 1u; i = i + 1u) {
        if x < knots[i + 1u] {
            span = i;
            break;
        }
    }

    // Handle right boundary
    if x >= knots[arrayLength(&knots) - 1u] {
        span = arrayLength(&knots) - degree - 2u;
    }

    // Check if this basis function is non-zero for this span
    let first_nonzero = span - degree;
    let last_nonzero = span;

    if target_idx < first_nonzero || target_idx > last_nonzero {
        return 0.0;
    }

    let local_idx = target_idx - first_nonzero;

    // Cox-de Boor recursion (unrolled for GPU)
    var N: array<f32, 10>; // Support up to degree 9

    for (var i: u32 = 0u; i <= degree; i = i + 1u) {
        N[i] = 0.0;
    }
    N[0] = 1.0;

    for (var j: u32 = 1u; j <= degree; j = j + 1u) {
        var saved: f32 = 0.0;
        for (var r: u32 = 0u; r < j; r = r + 1u) {
            let left_idx = span + 1u + r - j;
            let right_idx = span + 1u + r;

            var alpha: f32 = 0.0;
            if left_idx < arrayLength(&knots) && right_idx < arrayLength(&knots) {
                let left = knots[left_idx];
                let right = knots[right_idx];
                if right != left {
                    alpha = (x - left) / (right - left);
                }
            }

            let temp = N[r];
            N[r] = saved + (1.0 - alpha) * temp;
            saved = alpha * temp;
        }
        N[j] = saved;
    }

    if local_idx <= degree {
        return N[local_idx];
    }

    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx < arrayLength(&basis_output) {
        basis_output[idx] = evaluate_basis(params.x, params.degree, idx);
    }
}
