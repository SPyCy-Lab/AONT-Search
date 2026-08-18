use std::collections::HashMap;

use crate::field::FiniteFieldOps;
use crate::types::{Mapping, Tuple};

/// Exact GF(q)-linearity test.
///
/// Checks f(0)=0, additivity, and scalar compatibility. The scalar check is
/// essential for extension fields because additivity alone only gives
/// linearity over the prime subfield.
pub fn is_linear_aont(
    mapping: &Mapping,
    field: &dyn FiniteFieldOps,
    s: usize,
    print_reason: bool,
) -> bool {
    let lookup: HashMap<&Tuple, &Tuple> = mapping.iter().map(|(x, y)| (x, y)).collect();
    let zero = vec![0; s];

    let Some(f_zero) = lookup.get(&zero) else {
        panic!("mapping does not contain the zero input");
    };

    if **f_zero != zero {
        if print_reason {
            println!("Not linear because f(0) != 0: f(0) = {:?}", f_zero);
        }
        return false;
    }

    for (x, fx) in mapping {
        for (y, fy) in mapping {
            let x_plus_y = field.add_tuples(x, y);
            let fx_plus_fy = field.add_tuples(fx, fy);
            let f_x_plus_y = lookup
                .get(&x_plus_y)
                .expect("complete mapping is missing x+y");

            if **f_x_plus_y != fx_plus_fy {
                if print_reason {
                    println!("Nonlinear additive witness:");
                    println!("x             = {:?}", x);
                    println!("y             = {:?}", y);
                    println!("x + y         = {:?}", x_plus_y);
                    println!("f(x) + f(y)   = {:?}", fx_plus_fy);
                    println!("f(x + y)      = {:?}", f_x_plus_y);
                }
                return false;
            }
        }
    }

    for scalar in 0..field.order() {
        for (x, fx) in mapping {
            let scalar_x = field.scalar_mul_tuple(scalar, x);
            let scalar_fx = field.scalar_mul_tuple(scalar, fx);
            let f_scalar_x = lookup
                .get(&scalar_x)
                .expect("complete mapping is missing scalar*x");

            if **f_scalar_x != scalar_fx {
                if print_reason {
                    println!("Nonlinear scalar witness:");
                    println!("lambda          = {}", scalar);
                    println!("x               = {:?}", x);
                    println!("lambda*x        = {:?}", scalar_x);
                    println!("lambda*f(x)     = {:?}", scalar_fx);
                    println!("f(lambda*x)     = {:?}", f_scalar_x);
                }
                return false;
            }
        }
    }

    true
}
