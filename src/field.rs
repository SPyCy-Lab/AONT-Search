use ark_ff::{Field, PrimeField};

use crate::types::{Symbol, Tuple};

pub type FieldMatrix = Vec<Vec<Symbol>>;

/// Minimal field interface, used in checking non-linearity.
pub trait FiniteFieldOps: Sync {
    fn order(&self) -> usize;
    fn add(&self, left: Symbol, right: Symbol) -> Symbol;
    fn sub(&self, left: Symbol, right: Symbol) -> Symbol;
    fn mul(&self, left: Symbol, right: Symbol) -> Symbol;
    fn inv(&self, value: Symbol) -> Symbol;

    /// Convert the integer n into the corresponding
    /// element n * 1 of the prime subfield.
    fn from_prime_integer(
        &self,
        n: usize,
    ) -> Symbol {
        let mut result = 0;

        for _ in 0..n {
            result =
                self.add(
                    result,
                    1,
                );
        }

        result
    }
    
    fn add_tuples(&self, left: &[Symbol], right: &[Symbol]) -> Tuple {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(&x, &y)| self.add(x, y))
            .collect()
    }

    fn sub_tuples(&self, left: &[Symbol], right: &[Symbol]) -> Tuple {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(&x, &y)| self.sub(x, y))
            .collect()
    }

    fn scalar_mul_tuple(&self, scalar: Symbol, tuple: &[Symbol]) -> Tuple {
        tuple.iter().map(|&x| self.mul(scalar, x)).collect()
    }
}

/// Arkworks-backed adapter for a small prime field.
///
/// The symbol representation matches your old Zech/power convention:
///   0     = field zero
///   k + 1 = generator^k
///
/// Pass a primitive multiplicative generator to `new`.
pub struct ArkPrimeField<F: PrimeField> {
    elements: Vec<F>,
    add: Vec<Vec<Symbol>>,
    sub: Vec<Vec<Symbol>>,
    mul: Vec<Vec<Symbol>>,
}

impl<F: PrimeField> ArkPrimeField<F> {
    pub fn new(generator: F) -> Self {
        assert!(
            F::MODULUS_BIT_SIZE <= usize::BITS,
            "this symbol adapter is intended for small fields"
        );

        let q = F::MODULUS.as_ref()[0] as usize;
        assert!(q >= 2);

        let mut elements = Vec::with_capacity(q);
        elements.push(F::ZERO);

        let mut power = F::ONE;
        for _ in 0..(q - 1) {
            elements.push(power);
            power *= generator;
        }

        // A primitive generator must enumerate every nonzero element exactly once.
        for i in 1..q {
            for j in (i + 1)..q {
                assert!(elements[i] != elements[j], "generator is not primitive");
            }
        }

        let mut add = vec![vec![0; q]; q];
        let mut sub = vec![vec![0; q]; q];
        let mut mul = vec![vec![0; q]; q];

        for a in 0..q {
            for b in 0..q {
                add[a][b] = Self::symbol_of(&elements, elements[a] + elements[b]);
                sub[a][b] = Self::symbol_of(&elements, elements[a] - elements[b]);
                mul[a][b] = Self::symbol_of(&elements, elements[a] * elements[b]);
            }
        }

        Self {
            elements,
            add,
            sub,
            mul,
        }
    }

    fn symbol_of(elements: &[F], value: F) -> Symbol {
        elements
            .iter()
            .position(|candidate| *candidate == value)
            .expect("field element missing from symbol table")
    }
}

impl<F: PrimeField> FiniteFieldOps for ArkPrimeField<F> {
    fn order(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    fn add(&self, left: Symbol, right: Symbol) -> Symbol {
        self.add[left][right]
    }

    #[inline]
    fn sub(&self, left: Symbol, right: Symbol) -> Symbol {
        self.sub[left][right]
    }

    #[inline]
    fn mul(&self, left: Symbol, right: Symbol) -> Symbol {
        self.mul[left][right]
    }

    fn inv(&self, value: Symbol) -> Symbol {
        assert!(value != 0, "cannot invert zero");
    
        let inverse = self.elements[value]
            .inverse()
            .expect("nonzero field element should be invertible");
    
        Self::symbol_of(&self.elements, inverse)
    }
}

/// GF(4) compatibility backend using x^2 + x + 1 over GF(2).
///
/// Arkworks 0.6 provides prime fields plus quadratic/cubic extension models,
/// but its standard prime-field backend is Montgomery-based and its standard
/// quadratic extension model is not a direct fit for GF(2^2). This tiny table
/// backend keeps your important q=4 searches working while all odd-prime
/// fields use Arkworks.
///
/// Symbols preserve the old convention:
///   0 = 0, 1 = 1, 2 = alpha, 3 = alpha^2 = alpha + 1.
pub struct Gf4 {
    add: [[Symbol; 4]; 4],
    sub: [[Symbol; 4]; 4],
    mul: [[Symbol; 4]; 4],
}

impl Default for Gf4 {
    fn default() -> Self {
        let mut add = [[0; 4]; 4];
        let mut mul = [[0; 4]; 4];

        // raw symbol -> polynomial bits: 0, 1, x, x+1
        let bits = [0u8, 1u8, 2u8, 3u8];

        for a in 0..4 {
            for b in 0..4 {
                add[a][b] = (bits[a] ^ bits[b]) as usize;
                mul[a][b] = gf4_mul_bits(bits[a], bits[b]) as usize;
            }
        }

        // Characteristic two: subtraction equals addition.
        let sub = add;
        Self { add, sub, mul }
    }
}

fn gf4_mul_bits(a: u8, b: u8) -> u8 {
    let a0 = a & 1;
    let a1 = (a >> 1) & 1;
    let b0 = b & 1;
    let b1 = (b >> 1) & 1;

    // (a0 + a1*x)(b0 + b1*x), then x^2 = x + 1
    // modulo x^2 + x + 1 in characteristic two.
    let c0 = (a0 & b0) ^ (a1 & b1);
    let c1 = (a0 & b1) ^ (a1 & b0) ^ (a1 & b1);
    c0 | (c1 << 1)
}

impl FiniteFieldOps for Gf4 {
    fn order(&self) -> usize {
        4
    }

    #[inline]
    fn add(&self, left: Symbol, right: Symbol) -> Symbol {
        self.add[left][right]
    }

    #[inline]
    fn sub(&self, left: Symbol, right: Symbol) -> Symbol {
        self.sub[left][right]
    }

    #[inline]
    fn mul(&self, left: Symbol, right: Symbol) -> Symbol {
        self.mul[left][right]
    }

    fn inv(&self, value: Symbol) -> Symbol {
        assert!(value != 0, "cannot invert zero");
    
        (1..4)
            .find(|&candidate| self.mul[value][candidate] == 1)
            .expect("nonzero field element should be invertible")
    }
}

/// GF(8) backend using x^3 + x + 1 over GF(2).
///
/// Elements are represented by polynomial bits:
///
///   0 = 0
///   1 = 1
///   2 = x
///   3 = x + 1
///   4 = x^2
///   5 = x^2 + 1
///   6 = x^2 + x
///   7 = x^2 + x + 1
///
/// Arithmetic is performed modulo
///
///     x^3 + x + 1.
///
/// Since the field has characteristic two,
///
///     a - b = a + b.
pub struct Gf8 {
    add: [[Symbol; 8]; 8],
    sub: [[Symbol; 8]; 8],
    mul: [[Symbol; 8]; 8],
}

impl Default for Gf8 {
    fn default() -> Self {
        let mut add = [[0; 8]; 8];
        let mut mul = [[0; 8]; 8];

        for a in 0..8 {
            for b in 0..8 {
                // Addition in GF(2^3) is coefficient-wise XOR.
                add[a][b] = a ^ b;

                mul[a][b] =
                    gf8_mul_bits(
                        a as u8,
                        b as u8,
                    ) as usize;
            }
        }

        // Characteristic two:
        //
        //     a - b = a + b.
        let sub = add;

        Self {
            add,
            sub,
            mul,
        }
    }
}

/// Multiply two polynomial-bit representations in
///
///     GF(2)[x] / (x^3 + x + 1).
///
/// Bit representation:
///
///     bit 0 -> coefficient of 1
///     bit 1 -> coefficient of x
///     bit 2 -> coefficient of x^2
fn gf8_mul_bits(
    a: u8,
    b: u8,
) -> u8 {
    let mut result = 0u8;
    let mut multiplicand = a;
    let mut multiplier = b;

    // Polynomial multiplication over GF(2).
    while multiplier != 0 {
        if multiplier & 1 != 0 {
            result ^= multiplicand;
        }

        multiplier >>= 1;
        multiplicand <<= 1;
    }

    // Reduce modulo
    //
    //     x^3 + x + 1.
    //
    // Binary representation:
    //
    //     1 + x + x^3 = 0b1011.
    //
    // result can have degree at most 4 because both
    // inputs have degree at most 2.

    const MODULUS: u8 = 0b1011;

    // Eliminate x^4 if present.
    if result & 0b10000 != 0 {
        result ^= MODULUS << 1;
    }

    // Eliminate x^3 if present.
    if result & 0b01000 != 0 {
        result ^= MODULUS;
    }

    // Only degree < 3 remains.
    result & 0b111
}

impl FiniteFieldOps for Gf8 {
    fn order(&self) -> usize {
        8
    }

    #[inline]
    fn add(
        &self,
        left: Symbol,
        right: Symbol,
    ) -> Symbol {
        self.add[left][right]
    }

    #[inline]
    fn sub(
        &self,
        left: Symbol,
        right: Symbol,
    ) -> Symbol {
        self.sub[left][right]
    }

    #[inline]
    fn mul(
        &self,
        left: Symbol,
        right: Symbol,
    ) -> Symbol {
        self.mul[left][right]
    }

    fn inv(
        &self,
        value: Symbol,
    ) -> Symbol {
        assert!(
            value != 0,
            "cannot invert zero"
        );

        (1..8)
            .find(
                |&candidate| {
                    self.mul[value][candidate] == 1
                }
            )
            .expect(
                "nonzero field element should be invertible"
            )
    }
}

pub fn cauchy_matrix(
    a_values: &[Symbol],
    b_values: &[Symbol],
    field: &dyn FiniteFieldOps,
) -> FieldMatrix {
    let s = a_values.len();

    assert!(s > 0, "Cauchy matrix must have positive size");

    assert_eq!(
        s,
        b_values.len(),
        "a_values and b_values must have the same length"
    );

    // Check that a-values are distinct.
    for i in 0..s {
        for j in (i + 1)..s {
            assert!(
                a_values[i] != a_values[j],
                "Cauchy matrix requires distinct a-values"
            );
        }
    }

    // Check that b-values are distinct.
    for i in 0..s {
        for j in (i + 1)..s {
            assert!(
                b_values[i] != b_values[j],
                "Cauchy matrix requires distinct b-values"
            );
        }
    }

    let mut matrix = vec![vec![0; s]; s];

    for i in 0..s {
        for j in 0..s {
            let diff = field.sub(a_values[i], b_values[j]);

            assert!(
                diff != 0,
                "Cauchy matrix invalid: a[{i}] = b[{j}]"
            );

            matrix[i][j] = field.inv(diff);
        }
    }

    matrix
}

/// Invert a square matrix over a finite field using
/// Gauss-Jordan elimination.
///
/// Returns M^{-1}.
pub fn invert_matrix(
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
) -> FieldMatrix {
    let n = matrix.len();

    assert!(
        n > 0,
        "matrix must not be empty"
    );

    for row in matrix {
        assert_eq!(
            row.len(),
            n,
            "matrix must be square"
        );
    }

    // Augmented matrix:
    //
    //     [ M | I ]
    //
    let mut augmented =
        vec![vec![0; 2 * n]; n];

    for i in 0..n {
        for j in 0..n {
            augmented[i][j] =
                matrix[i][j];
        }

        augmented[i][n + i] = 1;
    }

    // ========================================================
    // Gauss-Jordan elimination
    // ========================================================

    for col in 0..n {
        // ----------------------------------------------------
        // Find nonzero pivot
        // ----------------------------------------------------

        let pivot_row =
            (col..n)
                .find(
                    |&row| {
                        augmented[row][col] != 0
                    }
                )
                .expect(
                    "matrix is singular"
                );

        // Move pivot into position.
        if pivot_row != col {
            augmented.swap(
                pivot_row,
                col,
            );
        }

        // ----------------------------------------------------
        // Normalize pivot to 1
        // ----------------------------------------------------

        let pivot =
            augmented[col][col];

        let pivot_inverse =
            field.inv(pivot);

        for j in 0..(2 * n) {
            augmented[col][j] =
                field.mul(
                    augmented[col][j],
                    pivot_inverse,
                );
        }

        // ----------------------------------------------------
        // Eliminate this column from every other row
        // ----------------------------------------------------

        for row in 0..n {
            if row == col {
                continue;
            }

            let factor =
                augmented[row][col];

            if factor == 0 {
                continue;
            }

            for j in 0..(2 * n) {
                let product =
                    field.mul(
                        factor,
                        augmented[col][j],
                    );

                augmented[row][j] =
                    field.sub(
                        augmented[row][j],
                        product,
                    );
            }
        }
    }

    // ========================================================
    // Extract right half
    // ========================================================

    let mut inverse =
        vec![vec![0; n]; n];

    for i in 0..n {
        for j in 0..n {
            inverse[i][j] =
                augmented[i][n + j];
        }
    }

    inverse
}

/// Construct the s x s Vandermonde matrix
///
///     M[r][c] = alpha^(r*c)
///
/// over the supplied finite field.
///
/// For the (2,s,q)-AONT construction from Theorem 2.5.1,
/// alpha must be a primitive element of GF(q), q - 1 must
/// be prime, and s <= q - 1.
///
/// Rows and columns are indexed from 0 to s - 1.
pub fn vandermonde_matrix(
    s: usize,
    alpha: Symbol,
    field: &dyn FiniteFieldOps,
) -> FieldMatrix {
    assert!(
        s > 0,
        "Vandermonde matrix must have positive size"
    );

    let q = field.order();

    assert!(
        alpha > 0 && alpha < q,
        "alpha must be a nonzero field element"
    );

    assert!(
        s <= q - 1,
        "this Vandermonde AONT construction requires s <= q - 1"
    );

    // Verify that alpha is primitive:
    //
    //     1, alpha, alpha^2, ..., alpha^(q-2)
    //
    // must contain all q - 1 nonzero field elements.
    let mut seen =
        vec![false; q];

    let mut power = 1;

    for _ in 0..(q - 1) {
        assert!(
            power != 0,
            "powers of a nonzero field element cannot be zero"
        );

        assert!(
            !seen[power],
            "alpha is not primitive"
        );

        seen[power] = true;

        power =
            field.mul(
                power,
                alpha,
            );
    }

    assert_eq!(
        power,
        1,
        "alpha does not have multiplicative order q - 1"
    );

    // Precompute
    //
    // alpha^0, alpha^1, ..., alpha^(q-2).
    //
    // Exponents can then be reduced modulo q - 1.
    let mut powers =
        vec![0; q - 1];

    powers[0] = 1;

    for exponent in 1..(q - 1) {
        powers[exponent] =
            field.mul(
                powers[exponent - 1],
                alpha,
            );
    }

    // Build
    //
    //     M[r][c] = alpha^(r*c).
    let mut matrix =
        vec![
            vec![0; s];
            s
        ];

    for r in 0..s {
        for c in 0..s {
            let exponent =
                (r * c) % (q - 1);

            matrix[r][c] =
                powers[exponent];
        }
    }

    matrix
}

/// Construct the linear (2,s,q)-AONT obtained from the
/// Vandermonde construction.
///
/// First construct
///
///     V[r][c] = alpha^(r*c)
///
/// and then return
///
///     M = V^{-1}.
///
/// This follows the convention used in the PDF, where the
/// displayed Vandermonde matrix is the inverse of the AONT
/// transformation matrix.
pub fn vandermonde_2_aont_matrix(
    s: usize,
    alpha: Symbol,
    field: &dyn FiniteFieldOps,
) -> FieldMatrix {
    let q =
        field.order();

    assert!(
        is_prime_usize(q - 1),
        "Vandermonde (2,s,q)-AONT construction requires q - 1 to be prime"
    );

    let vandermonde =
        vandermonde_matrix(
            s,
            alpha,
            field,
        );

    invert_matrix(
        &vandermonde,
        field,
    )
}

fn is_prime_usize(
    n: usize,
) -> bool {
    if n < 2 {
        return false;
    }

    if n == 2 {
        return true;
    }

    if n % 2 == 0 {
        return false;
    }

    let mut divisor = 3;

    while divisor <= n / divisor {
        if n % divisor == 0 {
            return false;
        }

        divisor += 2;
    }

    true
}

pub fn matrix_vector_mul(
    vector: &[Symbol],
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
) -> Tuple {
    let rows = matrix.len();

    assert!(rows > 0, "matrix must not be empty");

    let cols = matrix[0].len();

    assert_eq!(
        vector.len(),
        rows,
        "vector length must equal the number of matrix rows"
    );

    for row in matrix {
        assert_eq!(
            row.len(),
            cols,
            "matrix rows must all have the same length"
        );
    }

    let mut result = vec![0; cols];

    // Row-vector convention:
    //
    //     result = vector * matrix
    //
    //     result[j] = sum_i vector[i] * matrix[i][j]
    //
    for j in 0..cols {
        let mut sum = 0;

        for i in 0..rows {
            let product = field.mul(vector[i], matrix[i][j]);
            sum = field.add(sum, product);
        }

        result[j] = sum;
    }

    result
}

pub fn construction_2_5(
    field: &dyn FiniteFieldOps,
) -> FieldMatrix {
    let p = field.order();

    assert!(
        is_prime_usize(p),
        "Construction 2.5 requires a prime field F_p"
    );

    let mut matrix =
        vec![vec![0; p]; p];

    for s in 0..p {
        for t in 0..p {
            if s == t {
                // A(s,s) = 0
                matrix[s][t] = 0;
            } else if t == 0 {
                // A(s,0) = 1
                matrix[s][t] = 1;
            } else {
                // Compute (s - t) in F_p.
                let s_element =
                    field.from_prime_integer(s);

                let t_element =
                    field.from_prime_integer(t);

                let difference =
                    field.sub(
                        s_element,
                        t_element,
                    );

                // A(s,t) = (s-t)^(-1)
                matrix[s][t] =
                    field.inv(difference);
            }
        }
    }

    matrix
}

pub fn print_matrix_as_f7(
    matrix: &[Vec<usize>],
    field: &dyn FiniteFieldOps,
) {
    for row in matrix {
        let normal_row: Vec<usize> =
            row.iter()
                .map(|&symbol| {
                    (0..7)
                        .find(|&value| {
                            field.from_prime_integer(value) == symbol
                        })
                        .unwrap()
                })
                .collect();

        println!("{:?}", normal_row);
    }
}
// #[cfg(test)]
// mod construction_2_5_tests {
//     use super::*;

//     /// Tiny F_5 implementation used only for testing.
//     ///
//     /// Here symbols use the natural representation:
//     ///
//     ///     0, 1, 2, 3, 4
//     ///
//     /// correspond directly to the elements of F_5.
//     struct TestF5;

//     impl FiniteFieldOps for TestF5 {
//         fn order(&self) -> usize {
//             5
//         }

//         fn add(
//             &self,
//             left: Symbol,
//             right: Symbol,
//         ) -> Symbol {
//             (left + right) % 5
//         }

//         fn sub(
//             &self,
//             left: Symbol,
//             right: Symbol,
//         ) -> Symbol {
//             (left + 5 - right) % 5
//         }

//         fn mul(
//             &self,
//             left: Symbol,
//             right: Symbol,
//         ) -> Symbol {
//             (left * right) % 5
//         }

//         fn inv(
//             &self,
//             value: Symbol,
//         ) -> Symbol {
//             assert_ne!(
//                 value,
//                 0,
//                 "cannot invert zero"
//             );

//             match value {
//                 1 => 1,
//                 2 => 3,
//                 3 => 2,
//                 4 => 4,

//                 _ => panic!(
//                     "invalid F5 element: {}",
//                     value
//                 ),
//             }
//         }
//     }

//     // ========================================================
//     // Test exact matrix from Example 2.6
//     // ========================================================

//     #[test]
//     fn construction_2_5_p5_matches_paper() {
//         let field = TestF5;

//         let matrix =
//             construction_2_5(
//                 &field,
//             );

//         let expected = vec![
//             vec![0, 4, 2, 3, 1],
//             vec![1, 0, 4, 2, 3],
//             vec![1, 1, 0, 4, 2],
//             vec![1, 3, 1, 0, 4],
//             vec![1, 2, 3, 1, 0],
//         ];

//         assert_eq!(
//             matrix,
//             expected
//         );
//     }

//     // ========================================================
//     // Verify every 2x2 minor is invertible
//     // ========================================================

//     #[test]
//     fn construction_2_5_all_2x2_minors_nonzero() {
//         let field = TestF5;

//         let matrix =
//             construction_2_5(
//                 &field,
//             );

//         let n = matrix.len();

//         for r1 in 0..n {
//             for r2 in (r1 + 1)..n {
//                 for c1 in 0..n {
//                     for c2 in (c1 + 1)..n {
//                         // determinant:
//                         //
//                         //     a d - b c

//                         let ad =
//                             field.mul(
//                                 matrix[r1][c1],
//                                 matrix[r2][c2],
//                             );

//                         let bc =
//                             field.mul(
//                                 matrix[r1][c2],
//                                 matrix[r2][c1],
//                             );

//                         let determinant =
//                             field.sub(
//                                 ad,
//                                 bc,
//                             );

//                         assert_ne!(
//                             determinant,
//                             0,
//                             "singular 2x2 minor: \
//                              rows ({},{}) \
//                              columns ({},{})",
//                             r1,
//                             r2,
//                             c1,
//                             c2
//                         );
//                     }
//                 }
//             }
//         }
//     }

//     // ========================================================
//     // Verify the whole matrix is invertible
//     // ========================================================

//     #[test]
//     fn construction_2_5_whole_matrix_invertible() {
//         let field = TestF5;

//         let matrix =
//             construction_2_5(
//                 &field,
//             );

//         let inverse =
//             invert_matrix(
//                 &matrix,
//                 &field,
//             );

//         let n = matrix.len();

//         // Verify:
//         //
//         //     M M^-1 = I

//         for i in 0..n {
//             for j in 0..n {
//                 let mut sum = 0;

//                 for k in 0..n {
//                     let product =
//                         field.mul(
//                             matrix[i][k],
//                             inverse[k][j],
//                         );

//                     sum =
//                         field.add(
//                             sum,
//                             product,
//                         );
//                 }

//                 let expected =
//                     if i == j {
//                         1
//                     } else {
//                         0
//                     };

//                 assert_eq!(
//                     sum,
//                     expected,
//                     "M * M^-1 failed at ({},{})",
//                     i,
//                     j
//                 );
//             }
//         }
//     }
// }

// #[cfg(test)]
// mod gf8_tests {
//     use super::*;

//     #[test]
//     fn gf8_basic_arithmetic() {
//         let field = Gf8::default();

//         assert_eq!(field.order(), 8);

//         // x + x = 0
//         assert_eq!(
//             field.add(2, 2),
//             0
//         );

//         // x^2 + x = x^2 + x
//         assert_eq!(
//             field.add(4, 2),
//             6
//         );

//         // x^2 + (x + 1)
//         // = x^2 + x + 1
//         assert_eq!(
//             field.add(4, 3),
//             7
//         );

//         // x * x = x^2
//         assert_eq!(
//             field.mul(2, 2),
//             4
//         );

//         // x^2 * x = x^3 = x + 1
//         assert_eq!(
//             field.mul(4, 2),
//             3
//         );
//     }

//     #[test]
//     fn gf8_every_nonzero_element_has_inverse() {
//         let field = Gf8::default();

//         for a in 1..8 {
//             let inverse =
//                 field.inv(a);

//             assert_eq!(
//                 field.mul(a, inverse),
//                 1,
//                 "bad inverse for element {}",
//                 a
//             );
//         }
//     }

//     #[test]
//     fn gf8_field_axioms() {
//         let field = Gf8::default();

//         for a in 0..8 {
//             for b in 0..8 {
//                 // Closure
//                 assert!(
//                     field.add(a, b) < 8
//                 );

//                 assert!(
//                     field.mul(a, b) < 8
//                 );

//                 // Commutativity
//                 assert_eq!(
//                     field.add(a, b),
//                     field.add(b, a)
//                 );

//                 assert_eq!(
//                     field.mul(a, b),
//                     field.mul(b, a)
//                 );

//                 for c in 0..8 {
//                     // Associativity of addition
//                     assert_eq!(
//                         field.add(
//                             field.add(a, b),
//                             c
//                         ),
//                         field.add(
//                             a,
//                             field.add(b, c)
//                         )
//                     );

//                     // Associativity of multiplication
//                     assert_eq!(
//                         field.mul(
//                             field.mul(a, b),
//                             c
//                         ),
//                         field.mul(
//                             a,
//                             field.mul(b, c)
//                         )
//                     );

//                     // Distributivity
//                     assert_eq!(
//                         field.mul(
//                             a,
//                             field.add(b, c)
//                         ),
//                         field.add(
//                             field.mul(a, b),
//                             field.mul(a, c)
//                         )
//                     );
//                 }
//             }
//         }
//     }
// }

// #[test]
// fn gf8_vandermonde_aont_inverse_test() {
//     let field =
//         Gf8::default();

//     let alpha = 2;
//     let s = 7;

//     let v =
//         vandermonde_matrix(
//             s,
//             alpha,
//             &field,
//         );

//     let m =
//         vandermonde_2_aont_matrix(
//             s,
//             alpha,
//             &field,
//         );

//     // Verify:
//     //
//     //     M * V = I
//     //
//     for i in 0..s {
//         for j in 0..s {
//             let mut sum = 0;

//             for k in 0..s {
//                 sum =
//                     field.add(
//                         sum,
//                         field.mul(
//                             m[i][k],
//                             v[k][j],
//                         ),
//                     );
//             }

//             let expected =
//                 if i == j {
//                     1
//                 } else {
//                     0
//                 };

//             assert_eq!(
//                 sum,
//                 expected,
//                 "inverse verification failed at ({}, {})",
//                 i,
//                 j
//             );
//         }
//     }
// }
