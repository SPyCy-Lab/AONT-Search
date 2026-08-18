use ark_ff::{Field, PrimeField};

use crate::types::{Symbol, Tuple};

/// Minimal field interface, used in checking non-linearity.
pub trait FiniteFieldOps {
    fn order(&self) -> usize;
    fn add(&self, left: Symbol, right: Symbol) -> Symbol;
    fn sub(&self, left: Symbol, right: Symbol) -> Symbol;
    fn mul(&self, left: Symbol, right: Symbol) -> Symbol;

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
}
