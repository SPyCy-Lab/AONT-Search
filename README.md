# AONT Search

A Rust framework for enumerating **All-Or-Nothing Transforms (AONTs)** using a Branch-and-Bound (BnB) search. The project also contains tools for analyzing the resulting AONTs, including:

- Linearity testing
- Differential uniformity computation
- Search tree estimation
- Column-wise permutation equivalence testing

The current implementation can fully enumerate small parameters such as 

(2,3,3)-AONT, (1,2,4)-AONT, and (1,2,5)-AONT.

---

# Features

The project currently contains the following modules.

| Module | Purpose |
|---------|---------|
| `bnb.rs` | Branch-and-bound search for AONTs |
| `differential_uniformity.rs` | Computes differential uniformity of mappings |
| `equivalence.rs` | Tests whether nonlinear AONTs are column-wise symbol permutations of linear ones |
| `linearity.rs` | Exact GF(q)-linearity test |
| `time_estimator.rs` | Search complexity estimation and randomized tree sampling |
| `mapping_io.rs` | Reading/writing AONT files |
| `field.rs` | Finite field implementations |
| `hamming.rs` | Hamming distance utilities |
| `types.rs` | Common type definitions |
| `config.rs` | Search configuration |

---

# Supported Fields

## Prime Fields GF(p)

Prime fields are implemented using **Arkworks**.

Example:

```rust
use ark_ff::{Fp64, MontBackend, MontConfig};
use aont_search::field::ArkPrimeField;

#[derive(MontConfig)]
#[modulus = "5"]
#[generator = "2"]
pub struct F5Config;

pub type F5 = Fp64<MontBackend<F5Config, 1>>;

let field =
    ArkPrimeField::<F5>::new(F5::from(2u64));
```

---

## GF(4)

GF(4) uses a small compatibility implementation included in this project.

```rust
use aont_search::field::Gf4;

let field = Gf4::default();
```

---

# Running a Search

Most users only need to modify `main.rs`.

There are four things to choose.

---

## 1. Choose the field

Example:

```rust
let field = ArkPrimeField::<F5>::new(F5::from(2u64));

let q = 5;
```

or

```rust
let field = Gf4::default();

let q = 4;
```

---

## 2. Choose the AONT parameters

```rust
let t = 2;
let s = 3;
```

This searches for a


(t,s,q) - AONT.


---

## 3. Select optional analyses

```rust
let run_search = true;

let run_differential_uniformity = true;

let run_column_equivalence_check = true;
```

Any of these can be turned off.

---

## 4. Configure the search

```rust
let mut config = SearchConfig::new(t, s);

config.fix_zero = true;

config.first_nonlinear_to_keep = 5;

config.progress_report_interval = 1_000_000;

config.estimator_samples = 5_000;
```

### `fix_zero`

If enabled,

```
f(0)=0
```

is enforced during the search.

This removes the output translation symmetry and usually reduces the search space dramatically.

---

### `first_nonlinear_to_keep`

Controls how many nonlinear AONTs are kept in memory after the search.

**This does NOT affect what is written to disk.**

Every nonlinear AONT is still written to `nonlinear.txt`.

---

### `progress_report_interval`

Print a progress report every N visited search nodes.

Typical values:

```
100000
1000000
10000000
```

---

### `estimator_samples`

Number of randomized search paths used by the tree estimator.

Higher values generally give better estimates but take longer before the search starts.

---

# Running

Compile:

```bash
cargo build --release
```

Run:

```bash
cargo run --release
```

---

# Output Files

The search generates

```
linear.txt
```

containing all linear AONTs and

```
nonlinear.txt
```

containing all nonlinear AONTs.

Mappings are stored as

```
BEGIN_AONT

...

MAPPING:

[0,0,0] -> [1,2,0]

...

END_AONT
```

---

# Optional Analysis Tools

## Differential Uniformity

Computes the differential uniformity of every nonlinear AONT.

Enable by setting

```rust
run_differential_uniformity = true;
```

The output is written to

```
nonlinear_differential_uniformity.txt
```

---

## Column-wise Equivalence

Tests the conjecture that every nonlinear AONT is obtainable from a linear AONT by independently permuting the symbols in each input and output coordinate.

Enable by setting

```rust
run_column_equivalence_check = true;
```

The checker reports

- number of nonlinear mappings covered
- number missing
- indices of any missing nonlinear mappings

---

# Citation

If you use this code in academic work, please cite the associated paper once available.
