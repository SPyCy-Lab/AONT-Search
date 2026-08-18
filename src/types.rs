use std::collections::{HashMap, HashSet};

pub type Symbol = usize;
pub type Tuple = Vec<Symbol>;
pub type TupleSet = HashSet<Tuple>;
pub type TupleMap = HashMap<Tuple, TupleSet>;
pub type Mapping = Vec<(Tuple, Tuple)>;
