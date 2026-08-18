/// Basic I/O functions to print and parse the AONTs found.
/// 
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Write};

use crate::config::SearchConfig;
use crate::types::{Mapping, Tuple};

pub struct OutputFiles {
    linear: BufWriter<File>,
    nonlinear: BufWriter<File>,
}

impl OutputFiles {
    pub fn create(config: &SearchConfig) -> std::io::Result<Self> {
        let linear = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&config.linear_output)?;

        let nonlinear = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&config.nonlinear_output)?;

        Ok(Self {
            linear: BufWriter::new(linear),
            nonlinear: BufWriter::new(nonlinear),
        })
    }

    pub fn write_mapping(
        &mut self,
        mapping: &Mapping,
        linear: bool,
        s: usize,
        q: usize,
    ) -> std::io::Result<()> {
        let writer = if linear {
            &mut self.linear
        } else {
            &mut self.nonlinear
        };

        writeln!(writer, "BEGIN_AONT")?;
        writeln!(writer, "TYPE: {}", if linear { "LINEAR" } else { "NONLINEAR" })?;
        writeln!(writer, "SIZE: {s}  FIELD_SIZE: {q}")?;
        writeln!(writer, "REPRESENTATION: 0=zero, k+1=alpha^k")?;
        writeln!(writer, "MAPPING:")?;

        for (source, target) in mapping {
            writeln!(writer, "{:?}->{:?}", source, target)?;
        }

        writeln!(writer, "END_AONT\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.linear.flush()?;
        self.nonlinear.flush()?;
        Ok(())
    }
}

fn parse_usize_list(text: &str, line_number: usize) -> std::io::Result<Tuple> {
    let trimmed = text.trim();

    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Line {line_number}: expected a vector in brackets, found `{trimmed}`"),
        ));
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    inner
        .split(',')
        .map(|part| {
            let symbol_text = part.trim();
            symbol_text.parse::<usize>().map_err(|error| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("Line {line_number}: invalid field symbol `{symbol_text}`: {error}"),
                )
            })
        })
        .collect()
}

fn parse_mapping_line(line: &str, line_number: usize) -> std::io::Result<(Tuple, Tuple)> {
    let Some((source_text, target_text)) = line.split_once("->") else {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Line {line_number}: expected `[source]->[target]`"),
        ));
    };

    Ok((
        parse_usize_list(source_text, line_number)?,
        parse_usize_list(target_text, line_number)?,
    ))
}

pub fn parse_aont_mappings(path: &str) -> std::io::Result<Vec<Mapping>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mappings = Vec::new();
    let mut current_mapping: Option<Mapping> = None;
    let mut reading_mapping = false;

    for (zero_based_line, line_result) in reader.lines().enumerate() {
        let line_number = zero_based_line + 1;
        let line = line_result?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "BEGIN_AONT" => {
                if current_mapping.is_some() {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Line {line_number}: nested BEGIN_AONT block"),
                    ));
                }
                current_mapping = Some(Vec::new());
                reading_mapping = false;
            }
            "MAPPING:" => {
                if current_mapping.is_none() {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Line {line_number}: MAPPING outside an AONT block"),
                    ));
                }
                reading_mapping = true;
            }
            "END_AONT" => {
                let mapping = current_mapping.take().ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Line {line_number}: END_AONT without BEGIN_AONT"),
                    )
                })?;

                if mapping.is_empty() {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Line {line_number}: AONT block contains no mapping rows"),
                    ));
                }

                mappings.push(mapping);
                reading_mapping = false;
            }
            _ if reading_mapping => {
                current_mapping
                    .as_mut()
                    .expect("reading_mapping implies an active block")
                    .push(parse_mapping_line(trimmed, line_number)?);
            }
            _ => {}
        }
    }

    if current_mapping.is_some() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Input file ended before END_AONT",
        ));
    }

    Ok(mappings)
}

pub fn validate_complete_mapping(
    mapping: &Mapping,
    expected_s: usize,
    expected_q: usize,
) -> std::io::Result<HashMap<Tuple, Tuple>> {
    let expected_size = expected_q.pow(expected_s as u32);

    if mapping.len() != expected_size {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "Mapping has {} rows, but GF({expected_q})^{expected_s} requires {expected_size} rows",
                mapping.len()
            ),
        ));
    }

    let mut lookup = HashMap::with_capacity(mapping.len());
    let mut outputs = HashSet::with_capacity(mapping.len());

    for (source, target) in mapping {
        if source.len() != expected_s || target.len() != expected_s {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Mapping row has dimensions {} -> {}, expected {expected_s} -> {expected_s}",
                    source.len(),
                    target.len()
                ),
            ));
        }

        if source
            .iter()
            .chain(target.iter())
            .any(|&symbol| symbol >= expected_q)
        {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Mapping contains a raw symbol outside 0..{expected_q}"),
            ));
        }

        if lookup.insert(source.clone(), target.clone()).is_some() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Duplicate input vector in mapping: {source:?}"),
            ));
        }

        if !outputs.insert(target.clone()) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Duplicate output vector in mapping: {target:?}"),
            ));
        }
    }

    Ok(lookup)
}
