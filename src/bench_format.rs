//! Format selection for the benchmark catalog CLI.

use crate::bench::{csv_header, find, list_csv, list_json, list_report};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogFormat {
    Text,
    Json,
    Csv,
}

impl CatalogFormat {
    pub fn from_flags(json: bool, csv: bool) -> Result<Self, String> {
        match (json, csv) {
            (true, true) => Err("use only one of --json or --csv".into()),
            (true, false) => Ok(Self::Json),
            (false, true) => Ok(Self::Csv),
            (false, false) => Ok(Self::Text),
        }
    }
}

pub fn render(command: &str, id: Option<&str>, fmt: CatalogFormat) -> Result<String, String> {
    match command {
        "list" => Ok(match fmt {
            CatalogFormat::Text => list_report(),
            CatalogFormat::Json => list_json(),
            CatalogFormat::Csv => list_csv(),
        }),
        "describe" => {
            let id = id.ok_or_else(|| "bench describe requires an id".to_string())?;
            let spec = find(id).ok_or_else(|| format!("unknown bench {id}"))?;
            Ok(match fmt {
                CatalogFormat::Text => spec.describe(),
                CatalogFormat::Json => spec.json(),
                CatalogFormat::Csv => format!("{}{}", csv_header(), spec.csv_row()),
            })
        }
        other => Err(format!("unknown bench subcommand {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_exclusive() {
        assert!(CatalogFormat::from_flags(true, true).is_err());
        assert_eq!(CatalogFormat::from_flags(true, false).unwrap(), CatalogFormat::Json);
        assert_eq!(CatalogFormat::from_flags(false, true).unwrap(), CatalogFormat::Csv);
        assert_eq!(CatalogFormat::from_flags(false, false).unwrap(), CatalogFormat::Text);
    }

    #[test]
    fn describe_json_contains_id() {
        let out = render("describe", Some("engine"), CatalogFormat::Json).unwrap();
        assert!(out.contains("\"id\":\"engine\""));
    }

    #[test]
    fn unknown_id_is_an_error() {
        assert!(render("describe", Some("nope"), CatalogFormat::Text).is_err());
    }
}
