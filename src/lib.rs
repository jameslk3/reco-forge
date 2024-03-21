extern crate serde;
extern crate serde_json;

use serde::Deserialize;
use std::{fmt, fs::File};


#[derive(Debug, Deserialize, Clone)]
pub struct Data {
    name: String,
    summary: String,
    tags: Vec<String>
}

impl fmt::Display for Data {
    // Implement the Display trait for Link
    // Should display only the contents of thing
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Name: {}, Summary: {}", self.name, self.summary)
    }
}

pub fn extract_data(file_name: &String) -> Result<Vec<Data>, ()> {
    let file = File::open(file_name).expect("File didn't open");

    let vector_of_data: Vec<Data> = serde_json::from_reader(file).expect("Deserialization failed");

    return Ok(vector_of_data);
}