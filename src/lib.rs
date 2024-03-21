extern crate serde;
extern crate serde_json;

use serde::Deserialize;
use std::{fs::File, io::Read};


#[derive(Debug, Deserialize, Clone)]
pub struct Data {
    name: String,
    description: String,
    tags: Vec<String>
}

pub fn extract_data(file_name: &String) -> Result<Vec<Data>, ()> {
    let file_result = File::open(file_name);
    let mut vector_of_data: Vec<Data> = Vec::new();
    if file_result.is_err() {
        return Err(());
    }
    let mut file = file_result.unwrap();
    let mut contents = String::new();
    let read = file.read_to_string(&mut contents);
    if read.is_err() {
        return Err(());
    }
    let contents_split: Vec<&str> = contents.split(",").collect();
    let mut contents_split_string: Vec<String> = Vec::new();
    for x in 0..contents_split.len() {
        contents_split_string.push(contents_split[x].to_string());
    }

    for x in 0..contents_split.len() {
        let deserialize = serde_json::from_str::<Data>(&contents_split_string[x]);
        if deserialize.is_err() {
            return Err(());
        }
        vector_of_data.push(deserialize.unwrap().clone());
    }
    
    return Ok(vector_of_data);
}