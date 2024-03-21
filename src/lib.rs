use std::{fs::File, io::Read};

use json;

struct Data {
    name: String,
    description: String,
    tags: Vec<String>
}

pub fn extract_data(file_name: &String) -> Result<(Data), ()> {
    let file_result = File::open(file_name);
    if file_result.is_err() {
        return Err(());
    }
    let mut file = file_result.unwrap();
    let mut contents = String::new();
    let read = file.read_to_string(&mut contents);
    if read.is_err() {
        return Err(());
    }
    let parsed = json::parse(&contents);
    return Err(());
}