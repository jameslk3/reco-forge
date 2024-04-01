pub(crate) mod types {
    use serde::Deserialize;
    use std::fmt;

    #[derive(Debug, Deserialize, Clone)]
    pub struct Data {
        name: String,
        summary: String,
        embedding: Option<Vec<f64>>,
        tags: Vec<String>
    }

    impl fmt::Display for Data {
        // Implement the Display trait for Link
        // Should display only the contents of thing
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Name: {}, Summary: {}", self.name, self.summary)
        }
    }
}

pub(crate) mod functions {
    use std::fs::File;
    use super::types::Data;

    /// Receives the path of a JSON file as a &String. The function tries to open
    /// the file. If it doesn't, it will return Err.
    /// After it opens the file, it deserializes the JSON file into a vector of Data objects.
    /// If it successfully does so, it will return Ok. If it doesn't, it will return Err.
    /// 
    /// @param `file_name` - a String containing the file path of the JSON file
    /// 
    /// @return `Ok()` with the vector of data encapsulated in a `Result` enum [OR] `Err()` if the 
    /// file didn't open or didn't deserialize
    pub fn extract_data(file_name: &String) -> Result<Vec<Data>, ()> {

        // Opens file
        let file = File::open(file_name).expect("File didn't open");

        // Deserializes into Data object
        let vector_of_data: Vec<Data> = serde_json::from_reader(file).expect("Deserialization failed");

        return Ok(vector_of_data);
    }

    pub fn fill_embedding(vector: &Vec<Data>) {
        
    }

    
}

pub(crate) mod utils {
    use super::types::Data;

    pub fn print_data(vector: Vec<Data>) {
        for datum in vector {
            println!("{}", datum);
            println!("");
        }
    }
}