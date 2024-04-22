pub mod helpers;

use helpers::pre_recommendation::{extract_data, get_embeddings};
use helpers::post_recommendation::get_recommendations;
use helpers::types::Data;

use std::collections::HashMap;
use candle::Tensor;
use text_io::read;

pub fn create_model(file_path: String) -> Result<HashMap<Data, Option<Tensor>>, ()> {
    let mut nodes_wrapped: Result<HashMap<Data, Option<Tensor>>, ()> = extract_data(&file_path);
    while nodes_wrapped.is_err() {
        println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
        let path: String = read!();
        nodes_wrapped = extract_data(&path);
    }
    let mut nodes: HashMap<Data, Option<Tensor>> = nodes_wrapped.unwrap();
    let node_embeddings = get_embeddings(&mut nodes).unwrap();
    Ok(node_embeddings)
}

pub fn pass_description(node_embeddings: &HashMap<Data, Option<Tensor>>, description_input: String, tags_input: String, num_recommendations: usize) -> Vec<String> {
    let recommendations = get_recommendations(node_embeddings, &description_input, &tags_input, num_recommendations).unwrap();
    recommendations
}

