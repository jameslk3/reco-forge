pub mod helpers;

use helpers::pre_recommendation::{extract_data, get_embeddings, find_embedding};
use helpers::recommendation::{get_recommendations, create_input_embedding};
use helpers::types::Data;

use std::collections::HashMap;
use candle::Tensor;

pub fn create_model(file_path: &String) -> Result<HashMap<Data, Option<Tensor>>, String> {
    let nodes_wrapped: Result<HashMap<Data, Option<Tensor>>, ()> = extract_data(&file_path);
    if nodes_wrapped.is_err() {
        return Err(("File path is not valid or file cannot be deserialized, please input the correct file path and try again:").to_string());
    }
    let mut nodes: HashMap<Data, Option<Tensor>> = nodes_wrapped.unwrap();
    let node_embeddings = get_embeddings(&mut nodes).unwrap();
    Ok(node_embeddings)
}

pub fn pass_description(node_embeddings: &HashMap<Data, Option<Tensor>>, description_input: String, tags_input: String, num_recommendations: usize) -> Result<Vec<String>, ()> {

    // When we are given a description, we need to create an embedding for it and then find recommendations based on that
    let input_embedding = create_input_embedding(&description_input).unwrap().unwrap();
    let recommendations = get_recommendations(node_embeddings, &input_embedding, &tags_input, num_recommendations).unwrap();
    Ok(recommendations)
}

pub fn pass_item(node_embeddings: &HashMap<Data, Option<Tensor>>, item: String, tags_input: String, num_recommendations: usize) -> Result<Vec<String>, ()> {

    // When we want to find items similar to a specific item, we need to make sure that the item is in the embeddings and then retrieve the embedding
    let input_embedding = {
        if let Some(embedding) = find_embedding(node_embeddings, &item) {
            embedding
        } else {
            panic!("Item not found in embeddings");
        }
    };
    let recommendations = get_recommendations(node_embeddings, &input_embedding, &tags_input, num_recommendations).unwrap();
    Ok(recommendations)
}