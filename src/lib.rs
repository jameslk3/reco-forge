//! This crate provides an interface for users to turn any dataset with titles and descriptions into a recommendation system. It uses the BERT model to create embeddings for each item in the dataset and then finds recommendations based on the user's input.
//! 
//! Example usage:
//! ```no_run
//! use reco_forge::{create_model, pass_description, Data, Tensor, HashMap}; // Can also use pass_item
//! 
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!    let mut path = String::new();
//!    println!("Please enter the file path:");
//!
//!    let model_wrapped: Result<HashMap<Data, Option<Tensor>>, ()>;
//!
//!    loop {
//!        std::io::stdin().read_line(&mut path).expect("Failed to read line");
//!        path = path.trim().to_string();
//!
//!        if let Ok(model) = create_model(&path) {
//!            model_wrapped = Ok(model);
//!            break;
//!        } else {
//!            println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
//!            path.clear();
//!        }
//!    }
//! 
//!   let model: HashMap<Data, Option<Tensor>> = model_wrapped.unwrap();
//!
//!    println!("Input tags that you would like to use to filter, else enter NONE");
//!    let mut tags_input: String = String::new();
//!    std::io::stdin().read_line(&mut tags_input).expect("Failed to read line");
//!    tags_input = tags_input.trim().to_string();
//!
//!    println!("Describe what you want to be recommended:");
//!    let mut query: String = String::new();
//!    std::io::stdin().read_line(&mut query).expect("Failed to read line");
//!    query = query.trim().to_string();
//!    println!();
//!
//!    let recommendations = pass_description(&model, query, tags_input, 10);
//!    match recommendations {
//!        Ok(recommendations) => {
//!            println!("Recommendations:");
//!            for recommendation in recommendations {
//!                println!("{}", recommendation);
//!            }
//!        },
//!        Err(_) => println!("No recommendations found"),
//!    }
//!    Ok(())
//!}
//! 
//! Examples are also provided in the examples folder of the git repository at https://github.com/jameslk3/reco-forge.
//! 
//! ```
//! Required JSON file format:
//! ```json
//! [
//! 	{
//! 		"id": int,
//! 		"name": "string",
//! 		"summary": "string",
//! 		"tags": ["string1", "string2"]
//! 	},
//! 	{
//! 		...
//! 	}
//! ]
//! ```


pub(crate) mod helpers;

extern crate candle;

pub use candle::Tensor;
pub use helpers::types::Data;
pub use std::collections::HashMap;

use helpers::pre_recommendation::{extract_data, insert_embeddings, find_embedding};
use helpers::recommendation::{get_recommendations, create_input_embedding};

/// # create_model
/// This function creates the model from the file path given by the user
/// 
/// # Arguments
/// ```no_run
///     * file_path: &String - The file path to the model
/// ```
/// 
/// # Returns
/// ```no_run
///    * Result<HashMap<Data, Option<Tensor>>, String> - The model if it was created successfully, otherwise a wrapped error message
/// ```
/// 
/// # Example
/// ```no_run
/// 	let file_path = "path/to/model".to_string();
/// 	let model = create_model(&file_path);
/// 	match model {
/// 		Ok(model) => println!("Model created successfully"),
/// 		Err(e) => println!("Error: {}", e),
/// 	}
/// ```
pub fn create_model(file_path: &String) -> Result<HashMap<Data, Option<Tensor>>, String> {
    let nodes_wrapped: Result<HashMap<Data, Option<Tensor>>, ()> = extract_data(&file_path);
    if nodes_wrapped.is_err() {
        return Err(("File path is not valid or file cannot be deserialized, please input the correct file path and try again:").to_string());
    }
    let mut nodes: HashMap<Data, Option<Tensor>> = nodes_wrapped.unwrap();
    if let Ok(_) = insert_embeddings(&mut nodes) {
        return Ok(nodes);
    }
    return Err("Error inserting embeddings".to_string());
}

/// # pass_description
/// This function is used when the user wants to find recommendations based on a description
/// 
/// # Arguments
/// ```no_run
///    	* node_embeddings: &HashMap<Data, Option<Tensor> - The model
///    	* description_input: String - The description input by the user
/// 	* tags_input: String - The tags input by the user, each tag separated by a comma. If the user doesn't want to filter by tags, they can enter NONE
/// 	* num_recommendations: usize - The number of recommendations the user wants
/// ```
/// 
/// # Returns
/// ```no_run
///   	* Result<Vec<String, f32>, ()> - A vector of (Item name, similarity) tuples if recommendations were found, otherwise Err
/// ```
/// 
/// # Example
/// ```no_run
/// 	let recommendations = pass_description(&model, "description".to_string(), "tag1,tag2".to_string(), 10);
/// 	match recommendations {
/// 		Ok(recommendations) => {
/// 			println!("Recommendations:");
/// 			for recommendation in recommendations {
/// 				println!("{}% {}", (recommendation.1 * 100.0).round(), recommendation.0);
/// 			}
/// 		},
/// 		Err(_) => println!("No recommendations found"),
/// 	}
/// ```
pub fn pass_description(node_embeddings: &HashMap<Data, Option<Tensor>>, description_input: String, tags_input: String, num_recommendations: usize) -> Result<Vec<(String, f32)>, ()> {

    // When we are given a description, we need to create an embedding for it and then find recommendations based on that
    let input_embedding = create_input_embedding(&description_input).unwrap().unwrap();
    let recommendations = get_recommendations(node_embeddings, None, &input_embedding, &tags_input, num_recommendations);
    recommendations
}

/// # pass_item
/// This function is used when the user wants to find recommendations based on a specific item that is already in the model
/// 
/// # Arguments
/// ```no_run
///   	* node_embeddings: &HashMap<Data, Option<Tensor> - The model
///  	* item: String - The item the user wants recommendations for
/// 	* tags_input: String - The tags input by the user, each tag separated by a comma. If the user doesn't want to filter by tags, they can enter NONE
/// 	* num_recommendations: usize - The number of recommendations the user wants
/// ```
/// 
/// # Returns
/// ```no_run
///   	* Result<Vec<String, f32>, ()> - A vector of (Item name, similarity) tuples if recommendations were found, otherwise Err
/// ```
/// 
/// # Example
/// ```no_run
/// 	let recommendations = pass_item(&model, "item".to_string(), "tag1,tag2".to_string(), 10);
/// 	match recommendations {
/// 		Ok(recommendations) => {
/// 			println!("Recommendations:");
/// 			for recommendation in recommendations {
///                 println!("{}% {}", (recommendation.1 * 100.0).round(), recommendation.0);
/// 			}
/// 		},
/// 		Err(_) => println!("No recommendations found"),
/// 	}
/// ```
pub fn pass_item(node_embeddings: &HashMap<Data, Option<Tensor>>, item: String, tags_input: String, num_recommendations: usize) -> Result<Vec<(String, f32)>, ()> {

    // When we want to find items similar to a specific item, we need to make sure that the item is in the embeddings and then retrieve the embedding
    let input_embedding = {
        if let Ok(embedding) = find_embedding(node_embeddings, &item) {
            embedding
        } else {
            return Err(());
        }
    };
    let recommendations = get_recommendations(node_embeddings, Some(&item), &input_embedding, &tags_input, num_recommendations);
    recommendations
}