use super::types::DataWithEmbeddings;

use super::types::Data;
use std::fs::File;

/// Receives the path of a JSON file as a &String. The function tries to open
/// the file. If it doesn't, it will return Err.
/// After it opens the file, it deserializes the JSON file into a vector of Data objects.
/// If it successfully does so, it will return Ok. If it doesn't, it will return Err.
///
/// @param `file_name` - a String containing the file path of the JSON file
///
/// @return `Ok()` with the vector of data encapsulated in a `Result` enum [OR] `Err()` if the
/// file didn't open or didn't deserialize
pub fn extract_data(file_name: &String) -> Result<Vec<DataWithEmbeddings>, ()> {
    // Opens file
    let file = File::open(file_name).expect("File didn't open");

    // Deserializes into Data object
    let vector_of_data: Vec<Data> = serde_json::from_reader(file).expect("Deserialization failed");

    let mut vector_of_data_mappings: Vec<DataWithEmbeddings> = Vec::new();

    for x in 0..vector_of_data.len() {
        let datum: DataWithEmbeddings = DataWithEmbeddings::new(
            vector_of_data[x].id.clone(),
            vector_of_data[x].name.clone(),
            vector_of_data[x].summary.clone(),
            vector_of_data[x].tags.clone(),
            None,
        );
        vector_of_data_mappings.push(datum);
    }

    return Ok(vector_of_data_mappings);
}

use super::types::Args;
use super::utils::*;
use anyhow::{Error as E, Result};
use candle::Tensor;
use clap::Parser;

pub fn get_embeddings(data: &mut Vec<DataWithEmbeddings>) -> Result<()> {
    let args = Args::parse();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    }

    // Tokenize the data
    let summaries: Vec<&str> = data.iter().map(|d| d.summary.as_str()).collect();
    let tokens = tokenizer
        .encode_batch(summaries.to_vec(), true)
        .map_err(E::msg)?;

    for x in 0..tokens.len() {
        println!("{:?}", tokens[x].get_tokens().join(" "));
    }

    // Convert the tokens to tensors
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("running inference on batch {:?}", token_ids.shape());

    // Get the embeddings
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    println!("generated embeddings {:?}", embeddings.shape());

    // Pool the embeddings
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = normalize_l2(&embeddings)?;
    println!("pooled embeddings {:?}", embeddings.shape());

    // Insert embeddings into data
    for x in 0..data.len() {
        data[x].embedding = Some(embeddings.get(x).unwrap());
    }

    Ok(())
}
