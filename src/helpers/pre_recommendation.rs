use super::types::Data;
use std::collections::HashMap;
use std::fs::File;
use serde_json::from_reader;

/// Receives the path of a JSON file as a &String. The function tries to open
/// the file. If it doesn't, it will return Err.
/// After it opens the file, it deserializes the JSON file into a vector of Data objects.
/// If it successfully does so, it will return Ok. If it doesn't, it will return Err.
///
/// @param `file_name` - a String containing the file path of the JSON file
///
/// @return `Ok()` with the vector of data encapsulated in a `Result` enum [OR] `Err()` if the
/// file didn't open or didn't deserialize
pub(crate) fn extract_data(file_name: &String) -> Result<HashMap<Data, Option<Tensor>>, ()> {
    // Opens file
    let file_wrapped = File::open(file_name);
    if file_wrapped.is_err() {
        return Err(());
    }
    let file = file_wrapped.unwrap();

    // Deserializes into Data object
    let vector_of_data_wrapped: Result<Vec<Data>, serde_json::Error> = from_reader(file);
    if vector_of_data_wrapped.is_err() {
        return Err(());
    }
    let vector_of_data: Vec<Data> = vector_of_data_wrapped.unwrap();
    
    let mut vector_to_map: HashMap<Data, Option<Tensor>> = HashMap::new();

    for data in vector_of_data {
        vector_to_map.insert(data.clone(), None);
    }

    return Ok(vector_to_map);
}

use super::types::Args;
use super::utils::*;
use anyhow::{Error as E, Result};
use candle::Tensor;
use clap::Parser;

pub(crate) fn insert_embeddings(data: &mut HashMap<Data, Option<Tensor>>) -> Result<()> {
    let args = Args::parse();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    }

    // Tokenize the data
    let mut summaries: Vec<&str> = Vec::new();
    for (key, _value) in data.iter() {
        summaries.push(key.summary.as_str());
    }
    let tokens = tokenizer
        .encode_batch(summaries.to_vec(), true)
        .map_err(E::msg)?;

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
    // println!("running inference on batch {:?}", token_ids.shape());

    // Get the embeddings
    let embeddings = model.forward(&token_ids, &token_type_ids)?;
    // println!("generated embeddings {:?}", embeddings.shape());

    // Pool the embeddings
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = normalize_l2(&embeddings)?;
    // println!("pooled embeddings {:?}", embeddings.shape());

    // Insert embeddings into data
    for (i, (_key, value)) in data.iter_mut().enumerate() {
        *value = Some(embeddings.get(i)?);
    }

    Ok(())
}

pub(crate) fn find_embedding(data: &HashMap<Data, Option<Tensor>>, item_name: &String) -> Result<Tensor> {
    let item_name_cleaned = item_name.trim().to_lowercase();
    for (key, value) in data.iter() {
        if key.name.to_lowercase() == item_name_cleaned {
            return Ok(value.clone().unwrap());
        }
    }
    Err(E::msg("Item not found"))
}