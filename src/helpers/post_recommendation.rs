use std::collections::HashMap;
use anyhow::{Error as E, Result};
use candle::Tensor;
use clap::Parser;
use crate::helpers::{types::Args, utils::normalize_l2};
use super::types::Data;
use super::types::Recommendations;


pub fn recommendations(map: &HashMap<Data, Option<Tensor>>, description_input: &String, tags_input: &String) -> Result<Vec<String>, ()> {
    let num_recommendations = 5;
    let mut recommendations = Recommendations::new(num_recommendations);
    let mut filtered_map = map.clone();
    if tags_input != &String::from("NONE") {
        let filtered_map_wrapped = filter_by_tags(map, tags_input);
        if filtered_map_wrapped.is_err() {
            return Err(());
        }
        filtered_map = filtered_map_wrapped.unwrap();
    }
    let embedding_wrapped = embedding_for_input(description_input);
    if embedding_wrapped.is_err() {
        return Err(());
    }
    let embedding = embedding_wrapped.unwrap().unwrap();
    let a_dot_a_wrapped = &embedding * &embedding;
    if a_dot_a_wrapped.is_err() {
        return Err(());
    }
    let a_dot_a = a_dot_a_wrapped.unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    for (key, value) in filtered_map.iter() {
        let map_embedding_wrapped = value.clone();
        if map_embedding_wrapped.is_none() {
            return Err(());
        }
        let map_embedding = map_embedding_wrapped.clone().unwrap();
        let b_dot_b = (&map_embedding * &map_embedding).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let a_dot_b = (&embedding * &map_embedding).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let similarity = &a_dot_b / (&a_dot_a * &b_dot_b).sqrt();
        recommendations.insert_or_skip(key.name.clone(), similarity);
    }
    return Ok(recommendations.get_recommendations());
}

/// Receives the input of what the user wants as tags in a String.
/// The function will return a new `HashMap` with the nodes without the desired tags removed.
///
/// @param `map` - a `HashMap<Data, Option<Tensor>>`, which contains our nodes,
///  `tags_input` - a String containing what the user wants as tags
///
/// @return `Ok()` with a `HashMap<Data, Option<Tensor>>`, which is the new map with nodes w/o the desired tags removed [OR] `Err()`
fn filter_by_tags(map: &HashMap<Data, Option<Tensor>>, tags_input: &String) -> Result<HashMap<Data, Option<Tensor>>, ()> {
    let split: Vec<&str> = tags_input.split(',').collect();
    let mut new_map: HashMap<Data, Option<Tensor>> = HashMap::new();
    for str in split {
        let trimmed = str.trim();
        for (key, value) in map {
            if key.tags.contains(&String::from(trimmed)) {
                new_map.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(new_map)
}

/// Receives the input of what the user wants suggested as a String.
/// The function will return the embeddings of the String in question.
///
/// @param `description_input` - a String containing what the user wants suggested
///
/// @return `Ok()` with a Tensor (embedding) generated from the string [OR] `Err()`
fn embedding_for_input(description_input: &String) -> Result<Option<Tensor>> {
    let args = Args::parse();

    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    }

    // Tokenize the data
    let mut summaries: Vec<&str> = Vec::new();
    summaries.push(&description_input);
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

    Ok(Some(embeddings.get(0).unwrap()))
}