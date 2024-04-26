use super::types::Data;
use super::types::Recommendations;
use crate::helpers::{types::Args, utils::normalize_l2};
use anyhow::{Error as E, Result};
use candle::Tensor;
use clap::Parser;
use std::collections::HashMap;

/// Receives the input of what the user wants suggested as a String.
/// The function will return the embedding of the String in question.
///
/// @param `description_input` - a String containing what the user wants suggested
///
/// @return `Ok()` with a Tensor (embedding) generated from the string [OR] `Err()`
pub(crate) fn create_input_embedding(description_input: &String) -> Result<Option<Tensor>> {
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

    Ok(Some(embeddings.get(0).unwrap()))
}

pub(crate) fn get_recommendations(
    data: &HashMap<Data, Option<Tensor>>,
    raw_input: Option<&String>,
    input_embedding: &Tensor,
    tags_input: &String,
    num_recommendations: usize,
) -> Result<Vec<(String, f32)>, ()> {
    // Closure to filter out recommendations based on tags
    let tags_to_match = tags_input
        .split(',')
        .collect::<Vec<&str>>()
        .iter()
        .map(|x| x.trim().to_lowercase())
        .collect::<Vec<String>>();
    let through_filter = |tags: &Vec<String>| -> bool {
        if tags_input == &String::from("NONE") {
            true
        } else {
            let mut through_filter = true;
            for tag_to_match in tags_to_match.iter() {
                if !tags
                    .iter()
                    .map(|x| x.to_lowercase())
                    .collect::<Vec<String>>()
                    .contains(&String::from(tag_to_match))
                {
                    through_filter = false;
                    break;
                }
            }
            through_filter
        }
    };

    // Precompute the dot product of the input with itself
    let a_dot_a_wrapped = input_embedding * input_embedding;
    if a_dot_a_wrapped.is_err() {
        return Err(());
    }
    let a_dot_a = a_dot_a_wrapped
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    // Clean the input if it is an item because we want to make sure not to include the same item in the recommendations
    let input_cleaned = match raw_input {
        Some(input) => Some(input.trim().to_lowercase()),
        None => None,
    };

    // Compare the input with all the embeddings in the data and store the recommendations
    let mut recommendations = Recommendations::new(num_recommendations);
    for (key, value) in data.iter() {
        // Skip case
        if (input_cleaned.is_some() && input_cleaned.as_ref() == Some(&key.name.to_lowercase()))
            || !through_filter(&key.tags)
        {
            continue;
        }
        let map_embedding_wrapped = value.clone();
        if map_embedding_wrapped.is_none() {
            return Err(());
        }
        let map_embedding = map_embedding_wrapped.unwrap();
        let b_dot_b = (&map_embedding * &map_embedding)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let a_dot_b = (input_embedding * &map_embedding)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let similarity = &a_dot_b / (&a_dot_a * &b_dot_b).sqrt();
        recommendations.insert_or_skip(key.name.clone(), similarity);
    }
    return Ok(recommendations.get_recommendations());
}
