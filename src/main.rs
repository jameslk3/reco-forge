use core::time;
use std::collections::HashMap;
use std::io::stdout;
use std::io::Write;
use std::thread;
use std::io::BufRead;

use reco_forge::helpers::post_recommendation::recommendations;
use reco_forge::helpers::types::*;
use reco_forge::helpers::pre_recommendation::*;
use text_io::read;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Input a file path:");
    let mut path: String = read!();
    let mut nodes_wrapped: Result<HashMap<Data, Option<Tensor>>, ()> = extract_data(&String::from(path));
    while nodes_wrapped.is_err() {
        println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
        path = read!();
        nodes_wrapped = extract_data(&String::from(path));
    }
    let mut nodes: HashMap<Data, Option<Tensor>> = nodes_wrapped.unwrap();
    let result = get_embeddings(&mut nodes).unwrap();
    
    let mut counter = 0;
    for (_key, value) in result.iter() {
	    if value.is_some() {
		    println!("THERE IS SOMETHING IN THE EMBEDDING {}", counter);
	    } else {
		    println!("NOPE {}", counter);
	    }
        counter += 1;
    }
    println!("Input wanted tags. If you don't want to filter by tags, enter NONE");
    let tags_input: String = read!();
    println!("Input what you want.");
    let description_input: String = read!();
    // io::stdin().lock().read_line(&mut description_input).expect("problem");
    let _ = stdout().flush();
    println!("{}", description_input);
    let recommendations = recommendations(&result, &description_input, &tags_input).unwrap();
    for recommendation in recommendations {
        println!("{}", recommendation);
    }
    Ok(())
}


use candle_transformers::models::distilbert::{Config, DistilBertModel, DTYPE};

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,
}

/*
impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(DistilBertModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "distilbert-base-uncased".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let model = DistilBertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn get_mask(size: usize, device: &Device) -> Tensor {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device).unwrap()
}
*/

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}