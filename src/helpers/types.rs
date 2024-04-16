use anyhow::{Error as E, Result as OtherResult};
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::{BertModel, Config, HiddenAct, DTYPE},
    distilbert::DistilBertModel,
};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Deserialize;
use std::fmt;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, Clone)]
pub struct Data {
    pub id: i32,
    pub name: String,
    pub summary: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DataWithEmbeddings {
    pub id: i32,
    pub name: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub embedding: Option<Tensor>,
}

impl DataWithEmbeddings {
    pub fn new(
        id: i32,
        name: String,
        summary: String,
        tags: Vec<String>,
        embedding: Option<Tensor>,
    ) -> DataWithEmbeddings {
        DataWithEmbeddings {
            id: id,
            name: name,
            summary: summary,
            tags: tags,
            embedding: embedding,
        }
    }
}

impl fmt::Display for Data {
    // Implement the Display trait for Link
    // Should display only the contents of thing
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Name: {}, Summary: {}", self.name, self.summary)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "true")]
    approximate_gelu: bool,
}

impl Args {
    pub(crate) fn build_model_and_tokenizer(&self) -> OtherResult<(BertModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
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
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }

    // pub(crate) fn build_different_model_and_tokenizer(
    //     &self,
    // ) -> Result<(DistilBertModel, Tokenizer)> {
    //     let device = candle_examples::device(self.cpu)?;
    //     let default_model = "distilbert-base-uncased".to_string();
    //     let default_revision = "main".to_string();
    //     let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
    //         (Some(model_id), Some(revision)) => (model_id, revision),
    //         (Some(model_id), None) => (model_id, "main".to_string()),
    //         (None, Some(revision)) => (default_model, revision),
    //         (None, None) => (default_model, default_revision),
    //     };

    //     let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    //     let (config_filename, tokenizer_filename, weights_filename) = {
    //         let api = Api::new()?;
    //         let api = api.repo(repo);
    //         let config = api.get("config.json")?;
    //         let tokenizer = api.get("tokenizer.json")?;
    //         let weights = if self.use_pth {
    //             api.get("pytorch_model.bin")?
    //         } else {
    //             api.get("model.safetensors")?
    //         };
    //         (config, tokenizer, weights)
    //     };
    //     let config = std::fs::read_to_string(config_filename)?;
    //     let config: Config = serde_json::from_str(&config)?;
    //     let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    //     let vb = if self.use_pth {
    //         VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
    //     } else {
    //         unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
    //     };
    //     let model = DistilBertModel::load(vb, &config)?;
    //     Ok((model, tokenizer))
    // }
}
