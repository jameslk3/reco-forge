pub mod types {
    use serde::Deserialize;
    use std::fmt;

    #[derive(Debug, Deserialize, Clone)]
    pub struct Data {
        pub id: i32,
        name: String,
        pub summary: String,
        pub embedding: Option<Vec<f32>>,
        tags: Vec<String>,
    }

    impl fmt::Display for Data {
        // Implement the Display trait for Link
        // Should display only the contents of thing
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Name: {}, Summary: {}", self.name, self.summary)
        }
    }

    use anyhow::{Error as E, Result};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, DTYPE};
    use clap::Parser;
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use tokenizers::Tokenizer;

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    pub(crate) struct Args {
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

        /// Use tanh based approximation for Gelu instead of erf implementation.
        #[arg(long, default_value = "false")]
        approximate_gelu: bool,
    }

    impl Args {
        pub(crate) fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
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
            let config: Config = serde_json::from_str(&config)?;
            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

            let vb = if self.use_pth {
                VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
            } else {
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
            };

            let model = BertModel::load(vb, &config)?;
            Ok((model, tokenizer))
        }
    }
}

pub mod functions {
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
    pub fn extract_data(file_name: &String) -> Result<Vec<Data>, ()> {
        // Opens file
        let file = File::open(file_name).expect("File didn't open");

        // Deserializes into Data object
        let vector_of_data: Vec<Data> =
            serde_json::from_reader(file).expect("Deserialization failed");

        return Ok(vector_of_data);
    }

    use super::types::Args;
    use anyhow::{Error as E, Result};
    use candle_core::Tensor;
    use clap::Parser;
    use std::collections::HashMap;

    pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    pub fn get_embeddings(data: &Vec<Data>) -> Result<HashMap<i32, Tensor>> {
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

        // Create a hashmap mapping the id to the embeddings
        let mut embeddings_map = HashMap::new();
        for (i, d) in data.iter().enumerate() {
            embeddings_map.insert(d.id, embeddings.get(i)?);
        }

        Ok(embeddings_map)
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
