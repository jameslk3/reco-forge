pub mod types {

    use candle::Tensor;
    use serde::Deserialize;
    use std::fmt;

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
        pub embedding: Option<Tensor>
    }

    impl DataWithEmbeddings {
        pub fn new(id: i32, name: String, summary: String, tags: Vec<String>, embedding: Option<Tensor>) -> DataWithEmbeddings {
            DataWithEmbeddings {
                id: id,
                name: name,
                summary: summary,
                tags: tags,
                embedding: embedding
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

    use anyhow::{Error as E, Result};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
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
        #[arg(long, default_value = "true")]
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
    }
}

pub mod functions {
    use crate::types::DataWithEmbeddings;

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
        let vector_of_data: Vec<Data> =
            serde_json::from_reader(file).expect("Deserialization failed");

        let mut vector_of_data_mappings: Vec<DataWithEmbeddings> = Vec::new();

        for x in 0..vector_of_data.len() {
            let datum: DataWithEmbeddings = DataWithEmbeddings::new(vector_of_data[x].id.clone(), 
            vector_of_data[x].name.clone(), vector_of_data[x].summary.clone(),
             vector_of_data[x].tags.clone(), None);
            vector_of_data_mappings.push(datum);
        }
        
        return Ok(vector_of_data_mappings);
    }

    use super::types::Args;
    use anyhow::{Error as E, Result};
    use candle::Tensor;
    use clap::Parser;
    use super::utils::*;

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
}

pub(crate) mod utils {

    // use super::types::Data;

    /*
    pub fn print_data(vector: Vec<Data>) {
        for datum in vector {
            println!("{}", datum);
            println!("");
        }
    }
    */

    /*
    pub fn remove_punct(vector: &mut Vec<Data>) {
        for x in 0..vector.len() {
            let chars: Vec<char> = vector[x].summary.clone().chars().collect();
            let mut summary_no_punct = String::new();
            for y in 0..chars.len() {
                if chars[y] != '.' && chars[y] != ',' && chars[y] != '?' && chars[y] != '!' {
                    summary_no_punct += chars[y].clone().to_string().as_str();
                }
            }
            vector[x].summary = summary_no_punct;
        }
    }
    */

    /*
    pub fn remove_article(vector: &mut Vec<Data>) {
        for x in 0..vector.len() {
            let words: Vec<&str> = vector[x].summary.split(' ').collect();
            let mut summary_no_article = String::new();
            for y in words {
                if y != "the" && y != "a" && y != "an" && y != "some" && y != "and" && y != "is" && y != "in" && y != "as" && y != "of"
                    && y != "to" && y != "but" && y != "game" && y != "games" && y != "through" && y != "by" && y != "on" {
                    summary_no_article += y.to_string().as_str();
                    summary_no_article += " ";
                }
            }
            println!("{}", summary_no_article);
            vector[x].summary = summary_no_article;
        }
    }
    */

    use candle::Tensor;
    use anyhow::Result;

    pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
