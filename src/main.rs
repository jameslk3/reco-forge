use reco_forge::helpers::types::*;
use reco_forge::helpers::pre_recommendation::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut nodes: Vec<DataWithEmbeddings> = extract_data(&String::from("./src/sample-json/movies/movies_clean_mini.json")).unwrap();
    let _result = get_embeddings(&mut nodes);
    let end = start.elapsed();
    println!("Time elapsed: {:?}", end);

	for x in 0..nodes.len() {
		if nodes[x].embedding.is_some() {
			println!("THERE IS SOMETHING IN THE EMBEDDING {}", x);
		} else {
			println!("NOPE {}", x);
		}
	}


    Ok(())
}

// use candle_transformers::models::distilbert::{Config, DistilBertModel, DTYPE};

// use anyhow::{Error as E, Result};
// use candle::{Device, Tensor};
// use candle_nn::VarBuilder;
// use clap::Parser;
// use hf_hub::{api::sync::Api, Repo, RepoType};
// use tokenizers::Tokenizer;

// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// struct Args {
//     /// Run on CPU rather than on GPU.
//     #[arg(long)]
//     cpu: bool,

//     /// Enable tracing (generates a trace-timestamp.json file).
//     #[arg(long)]
//     tracing: bool,

//     /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
//     #[arg(long)]
//     model_id: Option<String>,

//     #[arg(long)]
//     revision: Option<String>,

//     /// When set, compute embeddings for this prompt.
//     #[arg(long)]
//     prompt: Option<String>,

//     /// Use the pytorch weights rather than the safetensors ones
//     #[arg(long)]
//     use_pth: bool,

//     /// The number of times to run the prompt.
//     #[arg(long, default_value = "1")]
//     n: usize,

//     /// L2 normalization for embeddings.
//     #[arg(long, default_value = "true")]
//     normalize_embeddings: bool,
// }

// impl Args {
//     fn build_model_and_tokenizer(&self) -> Result<(DistilBertModel, Tokenizer)> {
//         let device = candle_examples::device(self.cpu)?;
//         let default_model = "distilbert-base-uncased".to_string();
//         let default_revision = "main".to_string();
//         let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
//             (Some(model_id), Some(revision)) => (model_id, revision),
//             (Some(model_id), None) => (model_id, "main".to_string()),
//             (None, Some(revision)) => (default_model, revision),
//             (None, None) => (default_model, default_revision),
//         };

//         let repo = Repo::with_revision(model_id, RepoType::Model, revision);
//         let (config_filename, tokenizer_filename, weights_filename) = {
//             let api = Api::new()?;
//             let api = api.repo(repo);
//             let config = api.get("config.json")?;
//             let tokenizer = api.get("tokenizer.json")?;
//             let weights = if self.use_pth {
//                 api.get("pytorch_model.bin")?
//             } else {
//                 api.get("model.safetensors")?
//             };
//             (config, tokenizer, weights)
//         };
//         let config = std::fs::read_to_string(config_filename)?;
//         let config: Config = serde_json::from_str(&config)?;
//         let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

//         let vb = if self.use_pth {
//             VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
//         } else {
//             unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
//         };
//         let model = DistilBertModel::load(vb, &config)?;
//         Ok((model, tokenizer))
//     }
// }

// fn get_mask(size: usize, device: &Device) -> Tensor {
//     let mask: Vec<_> = (0..size)
//         .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
//         .collect();
//     Tensor::from_slice(&mask, (size, size), device).unwrap()
// }

// fn other_main() -> Result<()> {

// 	let start = std::time::Instant::now();

//     use tracing_chrome::ChromeLayerBuilder;
//     use tracing_subscriber::prelude::*;

// 	let data = extract_data(&String::from("./src/sample-json/games_clean_mini.json")).unwrap();

//     let args = Args::parse();
//     let _guard = if args.tracing {
//         println!("tracing...");
//         let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
//         tracing_subscriber::registry().with(chrome_layer).init();
//         Some(guard)
//     } else {
//         None
//     };
//     let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
//     let device = &model.device;

//     let tokenizer = tokenizer
//         .with_padding(None)
//         .with_truncation(None)
//         .map_err(E::msg)?;

// 	let summaries: Vec<&str> = data.iter().map(|d| d.summary.as_str()).collect();
//     let tokens_list: Vec<Vec<u32>> = {
// 		summaries.into_iter().map(|summary| {
// 			tokenizer
// 			.encode(summary, true)
// 			.map_err(E::msg)
// 			.unwrap()
// 			.get_ids()
// 			.to_vec()
// 		}).collect()
// 	};

// 	let mut token_ids_list: Vec<Tensor> = Vec::new();
// 	let mut mask_list: Vec<Tensor> = Vec::new();
// 	for tokens in tokens_list {
// 		let token_id = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
// 		let mask = get_mask(tokens.len(), device);
// 		token_ids_list.push(token_id);
// 		mask_list.push(mask);
// 	}

// 	let before_forward = start.elapsed();
// 	println!("before forward {:?}", before_forward);
// 	let mut embedding_list = Vec::new();
// 	// let (tx, rx) = std::sync::mpsc::channel();
// 	for (index, (token_ids, mask)) in token_ids_list.iter().zip(mask_list.iter()).enumerate() {
		
// 		let embedding = model.forward(&token_ids, mask)?;
// 		let elapsed = start.elapsed();
// 		println!("one embedding {:?}", elapsed);
// 		embedding_list.push(embedding);
// 	}

//     // println!("token_ids: {:?}", token_ids.to_vec2::<u32>());
//     // println!("mask: {:?}", mask.to_vec2::<u8>());

// 	println!("{:?}", embedding_list.len());

//     Ok(())
// }

// pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
//     Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
// }