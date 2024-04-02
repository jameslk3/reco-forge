use reco_forge::types::*;
use reco_forge::functions::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let nodes: Vec<Data> = extract_data(&String::from("./src/sample-json/games_clean_mini.json")).unwrap();
    let embedding_map = get_embeddings(&nodes);
    let end = start.elapsed();
    println!("Time elapsed: {:?}", end);
    Ok(())
}