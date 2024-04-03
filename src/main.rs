use reco_forge::types::*;
use reco_forge::functions::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut nodes: Vec<DataWithEmbeddings> = extract_data(&String::from("./src/sample-json/games_clean_mini.json")).unwrap();
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