extern crate reco_forge;
use reco_forge::{Data, extract_data};

fn main() {
	let x = extract_data(&String::from("./src/sample-json/games_clean.json"));
	if x.is_err() {
		panic!();
	}
	print(x.unwrap());
}

fn print(vector: Vec<Data>) {
	for datum in vector {
		println!("{}", datum);
		println!("");
	}
}