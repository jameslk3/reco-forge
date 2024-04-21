use std::collections::HashMap;
use std::io::stdout;
use std::io::Write;

use reco_forge::helpers::post_recommendation::recommendations;
use reco_forge::helpers::types::*;
use reco_forge::helpers::pre_recommendation::*;
use text_io::read;


use candle::Tensor;

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
    let node_embeddings = get_embeddings(&mut nodes).unwrap();
    
    let mut counter = 0;
    for (_key, value) in node_embeddings.iter() {
	    if value.is_some() {
		    println!("THERE IS SOMETHING IN THE EMBEDDING {}", counter);
			println!("{}", value.clone().unwrap());
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
    let recommendations = recommendations(&node_embeddings, &description_input, &tags_input).unwrap();
    for recommendation in recommendations {
        println!("{}", recommendation);
    }
    Ok(())
}

