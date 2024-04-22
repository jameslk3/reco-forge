use std::collections::HashMap;

use reco_forge::helpers::post_recommendation::get_recommendations;
use reco_forge::helpers::types::*;
use reco_forge::helpers::pre_recommendation::*;

use candle::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = String::new();
    println!("Please enter the file path:");
    let mut nodes_wrapped: Result<HashMap<Data, Option<Tensor>>, ()>;
    loop {
        std::io::stdin().read_line(&mut path).expect("Failed to read line");
        path = path.trim().to_string();

        nodes_wrapped = extract_data(&path);

        if let Ok(_nodes) = &nodes_wrapped {
            break; // Exit the loop if data extraction was successful
        } else {
            println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
            path.clear();
        }
    }
    let mut nodes: HashMap<Data, Option<Tensor>> = nodes_wrapped.unwrap();
    let node_embeddings = get_embeddings(&mut nodes).unwrap();
    
    let mut counter = 0;
    for (_key, value) in node_embeddings.iter() {
	    if value.is_some() {
		    println!("THERE IS SOMETHING IN THE EMBEDDING {}", counter);
	    } else {
		    println!("NOPE {}", counter);
	    }
        counter += 1;
    }
    println!("Input wanted tags. If you don't want to filter by tags, enter NONE");
    let mut tags_input: String = String::new();
    std::io::stdin().read_line(&mut tags_input).expect("Failed to read line");
    tags_input = tags_input.trim().to_string();

    println!("Input what you want.");
    let mut description_input: String = String::new();
    std::io::stdin().read_line(&mut description_input).expect("Failed to read line");
    description_input = description_input.trim().to_string();

    println!("{}", description_input);
    let recommendations = get_recommendations(&node_embeddings, &description_input, &tags_input, 10).unwrap();
    for recommendation in recommendations {
        println!("{}", recommendation);
    }
    Ok(())
}

