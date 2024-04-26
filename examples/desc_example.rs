use core::panic;

use reco_forge::{create_model, pass_description, Data, Tensor, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("This is an example of how you can describe something and find similar items to it in a dataset.");
    println!("Either input a path to your JSON file or press enter to use the default dataset (small dataset on movies).");
    println!("Creating the model may take a while, please be patient.");
    println!();

    let model_wrapped: Result<HashMap<Data, Option<Tensor>>, ()>;

    let mut path = String::new();
    println!("Please enter the file path:");
    std::io::stdin().read_line(&mut path).expect("Failed to read line");
    loop {
        path = path.trim().to_string();
        if path.is_empty() {
            path = "sample-json/movies/movies_clean_mini.json".to_string();
            if let Ok(model) = create_model(&path) {
                model_wrapped = Ok(model);
                break;
            } else {
                panic!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
            }
        }
        if let Ok(model) = create_model(&path) {
            model_wrapped = Ok(model);
            break;
        } else {
            println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
            path.clear();
            std::io::stdin().read_line(&mut path).expect("Failed to read line");
        }
    }

    let model: HashMap<Data, Option<Tensor>> = model_wrapped.unwrap();

    println!("Input tags that you would like to use to filter, else enter NONE");
    let mut tags_input: String = String::new();
    std::io::stdin().read_line(&mut tags_input).expect("Failed to read line");
    tags_input = tags_input.trim().to_string();

    println!("Describe what you want to be recommended:");
    let mut query: String = String::new();
    std::io::stdin().read_line(&mut query).expect("Failed to read line");
    query = query.trim().to_string();
    println!();

    let recommendations = pass_description(&model, query, tags_input, 10);
    match recommendations {
        Ok(recommendations) => {
            println!("Recommendations:");
            for recommendation in recommendations {
                println!("{}% {}", (recommendation.1 * 100.0).round(), recommendation.0);
            }
        },
        Err(_) => println!("No recommendations found, either an error occurred or the item you inputted is not in the dataset. Please try again."),
    }
    Ok(())
}
