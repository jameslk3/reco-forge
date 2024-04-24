use reco_forge::{create_model, pass_description, Data, Tensor, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = String::new();
    println!("Please enter the file path:");

    let model_wrapped: Result<HashMap<Data, Option<Tensor>>, ()>;

    loop {
        std::io::stdin().read_line(&mut path).expect("Failed to read line");
        path = path.trim().to_string();

        if let Ok(model) = create_model(&path) {
            model_wrapped = Ok(model);
            break;
        } else {
            println!("File path is not valid or file cannot be deserialized, please input the correct file path and try again:");
            path.clear();
        }
    }
    
    let model: HashMap<Data, Option<Tensor>> = model_wrapped.unwrap();

    println!("Input wanted tags. If you don't want to filter by tags, enter NONE");
    let mut tags_input: String = String::new();
    std::io::stdin().read_line(&mut tags_input).expect("Failed to read line");
    tags_input = tags_input.trim().to_string();

    println!("Input what you want.");
    let mut query: String = String::new();
    std::io::stdin().read_line(&mut query).expect("Failed to read line");
    query = query.trim().to_string();
    println!();

    let recommendations = pass_description(&model, query, tags_input, 10);
    match recommendations {
        Ok(recommendations) => {
            println!("Recommendations:");
            for recommendation in recommendations {
                println!("{}", recommendation);
            }
        },
        Err(_) => println!("No recommendations found"),
    }
    Ok(())
}
