# How to run our project

### Clone git repo (recommended):
- Clone the github repository
- Run "cargo run --example description" or "cargo run --example item"
- You can also work with the other provided JSON files in the sample-json directory

### Install the crate:
- Add the following line to your Cargo.toml file: reco-forge = "0.1.2" or run "cargo add reco-forge"
```toml
[dependencies]
reco-forge = "0.1.2"
```
- Follow the documentation [here](https://docs.rs/reco-forge/0.1.2/reco_forge/) to write your own code or simply copy the example below
- Either create your own JSON file following the requirements or use one in the git repo under sample-json
- We recommend sample-json/movies/movies_clean_10.json because it provides the highest quality descriptions and is also small (large or even medium sized files take very long)
```rust
use reco_forge::{create_model, pass_description, Data, Tensor, HashMap}; // Can also use pass_item

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
       Err(_) => println!("No recommendations found"),
   }
   Ok(())
}
```