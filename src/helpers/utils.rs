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

use anyhow::Result;
use candle::Tensor;

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
