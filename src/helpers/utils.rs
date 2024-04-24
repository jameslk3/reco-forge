// use super::types::Data;

/*
pub fn print_data(vector: Vec<Data>) {
    for datum in vector {
        println!("{}", datum);
        println!("");
    }
}
*/

pub fn filter(summary: String) -> String {
    remove_punct(remove_num(remove_article(summary)))
}


fn remove_punct(summary: String) -> String {
    let chars: Vec<char> = summary.clone().chars().collect();
    let mut summary_no_punct = String::new();
    for y in 0..chars.len() {
        if chars[y] != '.' && chars[y] != ',' && chars[y] != '?' && chars[y] != '!' && chars[y] != '-' {
            summary_no_punct += chars[y].clone().to_string().as_str();
        }
    }
    summary_no_punct
}

fn remove_num(summary: String) -> String {
    let chars: Vec<char> = summary.clone().chars().collect();
    let mut summary_no_punct = String::new();
    for y in 0..chars.len() {
        if chars[y] != '0' && chars[y] != '1' && chars[y] != '2' && chars[y] != '3' 
            && chars[y] != '4' && chars[y] != '5' && chars[y] != '6' 
            && chars[y] != '7' && chars[y] != '8' && chars[y] != '9' {
            summary_no_punct += chars[y].clone().to_string().as_str();
        }
    }
    summary_no_punct
}



fn remove_article(summary: String) -> String {
    let words: Vec<&str> = summary.split(" ").collect();
    let mut summary_no_article = String::new();
    for word in words {
        if word != "the" && word != "a" && word != "an" && word != "some" && word != "and" && word != "is" && word != "in" && word != "as" && word != "of"
            && word != "to" && word != "but" && word != "game" && word != "games" && word != "through" && word != "bword" && word != "on" {
            summary_no_article += word.to_string().as_str();
            summary_no_article += " ";
        }
    }
    summary_no_article
}


use anyhow::Result;
use candle::Tensor;

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
