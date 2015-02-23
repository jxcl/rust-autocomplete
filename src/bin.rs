#![feature(old_path,old_io)]
extern crate autocomplete;

use std::old_io;

use autocomplete::simplemodel::SimpleWordPredictor;

fn main() {
    println!("Loading training data.");
    let predictor = SimpleWordPredictor::from_file(&Path::new("training_data.csv"));
    loop {
        print!("Input: ");
        let input = old_io::stdin().read_line().ok().expect("Failed to read line.");
        let output = predictor.predict(input.trim());
        println!("Score\tWord");
        for entry in output {
            println!("{}\t{}", entry.score, entry.word);
         }

    }
}
