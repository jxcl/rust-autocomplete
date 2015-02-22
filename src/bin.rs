#![feature(core,old_path,old_io)]
extern crate autocomplete;

use std::old_io;
use std::old_io::BufferedReader;
use std::old_io::File;
use std::char::from_u32;

use autocomplete::models::SimpleWordModel;

fn clean_line(line: String) -> String {
    let mut new_string = String::new();
    let line_bytes = line.bytes();
    for byte in line_bytes {
        if byte == 32 {
            new_string.push(from_u32(byte as u32).unwrap());
        } else if byte >= 97 && byte <= 122 {
            new_string.push(from_u32(byte as u32).unwrap());
        } else if byte >= 64 && byte <= 90 {
            new_string.push(from_u32((byte + 32) as u32).unwrap());
        }
    }
    new_string
}

fn train_model(model: &mut SimpleWordModel, path: Path) {
    let mut file = BufferedReader::new(File::open(&path));
    for line in file.lines() {
        let cleaned_line = clean_line(line.unwrap());
        model.train_str(cleaned_line.as_slice());
    }
}

fn main() {
    let mut model = SimpleWordModel::new();
    let file_path = Path::new("big.txt");
    println!("Training.");
    train_model(&mut model, file_path);
    println!("Finalizing.");
    let predictor = model.finalize();
    println!("Ready.");
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
