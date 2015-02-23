# rust-autocomplete

`rust-autocomplete` is a rudimentary word completion library built in
rust. It is inspired by [Rodrigo
Palacios](https://github.com/rodricios)'s ELI5 explanation of an
autocompletion AI found [here](https://github.com/rodricios/autocomplete).
I figured it would be a good way to continue learning Rust.

# Usage

Using `SimpleWordPredictor` is easy. First it must be trained.

## Training

Autocomplete will work better with a larger corpus of training data available to it.
In this repository are provided two types of training data. The first is the file
named `training_data.csv`. This is an already processed count of a large amount of
input text. The other is the file named `big.txt` which is provided by [Peter Norvig]
(http://norvig.com). This is a raw collection of several books.

I also recommend watching Peter Norvig's lecture titled [The Unreasonable Effectiveness
of Data](https://www.youtube.com/watch?v=yvDCzhbjYWs).

### With `training_data.csv`
If you use the provided training data found in `training_data.csv`,
you only have to call `SimpleWordPredictor::from_file()` with a path
to the training data.

### With `big.txt` or other corpus
This method requires you to do a bit more heavy lifting. You will need open your corpus
and make sure the only characters are the ones that you want in your training data. I
settled on the characters `[a-z]` and spaces. You then feed this data into `SimpleWordTrainer`
using its `train_str()` method. Before you can predict, `SimpleWordTrainer` must be converted
to `SimpleWordPredictor`, which changes its internal representation of the training data.

This is how I trained autocomplete to create `training_data.csv`:

```rust
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

fn train_model(model: &mut SimpleWordTrainer, path: Path) {
    let mut file = BufferedReader::new(File::open(&path));
    for line in file.lines() {
        let cleaned_line = clean_line(line.unwrap());
        model.train_str(cleaned_line.as_slice());
    }
}

fn main() {
    let mut model = SimpleWordTrainer::new();
    let file_path = Path::new("big.txt");
    println!("Training.");
    train_model(&mut model, file_path);
    println!("Finalizing.");
    let predictor = model.finalize();
    // Save predictor here or use it to run predictions
}
```

## Predicting

Call `SimpleWordPredictor.predict()` with a `&str` to get back a `vec<SimpleWordEntry>`.
`SimpleWordEntry` has public fields `score` and `word`.

```rust
    loop {
        print!("Input: ");
        let input = old_io::stdin().read_line().ok().expect("Failed to read line.");
        let output = predictor.predict(input.trim());
        println!("Score\tWord");
        for entry in output {
            println!("{}\t{}", entry.score, entry.word);
         }
    }
```