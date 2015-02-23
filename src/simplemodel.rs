use predictionentry::PredictionEntry;

use std::old_io::File;
use std::old_io::BufferedReader;
use std::collections::HashMap;
use std::collections::hash_map::Entry;


/// Single-word entry trainer
///
/// SimpleWordTrainer uses a HashMap representation to train on word
/// frequency. It is fast for looking up words and incrementing their
/// count but since HashMap does not keep track of order, searching the
/// keys of a hashmap takes a long time. After the model is trained it
/// must be converted to a SimpleWordPredictor which stores the words
/// in lexical order and has an index of first letters.
#[derive(Debug)]
pub struct SimpleWordTrainer(HashMap<String, u32>);

/// Single-word prediction engine
///
/// SimpleWordPredictor uses a constant sized vector of entries indexed by
/// first letter. Prediction starts from that index and continues until
/// the first letter in the vector changes.
#[derive(Debug)]
pub struct SimpleWordPredictor {
    entries: Vec<PredictionEntry>,
    ixs: HashMap<char, u32>,
}

impl SimpleWordPredictor {
    /// Given an initial input string, return 10 predictions.
    pub fn predict(&self, input: &str) -> Vec<PredictionEntry> {
        let mut predictions: Vec<PredictionEntry> = Vec::new();
        let iter = self.entries.iter();
        let first_letter = input.char_at(0);
        let skip_n = self.ixs.get(&first_letter);

        match skip_n {
            Some(n) => {
                let iter = iter.skip(*n as usize);
                for entry in iter {
                    let word = entry.word.as_slice();
                    if word.char_at(0) != first_letter {
                        break;
                    }

                    if word.starts_with(input) {
                        predictions.push(entry.clone());
                    }
                }

                predictions.sort_by(|a, b| {
                    b.score.cmp(&a.score)
                });

                predictions.truncate(10);

                predictions
            },
            None => {
                return predictions;
            },
        }
    }

    /// Load training data from a CSV file.
    pub fn from_file(path: &Path) -> SimpleWordPredictor {
        let mut entries = Vec::new();
        let mut file = BufferedReader::new(File::open(path));
        for line_res in file.lines() {
            let line = line_res.unwrap();
            let line = line.trim();
            let str_entry: Vec<&str> = line.split(',').collect();
            let word: String = String::from_str(str_entry[0]);
            let n = str_entry[1].parse().ok().unwrap();
            entries.push(PredictionEntry {word: word, score: n});
        }

        let ixs = generate_ixs(&entries);

        SimpleWordPredictor {entries: entries, ixs: ixs}
    }

    /// Save training data to a CSV file.
    pub fn to_file(&self, path: &Path) {
        let mut file = File::create(path);
        for entry in &self.entries {
            write!(&mut file, "{},{}\n", &entry.word, entry.score)
                .unwrap();
        }
    }
}

impl SimpleWordTrainer {
    /// Create a trainer trained on an initial str.
    pub fn from_str(input: &str) -> SimpleWordTrainer {
        let model_hm: HashMap<String, u32> = HashMap::new();
        let mut model = SimpleWordTrainer(model_hm);
        let v_input = input.split(' ').collect();
        count_words(&mut model, &v_input);

        model
    }

    /// Create a trainer trained on an initial vector strs.
    pub fn from_vec(input: &Vec<&str>) -> SimpleWordTrainer {
        let model_hm: HashMap<String, u32> = HashMap::new();
        let mut model = SimpleWordTrainer(model_hm);

        count_words(&mut model, input);

        model
    }

    /// Create a new untrained trainer.
    pub fn new() -> SimpleWordTrainer {
        let model_hm: HashMap<String, u32> = HashMap::new();

        SimpleWordTrainer(model_hm)
    }

    /// Train the model on a string that will be split on spaces.
    pub fn train_str(&mut self, input: &str) {
        let v_input = input.split(' ').collect();
        count_words(self, &v_input);
    }

    /// Train the model on a vector of words.
    pub fn train_vec(&mut self, input: Vec<&str>) {
        count_words(self, &input);
    }

    pub fn train_word(&mut self, input: &str) {
        let &mut SimpleWordTrainer(ref mut model_hm) = self;
        train_single_word(model_hm, input);
    }

    /// Perform the calculations needed to predict the next word.
    pub fn finalize(self) -> SimpleWordPredictor {
        let SimpleWordTrainer(hm) = self;
        let size = hm.len();
        let mut entries = Vec::with_capacity(size);

        for (key, value) in hm {
            entries.push(PredictionEntry {word: key, score: value});
        }

        entries.sort();

        let ixs = generate_ixs(&entries);
        SimpleWordPredictor {entries: entries, ixs: ixs}
    }

    /// This method is not meant to be called from outside the library.
    pub fn debug_get_word_score(&self, word: &str) -> Option<&u32> {
        let &SimpleWordTrainer(ref hm) = self;

        hm.get(word)
    }

}

fn generate_ixs(entries: &Vec<PredictionEntry>) -> HashMap<char, u32>{
    let mut ixs: HashMap<char, u32> = HashMap::new();

    // There will be no newlines in the input. This is a placeholder
    // until the first time the loop runs.
    let mut last_c = '\n';
    let mut ix = 0;

    for entry in entries {
        let c = entry.word.char_at(0);
        if c != last_c {
            ixs.insert(c, ix);
            last_c = c;
        }
        ix += 1;
    }

    ixs
}

fn train_single_word(model_hm: &mut HashMap<String, u32>, word: &str) {
    if word.len() == 0 {
        return;
    }
    let string_word = String::from_str(word);
    let entry = model_hm.entry(string_word);
    match entry {
        Entry::Vacant(vacant_entry) => {
            vacant_entry.insert(1);
        },
        Entry::Occupied(mut occ_entry) => {
            let c = occ_entry.get_mut();
            *c += 1;
        }
    }
}

fn count_words(model: &mut SimpleWordTrainer, input: &Vec<&str>) {
    let &mut SimpleWordTrainer(ref mut model_hm) = model;
    for word in input {
        train_single_word(model_hm, word);
    }
}

#[cfg(test)]
mod tests {
    use simplemodel::SimpleWordTrainer;
    use predictionentry::PredictionEntry;

    #[test]
    fn test_debug() {
        let model = SimpleWordTrainer::from_str("world domination is my profession hello hello");

        assert_eq!(&1, model.debug_get_word_score("world").unwrap());
    }

    #[test]
    fn test_from_str() {
        let model = SimpleWordTrainer::from_str("world domination is my profession hello hello");

        let SimpleWordTrainer(hash_map) = model;

        assert_eq!(1, *hash_map.get("world").unwrap());
        assert_eq!(2, *hash_map.get("hello").unwrap());
    }

    #[test]
    fn test_from_vec() {
        let model = SimpleWordTrainer::from_vec(&vec!["rabbit", "rabbit", "hare"]);

        let SimpleWordTrainer(hash_map) = model;

        assert_eq!(1, *hash_map.get("hare").unwrap());
        assert_eq!(2, *hash_map.get("rabbit").unwrap());
    }

    #[test]
    fn test_train_str() {
        let mut model = SimpleWordTrainer::new();

        model.train_str("hello hello hello there there");

        let SimpleWordTrainer(hash_map) = model;
        assert_eq!(3, *hash_map.get("hello").unwrap());
        assert_eq!(2, *hash_map.get("there").unwrap());
    }

    #[test]
    fn test_train_vec() {
        let mut model = SimpleWordTrainer::new();

        model.train_vec(vec!["hello", "hello", "hello", "what",
                             "is", "this"]);

        let SimpleWordTrainer(hash_map) = model;
        assert_eq!(3, *hash_map.get("hello").unwrap());
        assert_eq!(1, *hash_map.get("what").unwrap());
    }

    #[test]
    fn test_finalize() {
        let mut model = SimpleWordTrainer::new();

        model.train_str(concat!["anybody can become angry that is easy but to be ",
                                "angry with the right person and to the right degree ",
                                "and at the right time and for the right purpose ",
                                "and in the right way that is not within everybody's ",
                                "power and is not easy"]);

        let predictor = model.finalize();

        assert_eq!(predictor.entries[0].word, "and");
        assert_eq!(predictor.entries[0].score, 5);
    }

    #[test]
    fn test_predict() {
        let mut model = SimpleWordTrainer::new();

        model.train_str(concat!["anybody can become angry that is easy but to be ",
                                "angry with the right person and to the right degree ",
                                "and at the right time and for the right purpose ",
                                "and in the right way that is not within everybody's ",
                                "power and is not easy"]);

        let predictor = model.finalize();
        let prediction = predictor.predict("a");

        assert_eq!(prediction[0], PredictionEntry {word: String::from_str("and"), score: 4});
    }

}
