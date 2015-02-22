#![feature(collections,std_misc,core,old_path,old_io)]
// This library is inspired by Rodrigo Palacios's excellent
// explanation of autocompletion that can be found here:
// https://github.com/rodricios/autocomplete
use std::old_io::File;
use std::old_io::BufferedReader;
use std::cmp::{Ord,PartialOrd,Ordering};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

// SimpleWordModel uses a HashMap representation to train on word
// frequency. It is fast for looking up words and incrementing their
// count but since HashMap does not keep track of order, searching the
// keys of a hashmap takes a long time. After the model is trained it
// must be converted to a SimpleWordPredictor which stores the words
// in lexical order and has an index of first letters.
#[derive(Debug)]
pub struct SimpleWordModel(HashMap<String, u32>);

// SimpleWordPredictor uses a constant sized vector of entries indexed by
// first letter. Prediction starts from that index and continues until
// the first letter in the vector changes.
#[derive(Debug)]
pub struct SimpleWordPredictor {
    entries: Vec<SimpleWordEntry>,
    ixs: HashMap<char, u32>,
}

#[derive(Debug)]
pub struct SimpleWordEntry {
    pub word: String,
    pub score: u32,
}

impl Clone for SimpleWordEntry {
    fn clone(&self) -> Self {
        SimpleWordEntry {
            word: self.word.clone(),
            score: self.score
        }
    }
}

// Eq, PartialEq, PartialOrd and Ord are necessary to be able to
// .sort() a vec<SimpleWordEntry>.
impl Eq for SimpleWordEntry { }

impl PartialEq for SimpleWordEntry {
    fn eq(&self, other: &Self) -> bool {
        self.word.eq(&other.word)
    }
}

impl PartialOrd for SimpleWordEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.word.partial_cmp(&other.word)
    }
}

impl Ord for SimpleWordEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.word.cmp(&other.word)
    }
}

impl SimpleWordPredictor {
    // Given an input string, return the top 10 suggestions based on
    // training data.
    pub fn predict(&self, input: &str) -> Vec<SimpleWordEntry> {
        let mut predictions: Vec<SimpleWordEntry> = Vec::new();
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

    // Load training data from a CSV file.
    pub fn from_file(path: &Path) -> SimpleWordPredictor {
        let mut entries = Vec::new();
        let mut file = BufferedReader::new(File::open(path));
        for line_res in file.lines() {
            let line = line_res.unwrap();
            let line = line.trim();
            let str_entry: Vec<&str> = line.split(',').collect();
            let word: String = String::from_str(str_entry[0]);
            let n = str_entry[1].parse().ok().unwrap();
            entries.push(SimpleWordEntry {word: word, score: n});
        }

        let ixs = generate_ixs(&entries);

        SimpleWordPredictor {entries: entries, ixs: ixs}
    }

    // Save training data to a CSV file.
    pub fn to_file(&self, path: &Path) {
        let mut file = File::create(path);
        for entry in &self.entries {
            write!(&mut file, "{},{}\n", &entry.word, entry.score)
                .unwrap();
        }
    }
}

impl SimpleWordModel {
    pub fn from_str(input: &str) -> SimpleWordModel {
        let model_hm: HashMap<String, u32> = HashMap::new();
        let mut model = SimpleWordModel(model_hm);
        let v_input = input.split(' ').collect();
        count_words(&mut model, &v_input);

        model
    }

    pub fn from_vec(input: &Vec<&str>) -> SimpleWordModel {
        let model_hm: HashMap<String, u32> = HashMap::new();
        let mut model = SimpleWordModel(model_hm);

        count_words(&mut model, input);

        model
    }

    pub fn new() -> SimpleWordModel {
        let model_hm: HashMap<String, u32> = HashMap::new();

        SimpleWordModel(model_hm)
    }

    pub fn train_str(&mut self, input: &str) {
        let v_input = input.split(' ').collect();
        count_words(self, &v_input);
    }

    pub fn train_vec(&mut self, input: Vec<&str>) {
        count_words(self, &input);
    }

    // Convert the HashMap representation to an indexed vec.
    pub fn finalize(self) -> SimpleWordPredictor {
        let SimpleWordModel(hm) = self;
        let size = hm.len();
        let mut entries = Vec::with_capacity(size);

        for (key, value) in hm {
            entries.push(SimpleWordEntry {word: key, score: value});
        }

        entries.sort();

        let ixs = generate_ixs(&entries);
        SimpleWordPredictor {entries: entries, ixs: ixs}
    }
}

fn generate_ixs(entries: &Vec<SimpleWordEntry>) -> HashMap<char, u32>{
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

fn count_words(model: &mut SimpleWordModel, input: &Vec<&str>) {
    let &mut SimpleWordModel(ref mut model_hm) = model;
    for word in input {
        if word.len() == 0 {
            continue;
        }
        let string_word = String::from_str(*word);
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
}

#[cfg(test)]
mod tests {
    use SimpleWordModel;

    #[test]
    fn test_from_str() {
        let model = SimpleWordModel::from_str("world domination is my profession hello hello");

        let SimpleWordModel(hash_map) = model;

        assert_eq!(1, *hash_map.get("world").unwrap());
        assert_eq!(2, *hash_map.get("hello").unwrap());
    }

    #[test]
    fn test_from_vec() {
        let model = SimpleWordModel::from_vec(&vec!["rabbit", "rabbit", "hare"]);

        let SimpleWordModel(hash_map) = model;

        assert_eq!(1, *hash_map.get("hare").unwrap());
        assert_eq!(2, *hash_map.get("rabbit").unwrap());
    }

    #[test]
    fn test_train_str() {
        let mut model = SimpleWordModel::new();

        model.train_str("hello hello hello there there");

        let SimpleWordModel(hash_map) = model;
        assert_eq!(3, *hash_map.get("hello").unwrap());
        assert_eq!(2, *hash_map.get("there").unwrap());
    }

    #[test]
    fn test_train_vec() {
        let mut model = SimpleWordModel::new();

        model.train_vec(vec!["hello", "hello", "hello", "what",
                             "is", "this"]);

        let SimpleWordModel(hash_map) = model;
        assert_eq!(3, *hash_map.get("hello").unwrap());
        assert_eq!(1, *hash_map.get("what").unwrap());
    }

    #[test]
    fn test_finalize() {
        let mut model = SimpleWordModel::new();

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
        let mut model = SimpleWordModel::new();

        model.train_str(concat!["anybody can become angry that is easy but to be ",
                                "angry with the right person and to the right degree ",
                                "and at the right time and for the right purpose ",
                                "and in the right way that is not within everybody's ",
                                "power and is not easy"]);

        let predictor = model.finalize();
        println!("{:?}", predictor.predict("a"));
    }

}
