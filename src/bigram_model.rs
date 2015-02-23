use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Bigram prediction trainer.
pub struct BigramTrainer {
    // The collection of all position 1 words in the bigram.
    outer_map: HashMap<String, InnerMap>,

    // For creating a bigram from the first word of a new .train()
    // call.
    prev_word: Option<String>,
}

struct InnerMap(HashMap<String, u32>);

impl BigramTrainer {
    /// Creates a new, empty BigramTrainer.
    pub fn new() -> BigramTrainer {
        let model_hm = HashMap::new();

        BigramTrainer {outer_map: model_hm, prev_word: None}
    }

    /// Train the model on a vector of individual words.
    pub fn train_vec(&mut self, input: Vec<&str>) {
        count_words(self, &input);
    }

    /// Train the model on a str that will be split in to words.
    pub fn train_str(&mut self, input: &str) {
        let v_input = input.split(' ').collect();
        count_words(self, &v_input);
    }
}

// Check if a word exists in the inner map. Add it if it does not,
// increment it if it does.
fn add_or_increment_inner(inner: &mut InnerMap, word: String) {
    let &mut InnerMap(ref mut hm) = inner;
    let entry = hm.entry(word);

    match entry {
        Entry::Vacant(vacant_entry) => {
            vacant_entry.insert(1);
        },
        Entry::Occupied(mut occ_entry) => {
            let count = occ_entry.get_mut();
            *count += 1;
        }
    }
}

fn count_words(trainer: &mut BigramTrainer, input: &Vec<&str>) {
    let mut word_iter = input.iter();
    let ref mut model = trainer.outer_map;

    let mut last_word = match trainer.prev_word {
        None => {
            word_iter.next();
            String::from_str(input[0])
        },
        Some(ref word) => word.clone(),
    };

    for word in word_iter {
        let word = String::from_str(word);
        let outer_entry = model.entry(last_word.clone());

        match outer_entry {
            Entry::Vacant(vacant_entry) => {
                let mut inner_hm = HashMap::new();
                inner_hm.insert(word.clone(), 1);
                vacant_entry.insert(InnerMap(inner_hm));
            },
            Entry::Occupied(mut occ_entry) => {
                let inner_hm = occ_entry.get_mut();
                add_or_increment_inner(inner_hm, word.clone());
            }
        }

        last_word = word.clone();
    }
    trainer.prev_word = Some(last_word);
}

#[cfg(test)]
mod tests {
    use bigram_model::BigramTrainer;
    use bigram_model::InnerMap;

    fn get_score(trainer: &BigramTrainer, word1: &str, word2: &str) -> u32 {
        let &InnerMap(ref inner) = trainer.outer_map.get(word1).unwrap();

        inner.get(word2).unwrap().clone()
    }

    #[test]
    fn test_from_str() {
        let mut model = BigramTrainer::new();

        model.train_str("there once was a man from dundee");
        let prev_word = model.prev_word.clone();
        assert_eq!("dundee", prev_word.unwrap());

        model.train_str("joe was a happy man");
        let prev_word = model.prev_word.clone();
        assert_eq!("man", prev_word.unwrap());

        model.train_str("happy happy happy");

        assert_eq!(1, get_score(&model, "there", "once"));
        assert_eq!(1, get_score(&model, "dundee", "joe"));
        assert_eq!(2, get_score(&model, "happy", "happy"));
    }
}
