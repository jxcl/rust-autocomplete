use simplemodel::SimpleWordTrainer;

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Bigram prediction trainer.
pub struct BigramTrainer {
    // The collection of all position 1 words in the bigram.
    outer_map: HashMap<String, SimpleWordTrainer>,

    // For creating a bigram from the first word of a new .train()
    // call.
    prev_word: Option<String>,
}

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
                let mut inner_trainer = SimpleWordTrainer::new();
                inner_trainer.train_word(word.as_slice());
                vacant_entry.insert(inner_trainer);
            },
            Entry::Occupied(mut occ_entry) => {
                let inner_trainer = occ_entry.get_mut();
                inner_trainer.train_word(word.as_slice());
            }
        }

        last_word = word.clone();
    }
    trainer.prev_word = Some(last_word);
}

#[cfg(test)]
mod tests {
    use bigram_model::BigramTrainer;

    fn get_score(trainer: &BigramTrainer, word1: &str, word2: &str) -> u32 {
        let inner_trainer  = trainer.outer_map.get(word1).unwrap();

        inner_trainer.debug_get_word_score(word2).unwrap().clone()
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
