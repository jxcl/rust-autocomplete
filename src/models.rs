use std::collections::HashMap;
use std::collections::hash_map::Entry;

pub struct SimpleWordModel(HashMap<String, u32>);
pub struct Prediction {
    word: String,
    score: u32,
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

    pub fn suggest(&self, letters: &str) -> Vec<Prediction> {
        let mut predictions: Vec<Prediction> = Vec::new();
        let &SimpleWordModel(ref model_hm) = self;

        for (key, value) in model_hm {
            if key.starts_with(letters) {
                predictions.push(Prediction {word: key.clone(), score: *value});
            }
        }

        predictions
    }
}

fn count_words(model: &mut SimpleWordModel, input: &Vec<&str>) {
    let &mut SimpleWordModel(ref mut model_hm) = model;
    for word in input {
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
    use models::SimpleWordModel;

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
    fn test_train() {
        let mut model = SimpleWordModel::new();

        model.train_str("hello hello hello there there");

        let SimpleWordModel(hash_map) = model;
        assert_eq!(3, *hash_map.get("hello").unwrap());
        assert_eq!(2, *hash_map.get("there").unwrap());
    }

}
