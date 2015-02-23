use std::cmp::Ord;
use std::cmp::PartialOrd;
use std::cmp::Ordering;

/// Struct returned by prediction engines.
#[derive(Debug)]
pub struct PredictionEntry {
    pub word: String,
    pub score: u32,
}

impl Clone for PredictionEntry {
    fn clone(&self) -> Self {
        PredictionEntry {
            word: self.word.clone(),
            score: self.score
        }
    }
}

// Eq, PartialEq, PartialOrd and Ord are necessary to be able to
// .sort() a vec<PredictionEntry>.
impl Eq for PredictionEntry { }

impl PartialEq for PredictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.word.eq(&other.word)
    }
}

impl PartialOrd for PredictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.word.partial_cmp(&other.word)
    }
}

impl Ord for PredictionEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.word.cmp(&other.word)
    }
}
