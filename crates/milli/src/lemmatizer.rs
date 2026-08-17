//! Lemmatization backed by udlex dictionaries.
//!
//! Dictionaries are a process resource: they are memory-mapped once and live
//! until the process ends, so they sit behind a `OnceLock` the binary fills at
//! start-up. Whether they are *applied* is decided per tokenizer: only the
//! document, query and highlighting pipelines pass the lemmatizer along, which
//! leaves facet values untouched.
//!
//! A word is only lemmatized when its language is known, and for most scripts
//! that means `localizedAttributes` on the index and `locales` on the search.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use charabia::normalizer::Lemmatizer as LemmatizerTrait;
use charabia::Language;
use udlex_rs::{catalog, Error, Lexicon};

static LEMMATIZER: OnceLock<Lemmatizer> = OnceLock::new();

/// The dictionaries of every language found in a bundle directory.
pub struct Lemmatizer {
    lexicons: HashMap<Language, Lexicon>,
}

impl Lemmatizer {
    /// Opens every dictionary of a udlex bundle, skipping languages charabia
    /// has no identifier for.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be listed or a dictionary is
    /// unreadable.
    pub fn open(directory: &Path) -> Result<Self, Error> {
        let mut lexicons = HashMap::new();
        for (code, path) in catalog(directory)? {
            let Some(language) = Language::from_code(&code) else {
                tracing::warn!("lemmatizer: no charabia language for {code}, dictionary skipped");
                continue;
            };
            lexicons.insert(language, Lexicon::open(path)?);
        }
        Ok(Self { lexicons })
    }

    /// The languages this lemmatizer answers for.
    pub fn languages(&self) -> impl Iterator<Item = Language> + '_ {
        self.lexicons.keys().copied()
    }
}

impl fmt::Debug for Lemmatizer {
    /// The lexicons themselves are memory-mapped tries; only their count is
    /// worth printing.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Lemmatizer").field("languages", &self.lexicons.len()).finish()
    }
}

impl LemmatizerTrait for Lemmatizer {
    fn lemma<'o>(
        &self,
        word: &'o str,
        language: Option<Language>,
        sentence_initial: bool,
    ) -> Option<Cow<'o, str>> {
        let lexicon = self.lexicons.get(&language?)?;
        // The lexicon lends its strings for as long as it is borrowed, which is
        // shorter than the token it answers about, so a changed word is handed
        // over owned. An unchanged one keeps borrowing the token itself.
        let lemma = lexicon.lemma(word, sentence_initial);
        Some(if lemma.as_ref() == word {
            Cow::Borrowed(word)
        } else {
            Cow::Owned(lemma.into_owned())
        })
    }
}

/// Installs the dictionaries for the whole process. Later calls are ignored.
pub fn configure(lemmatizer: Lemmatizer) {
    let _ = LEMMATIZER.set(lemmatizer);
}

/// The dictionaries of this process, if any were installed.
pub fn get() -> Option<&'static Lemmatizer> {
    LEMMATIZER.get()
}
