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
//! When more than one is named, the dictionaries themselves say which one the
//! word belongs to — see [`Lemmatizer::resolve`]. Statistical detection cannot:
//! it knows a fraction of the languages a bundle covers, and a single word is
//! not enough text for it anyway.
//!
//! An index and the dictionaries that filled it are one pair: it stores lemmas,
//! not the words the documents spell, so a query lemmatized by another bundle
//! asks for something the index never wrote. Nothing fails when that happens —
//! the documents are there, the tasks succeeded, the words are simply not
//! found. So every index records the [`Generations`] it was built with, and
//! [`check_index`] says out loud when they are not the ones loaded.
//!
//! Пишется при этом не весь бандл, а только языки, словари которых на
//! самом деле применились: чем индекс не лемматизировали, то его и не касается.
//! О применённых говорит [`Recording`] — окно, которое индексатор держит открытым
//! на время своего прогона.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::{fmt, fs};

use charabia::normalizer::Lemmatizer as LemmatizerTrait;
use charabia::{Language, Token};
use serde::Deserialize;
use udlex_rs::{catalog, Error, Lexicon, Options, Source};
use uuid::Uuid;

use crate::ThreadPoolNoAbort;

static LEMMATIZER: OnceLock<Lemmatizer> = OnceLock::new();

/// Indexes already compared against the loaded dictionaries, so that a
/// mismatch is reported once per index instead of once per request.
///
/// Ключ — пара «что лежит на диске» и «как его сейчас зовут»: `swap-indexes`
/// меняет под именем содержимое, а переименование — имя под содержимым;
/// по одному uid проверка после такого обмена больше не повторилась бы.
static CHECKED: Mutex<BTreeSet<(Uuid, String)>> = Mutex::new(BTreeSet::new());

/// How many language codes a mismatch lists before it sums the rest up: a
/// bundle carries dozens of them, and what matters is that they disagree.
const LISTED_CODES: usize = 8;

/// The generation of every dictionary of a bundle, keyed by ISO 639-3 code.
///
/// udlex derives a generation from everything that went into a dictionary, so
/// two bundles lemmatize alike exactly when their generations agree. That
/// makes it the one thing worth writing next to an index built from them.
pub type Generations = BTreeMap<String, String>;

/// The dictionaries of every language found in a bundle directory.
pub struct Lemmatizer {
    lexicons: HashMap<Language, Lexicon>,
    /// Какой ISO 639-3 код у языка charabia: лемматизация отчитывается
    /// языками, а штамп хранит коды.
    codes: HashMap<Language, String>,
    generations: Generations,
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
        let mut codes = HashMap::new();
        let mut generations = Generations::new();
        for (code, path) in catalog(directory)? {
            let Some(language) = Language::from_code(&code) else {
                tracing::warn!("lemmatizer: no charabia language for {code}, dictionary skipped");
                continue;
            };
            match generation(&path) {
                Some(stamp) => {
                    generations.insert(code.clone(), stamp);
                }
                // Not fatal: the dictionary still lemmatizes, it just cannot be
                // told apart from another build of the same language.
                None => tracing::warn!("lemmatizer: dictionary {code} names no generation"),
            }
            codes.insert(language, code);
            lexicons.insert(language, Lexicon::open(path)?);
        }
        Ok(Self { lexicons, codes, generations })
    }

    /// The languages this lemmatizer answers for.
    pub fn languages(&self) -> impl Iterator<Item = Language> + '_ {
        self.lexicons.keys().copied()
    }

    /// The generation of every dictionary it holds.
    pub fn generations(&self) -> &Generations {
        &self.generations
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
        let language = language?;
        let lexicon = self.lexicons.get(&language)?;
        // Словарь взялся за слово — значит, если сейчас идёт индексация,
        // в индекс ляжет то, что сказал именно этот словарь. Вне окна
        // записи вызов — одна проверка локальной переменной потока.
        note(language);
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

    /// Спрашивает словари кандидатов, чей это язык.
    ///
    /// Сначала — у кого слово записано. Форма, которую словарь взял из корпуса
    /// своего языка, весит больше взятой из подключённого к нему внешнего
    /// справочника: справочники разных языков пересекаются заимствованиями,
    /// именами и просто совпадениями написания, и почти весь ложный выбор
    /// приходится на них. При равном весе побеждает наименьший язык по порядку
    /// [`Language`]. Правила по суффиксу на этом круге выключены: они отвечают
    /// на любое слово любого языка и спор не разрешили бы, а стёрли.
    ///
    /// Если слова не знает никто — у кого на него хотя бы срабатывает правило.
    /// Это единственный оставшийся способ что-то о слове сказать, и сказать
    /// его должен словарь: [`whatlang`] читает целый кусок текста, а при
    /// индексации это документ, тогда как на запросе — сам запрос, так что
    /// его ответ на двух сторонах разный по построению. Ничьё правило не
    /// сработало — вернуть нечего, и слово останется таким, как написано,
    /// каким бы языком оно ни было помечено.
    ///
    /// Отсюда и «наименьший», а не «первый в списке»: при индексации список —
    /// это локали поля, на запросе — объединение локалей индекса, и совпадают
    /// они по составу, а не по порядку. Ответ зависит только от слова и от
    /// набора языков — ровно настолько, чтобы слово легло в индекс и искалось
    /// одной и той же леммой.
    ///
    /// Кандидат, которому уже нечего выиграть, не спрашивается вовсе, так что
    /// упорядоченный список — а Meilisearch отдаёт именно такой — обычно стоит
    /// одного поиска по словарю, а не одного на язык.
    ///
    /// [`whatlang`]: https://docs.rs/whatlang
    fn resolve(
        &self,
        word: &str,
        candidates: &[Language],
        sentence_initial: bool,
    ) -> Option<Language> {
        let stored = Options { use_rules: false, ..Options::default() };
        let mut chosen: Option<(Weight, Language)> = None;
        for &language in candidates {
            if chosen.is_some_and(|(weight, chosen)| weight == Weight::Corpus && chosen <= language)
            {
                continue;
            }
            let Some(lexicon) = self.lexicons.get(&language) else { continue };
            let answer = lexicon.lemma_with_info(word, sentence_initial, stored);
            if !answer.source.is_known() {
                continue;
            }
            let candidate = (Weight::of(answer.source), language);
            if chosen.is_none_or(|chosen| candidate < chosen) {
                chosen = Some(candidate);
            }
        }
        if let Some((_, language)) = chosen {
            return Some(language);
        }

        let mut guessed: Option<Language> = None;
        for &language in candidates {
            if guessed.is_some_and(|guessed| guessed <= language) {
                continue;
            }
            let Some(lexicon) = self.lexicons.get(&language) else { continue };
            if lexicon.lemma(word, sentence_initial) != word {
                guessed = Some(language);
            }
        }
        guessed
    }
}

/// Насколько веско словарь знает слово: из корпуса своего языка или из
/// подключённого к нему внешнего справочника.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Weight {
    Corpus,
    External,
}

impl Weight {
    fn of(source: Source) -> Self {
        match source {
            Source::External | Source::ExternalLower => Self::External,
            _ => Self::Corpus,
        }
    }
}

/// Формы слова, которые ложатся в индекс: набранная и, когда словарь её
/// изменил, лемма. Отвечает обоим индексаторам, новому и легаси, — индекс
/// обязан выйти один и тот же, каким путём его ни строй.
///
/// Индекс Meilisearch устроен вокруг «в индексе то, что написано»: по набранной
/// форме работают набор по буквам, точное совпадение, исключение слова и
/// порядок выдачи. Лемма кладётся сверху и на ту же позицию.
///
/// Каждая форма проверяется отдельно: словарь может отдать лемму длиннее ключа
/// LMDB, и это не повод терять написанное.
pub fn indexed_forms<'t>(token: &'t Token<'_>) -> Option<(&'t str, Option<&'t str>)> {
    let indexable = |word: &'t str| {
        let word = word.trim();
        (!word.is_empty() && word.len() <= crate::MAX_WORD_LENGTH).then_some(word)
    };
    match (token.surface().and_then(indexable), indexable(token.lemma())) {
        (Some(surface), lemma) => Some((surface, lemma)),
        (None, lemma) => lemma.map(|lemma| (lemma, None)),
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

/// What the dictionaries of this process are, empty when there are none.
///
/// Это то, с чем сверяется штамп, а не то, что в него пишут: индексу
/// принадлежат только языки, которыми его лемматизировали.
pub fn generations() -> Generations {
    get().map_or_else(Generations::new, |lemmatizer| lemmatizer.generations.clone())
}

/// Поколения словарей перечисленных языков — ровно то, что уложило слова.
///
/// Язык без словаря и словарь без поколения выпадают: назвать их нечем, а
/// придумывать имя значило бы поднять тревогу на пустом месте.
pub fn generations_of(languages: &BTreeSet<Language>) -> Generations {
    let Some(lemmatizer) = get() else { return Generations::new() };
    languages
        .iter()
        .filter_map(|language| lemmatizer.codes.get(language))
        .filter_map(|code| {
            lemmatizer.generations.get(code).map(|stamp| (code.clone(), stamp.clone()))
        })
        .collect()
}

/// Словари, применённые за один прогон индексации.
///
/// Поток пула копит наблюдения у себя и отдаёт их сюда, закрывая окно: у
/// одного документа языков единицы, а замок на каждое слово стоил бы дороже
/// самого похода в словарь.
#[derive(Debug, Default)]
struct Applied(Mutex<BTreeSet<Language>>);

thread_local! {
    /// Куда этот поток складывает применённые словари. Пусто у всех, кроме
    /// позванных в окно: те же словари применяет и поиск, но он говорит о
    /// запросе, а не о том, чем уложен индекс.
    static RECORDING: RefCell<Option<(Arc<Applied>, BTreeSet<Language>)>> =
        const { RefCell::new(None) };
}

/// Отмечает, что словарь этого языка применился на этом потоке.
fn note(language: Language) {
    RECORDING.with_borrow_mut(|slot| {
        if let Some((_, seen)) = slot.as_mut() {
            seen.insert(language);
        }
    });
}

/// Окно, за время которого прогон индексации узнаёт, чем он лемматизировал.
///
/// Открывается на всех потоках пула индексации и только на них: токенизация
/// документов целиком идёт через `pool.install`, а поиск живёт на потоках
/// HTTP и в окно не попадает — иначе чужой запрос дописывал бы индексу языки,
/// которых в нём нет.
///
/// Пул на процесс один, и батчи он обрабатывает по одному, так что два окна
/// одновременно не открываются.
pub struct Recording<'pool> {
    /// `None`, когда окно уже закрыто или открывать его не для чего.
    pool: Option<&'pool ThreadPoolNoAbort>,
    applied: Arc<Applied>,
}

impl<'pool> Recording<'pool> {
    /// Открывает окно. Без словарей записывать нечего, и пул не тревожат.
    pub fn open(pool: &'pool ThreadPoolNoAbort) -> Self {
        let applied = Arc::<Applied>::default();
        if get().is_none() {
            return Self { pool: None, applied };
        }
        // Ошибку рассылки глотаем: паника в пуле всплывёт там, где её ловит
        // сама индексация, а штамп из-за неё терять незачем.
        let _ = pool.broadcast(|_| {
            RECORDING.with_borrow_mut(|slot| *slot = Some((applied.clone(), BTreeSet::new())));
        });
        Self { pool: Some(pool), applied }
    }

    /// Языки, чьи словари в этом окне применились. Закрывает окно.
    pub fn languages(&mut self) -> BTreeSet<Language> {
        self.close();
        std::mem::take(&mut self.applied.0.lock().unwrap())
    }

    /// Снимает запись с потоков пула, забирая накопленное. Повторный вызов
    /// ничего не делает — тем и годится для [`Drop`].
    fn close(&mut self) {
        let Some(pool) = self.pool.take() else { return };
        let _ = pool.broadcast(|_| {
            RECORDING.with_borrow_mut(|slot| {
                if let Some((applied, seen)) = slot.take() {
                    applied.0.lock().unwrap().extend(seen);
                }
            });
        });
    }
}

impl Drop for Recording<'_> {
    /// Прогон, оборвавшийся на ошибке, не оставляет пул размеченным.
    fn drop(&mut self) {
        self.close();
    }
}

/// Warns when an index was filled by other dictionaries than the loaded ones,
/// once per index for the lifetime of the process.
///
/// `recorded` is only called on the first check of an index, which keeps this
/// affordable on the path of every index access.
///
/// An index last written before generations were recorded holds none, and
/// there is nothing to compare it against: stay quiet rather than accuse every
/// pre-existing database at every start-up. The next indexing stamps it.
///
/// Пара `uid` и `uuid` — это и есть проверяемое: имя, под которым индекс
/// отвечает, вместе с данными, которые под этим именем лежат. `swap-indexes`
/// меняет второе, не трогая первого, и проверка обязана пройти заново.
pub fn check_index(
    uid: &str,
    uuid: Uuid,
    recorded: impl FnOnce() -> crate::Result<Option<Generations>>,
) -> crate::Result<()> {
    if !CHECKED.lock().unwrap().insert((uuid, uid.to_owned())) {
        return Ok(());
    }
    let Some(recorded) = recorded()? else { return Ok(()) };
    let loaded = generations();
    let mismatch = Mismatch::between(&recorded, &loaded);
    if mismatch.is_empty() {
        return Ok(());
    }
    tracing::warn!(
        "lemmatizer: index {uid:?} was filled by other dictionaries than the ones loaded now \
         ({}); it stays searchable, but words stored as lemmas may not be found until it is \
         reindexed — /indexes/{uid}/stats reports what filled it",
        mismatch
    );
    Ok(())
}

/// How the dictionaries that filled an index differ from the loaded ones.
///
/// Сравнение идёт по записям индекса и только по ним. Язык, которого в бандле
/// прибавилось, ни одного слова этого индекса не трогал: у бандла своя жизнь,
/// и предъявлять её индексу не за что.
struct Mismatch<'a> {
    /// Filled the index, absent from this process.
    missing: Vec<&'a str>,
    /// Loaded, but not the generation the index was filled by.
    changed: Vec<&'a str>,
}

impl<'a> Mismatch<'a> {
    fn between(recorded: &'a Generations, loaded: &'a Generations) -> Self {
        let mut mismatch = Self { missing: Vec::new(), changed: Vec::new() };
        for (code, generation) in recorded {
            match loaded.get(code) {
                None => mismatch.missing.push(code),
                Some(other) if other != generation => mismatch.changed.push(code),
                Some(_) => (),
            }
        }
        mismatch
    }

    /// Ни одного расхождения — молчать.
    fn is_empty(&self) -> bool {
        self.missing.is_empty() && self.changed.is_empty()
    }
}

impl fmt::Display for Mismatch<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for (label, codes) in [("missing", &self.missing), ("changed", &self.changed)] {
            if codes.is_empty() {
                continue;
            }
            let listed = &codes[..codes.len().min(LISTED_CODES)];
            write!(formatter, "{separator}{label}: {}", listed.join(", "))?;
            if let Some(rest) = codes.len().checked_sub(LISTED_CODES).filter(|rest| *rest > 0) {
                write!(formatter, " and {rest} more")?;
            }
            separator = "; ";
        }
        Ok(())
    }
}

/// What udlex stamped the dictionary in `directory` with.
///
/// A bundle that keeps immutable generations side by side names the live one
/// in `current`; a plain directory carries it in its own metadata.
fn generation(directory: &Path) -> Option<String> {
    if let Ok(current) = fs::read_to_string(directory.join("current")) {
        return Some(current.trim().to_owned());
    }
    let metadata = fs::read_to_string(directory.join("meta.json")).ok()?;
    serde_json::from_str::<Metadata>(&metadata).ok()?.generation
}

#[derive(Deserialize)]
struct Metadata {
    generation: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    fn token(lemma: &str, surface: Option<&str>) -> Token<'static> {
        Token {
            lemma: Cow::Owned(lemma.to_owned()),
            surface: surface.map(|surface| Cow::Owned(surface.to_owned())),
            ..Default::default()
        }
    }

    #[test]
    fn a_word_the_dictionary_left_alone_goes_in_once() {
        assert_eq!(indexed_forms(&token("рама", None)), Some(("рама", None)));
    }

    #[test]
    fn a_lemmatized_word_goes_in_as_written_and_as_lemma() {
        assert_eq!(indexed_forms(&token("мыть", Some("мыла"))), Some(("мыла", Some("мыть"))));
    }

    #[test]
    fn a_form_that_does_not_fit_the_key_is_dropped_alone() {
        let long = "я".repeat(crate::MAX_WORD_LENGTH);
        // Не влезла лемма — набранное всё равно ложится.
        assert_eq!(indexed_forms(&token(&long, Some("мыла"))), Some(("мыла", None)));
        // Не влезло набранное — его место занимает лемма, а не пустота.
        assert_eq!(indexed_forms(&token("мыть", Some(&long))), Some(("мыть", None)));
        // Не влезло ничего — слова нет.
        assert_eq!(indexed_forms(&token(&long, Some(&long))), None);
    }

    #[test]
    fn an_empty_word_is_not_stored() {
        assert_eq!(indexed_forms(&token("   ", None)), None);
        assert_eq!(indexed_forms(&token("мыть", Some("  "))), Some(("мыть", None)));
    }

    fn generations(pairs: &[(&str, &str)]) -> Generations {
        pairs
            .iter()
            .map(|(code, generation)| ((*code).to_owned(), (*generation).to_owned()))
            .collect()
    }

    #[test]
    fn mismatch_names_every_way_two_bundles_can_disagree() {
        let recorded = generations(&[("rus", "g1"), ("fin", "g1"), ("deu", "g1")]);
        let loaded = generations(&[("rus", "g1"), ("fin", "g2"), ("spa", "g1")]);
        assert_eq!(
            Mismatch::between(&recorded, &loaded).to_string(),
            "missing: deu; changed: fin"
        );
    }

    #[test]
    fn a_bundle_that_only_grew_is_no_mismatch() {
        let recorded = generations(&[("rus", "g1")]);
        let loaded = generations(&[("rus", "g1"), ("bre", "g1"), ("fin", "g1")]);
        assert!(Mismatch::between(&recorded, &loaded).is_empty());
    }

    #[test]
    fn a_bundle_that_vanished_is_reported_whole() {
        let recorded = generations(&[("rus", "g1"), ("fin", "g1")]);
        assert_eq!(
            Mismatch::between(&recorded, &Generations::new()).to_string(),
            "missing: fin, rus"
        );
    }

    #[test]
    fn long_lists_are_summed_up_instead_of_printed() {
        let codes: Vec<_> =
            (0..12).map(|index| (format!("l{index:02}"), "g1".to_owned())).collect();
        let recorded: Generations = codes.into_iter().collect();
        assert_eq!(
            Mismatch::between(&recorded, &Generations::new()).to_string(),
            "missing: l00, l01, l02, l03, l04, l05, l06, l07 and 4 more"
        );
    }
}
