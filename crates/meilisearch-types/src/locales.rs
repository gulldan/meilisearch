use milli::{AttributePatterns, LocalizedAttributesRule};

/// Defines a rule for associating specific locales (languages) with
/// attributes. This allows Meilisearch to use language-specific tokenization
/// and processing for matched attributes, improving search quality for
/// multilingual content.
#[routes::request(no_error, setting)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalizedAttributesRuleView {
    /// Patterns to match attribute names. Use `*` as a wildcard to match any
    /// characters. For example, `["title_*", "description"]` matches
    /// `title_en`, `title_fr`, and `description`.
    #[request(required, schema_type = Vec<String>, example = json!(["*_ja"]))]
    pub attribute_patterns: AttributePatterns,
    /// The list of locales (languages) to apply to matching attributes. When
    /// these attributes are indexed, Meilisearch will use language-specific
    /// tokenization rules. Examples: `["en", "fr"]` or `["jpn", "zho"]`.
    #[request(required)]
    pub locales: Vec<Locale>,
}

impl From<LocalizedAttributesRule> for LocalizedAttributesRuleView {
    fn from(rule: LocalizedAttributesRule) -> Self {
        Self {
            attribute_patterns: rule.attribute_patterns,
            locales: rule.locales.into_iter().map(|l| l.into()).collect(),
        }
    }
}

impl From<LocalizedAttributesRuleView> for LocalizedAttributesRule {
    fn from(view: LocalizedAttributesRuleView) -> Self {
        Self {
            attribute_patterns: view.attribute_patterns,
            locales: view.locales.into_iter().map(|l| l.into()).collect(),
        }
    }
}

/// Generate a Locale enum and its From and Into implementations for milli::tokenizer::Language.
///
/// this enum implements `Deserr` in order to be used in the API.
macro_rules! make_locale {
    (
        $(($iso_639_1:ident, $iso_639_1_str:expr) => ($iso_639_3:ident, $iso_639_3_str:expr),)+ ;
        // Languages that have no ISO 639-1 code at all.
        $($only_639_3:ident => $only_639_3_str:expr,)+
    ) => {
        #[routes::request(no_error, setting)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
        pub enum Locale {
            $($iso_639_1,)+
            $($iso_639_3,)+
            $($only_639_3,)+
            Cmn,
        }

        impl From<milli::tokenizer::Language> for Locale {
            fn from(other: milli::tokenizer::Language) -> Locale {
                match other {
                    $(milli::tokenizer::Language::$iso_639_3 => Locale::$iso_639_3,)+
                    $(milli::tokenizer::Language::$only_639_3 => Locale::$only_639_3,)+
                    milli::tokenizer::Language::Cmn => Locale::Cmn,
                }
            }
        }

        impl From<Locale> for milli::tokenizer::Language {
            fn from(other: Locale) -> milli::tokenizer::Language {
                match other {
                    $(Locale::$iso_639_1 => milli::tokenizer::Language::$iso_639_3,)+
                    $(Locale::$iso_639_3 => milli::tokenizer::Language::$iso_639_3,)+
                    $(Locale::$only_639_3 => milli::tokenizer::Language::$only_639_3,)+
                    Locale::Cmn => milli::tokenizer::Language::Cmn,
                }
            }
        }

        impl std::str::FromStr for Locale {
            type Err = LocaleFormatError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let locale = match s {
                    $($iso_639_1_str => Locale::$iso_639_1,)+
                    $($iso_639_3_str => Locale::$iso_639_3,)+
                    $($only_639_3_str => Locale::$only_639_3,)+
                    "cmn" => Locale::Cmn,
                    _ => return Err(LocaleFormatError { invalid_locale: s.to_string() }),
                };

                Ok(locale)
            }
        }

        #[derive(Debug)]
        pub struct LocaleFormatError {
            pub invalid_locale: String,
        }

        impl std::fmt::Display for LocaleFormatError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut valid_locales = [$($iso_639_1_str),+,$($iso_639_3_str),+,$($only_639_3_str),+,"cmn"];
                valid_locales.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
                write!(f, "Unsupported locale `{}`, expected one of {}", self.invalid_locale, valid_locales.join(", "))
            }
        }

        impl std::error::Error for LocaleFormatError {}
    };
}

make_locale!(
    (Af, "af") => (Afr, "afr"),
    (Ak, "ak") => (Aka, "aka"),
    (Am, "am") => (Amh, "amh"),
    (Ar, "ar") => (Ara, "ara"),
    (Az, "az") => (Aze, "aze"),
    (Be, "be") => (Bel, "bel"),
    (Bn, "bn") => (Ben, "ben"),
    (Bg, "bg") => (Bul, "bul"),
    (Ca, "ca") => (Cat, "cat"),
    (Cs, "cs") => (Ces, "ces"),
    (Cy, "cy") => (Cym, "cym"),
    (Da, "da") => (Dan, "dan"),
    (De, "de") => (Deu, "deu"),
    (Cy, "cy") => (Cym, "cym"),
    (El, "el") => (Ell, "ell"),
    (En, "en") => (Eng, "eng"),
    (Eo, "eo") => (Epo, "epo"),
    (Et, "et") => (Est, "est"),
    (Fi, "fi") => (Fin, "fin"),
    (Fr, "fr") => (Fra, "fra"),
    (Gu, "gu") => (Guj, "guj"),
    (He, "he") => (Heb, "heb"),
    (Hi, "hi") => (Hin, "hin"),
    (Hr, "hr") => (Hrv, "hrv"),
    (Hu, "hu") => (Hun, "hun"),
    (Hy, "hy") => (Hye, "hye"),
    (Id, "id") => (Ind, "ind"),
    (It, "it") => (Ita, "ita"),
    (Jv, "jv") => (Jav, "jav"),
    (Ja, "ja") => (Jpn, "jpn"),
    (Kn, "kn") => (Kan, "kan"),
    (Ka, "ka") => (Kat, "kat"),
    (Kk, "kk") => (Kaz, "kaz"),
    (Km, "km") => (Khm, "khm"),
    (Ko, "ko") => (Kor, "kor"),
    (La, "la") => (Lat, "lat"),
    (Lv, "lv") => (Lav, "lav"),
    (Lt, "lt") => (Lit, "lit"),
    (Ml, "ml") => (Mal, "mal"),
    (Mr, "mr") => (Mar, "mar"),
    (Mk, "mk") => (Mkd, "mkd"),
    (My, "my") => (Mya, "mya"),
    (Ne, "ne") => (Nep, "nep"),
    (Nl, "nl") => (Nld, "nld"),
    (Nb, "nb") => (Nob, "nob"),
    (Or, "or") => (Ori, "ori"),
    (Pa, "pa") => (Pan, "pan"),
    (Fa, "fa") => (Pes, "pes"),
    (Pl, "pl") => (Pol, "pol"),
    (Pt, "pt") => (Por, "por"),
    (Ro, "ro") => (Ron, "ron"),
    (Ru, "ru") => (Rus, "rus"),
    (Si, "si") => (Sin, "sin"),
    (Sk, "sk") => (Slk, "slk"),
    (Sl, "sl") => (Slv, "slv"),
    (Sn, "sn") => (Sna, "sna"),
    (Es, "es") => (Spa, "spa"),
    (Sr, "sr") => (Srp, "srp"),
    (Sv, "sv") => (Swe, "swe"),
    (Ta, "ta") => (Tam, "tam"),
    (Te, "te") => (Tel, "tel"),
    (Tl, "tl") => (Tgl, "tgl"),
    (Th, "th") => (Tha, "tha"),
    (Tk, "tk") => (Tuk, "tuk"),
    (Tr, "tr") => (Tur, "tur"),
    (Uk, "uk") => (Ukr, "ukr"),
    (Ur, "ur") => (Urd, "urd"),
    (Uz, "uz") => (Uzb, "uzb"),
    (Vi, "vi") => (Vie, "vie"),
    (Yi, "yi") => (Yid, "yid"),
    (Zh, "zh") => (Zho, "zho"),
    (Zu, "zu") => (Zul, "zul"),
    (Ab, "ab") => (Abk, "abk"),
    (Bm, "bm") => (Bam, "bam"),
    (Br, "br") => (Bre, "bre"),
    (Cu, "cu") => (Chu, "chu"),
    (Eu, "eu") => (Eus, "eus"),
    (Fo, "fo") => (Fao, "fao"),
    (Gd, "gd") => (Gla, "gla"),
    (Ga, "ga") => (Gle, "gle"),
    (Gl, "gl") => (Glg, "glg"),
    (Gv, "gv") => (Glv, "glv"),
    (Ht, "ht") => (Hat, "hat"),
    (Ha, "ha") => (Hau, "hau"),
    (Is, "is") => (Isl, "isl"),
    (Ky, "ky") => (Kir, "kir"),
    (No, "no") => (Nor, "nor"),
    (Oc, "oc") => (Oci, "oci"),
    (Sa, "sa") => (San, "san"),
    (Se, "se") => (Sme, "sme"),
    (Sd, "sd") => (Snd, "snd"),
    (Sq, "sq") => (Sqi, "sqi"),
    (Tt, "tt") => (Tat, "tat"),
    (Ug, "ug") => (Uig, "uig"),
    (Wo, "wo") => (Wol, "wol"),
    (Yo, "yo") => (Yor, "yor"),
    ;
    Abq => "abq",
    Aii => "aii",
    Ajp => "ajp",
    Akk => "akk",
    Aln => "aln",
    Apu => "apu",
    Aqz => "aqz",
    Arb => "arb",
    Arh => "arh",
    Arr => "arr",
    Axm => "axm",
    Azz => "azz",
    Bho => "bho",
    Bor => "bor",
    Brh => "brh",
    Bxr => "bxr",
    Ceb => "ceb",
    Cop => "cop",
    Cpg => "cpg",
    Ctn => "ctn",
    Egy => "egy",
    Eme => "eme",
    Ess => "ess",
    Frm => "frm",
    Fro => "fro",
    Got => "got",
    Grc => "grc",
    Gub => "gub",
    Gun => "gun",
    Gwi => "gwi",
    Gya => "gya",
    Hbo => "hbo",
    Hit => "hit",
    Hsb => "hsb",
    Hyw => "hyw",
    Kbc => "kbc",
    Kmr => "kmr",
    Koi => "koi",
    Kpv => "kpv",
    Krl => "krl",
    Lij => "lij",
    Lzh => "lzh",
    Mdf => "mdf",
    Myu => "myu",
    Myv => "myv",
    Naq => "naq",
    Nds => "nds",
    Nhi => "nhi",
    Nmf => "nmf",
    Oge => "oge",
    Olo => "olo",
    Orv => "orv",
    Ota => "ota",
    Pay => "pay",
    Pcm => "pcm",
    Pro => "pro",
    Pst => "pst",
    Qaf => "qaf",
    Qpm => "qpm",
    Qtd => "qtd",
    Qti => "qti",
    Quc => "quc",
    Ruc => "ruc",
    Sab => "sab",
    Sah => "sah",
    Say => "say",
    Scn => "scn",
    Sga => "sga",
    Sjo => "sjo",
    Sms => "sms",
    Ssp => "ssp",
    Tpn => "tpn",
    Urb => "urb",
    Vep => "vep",
    Wbp => "wbp",
    Wuu => "wuu",
    Xav => "xav",
    Xcl => "xcl",
    Xnr => "xnr",
    Xpg => "xpg",
    Xum => "xum",
    Yrk => "yrk",
    Yrl => "yrl",
    Yue => "yue",
    Zza => "zza",
);
