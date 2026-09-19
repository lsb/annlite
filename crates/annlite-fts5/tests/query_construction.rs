//! The MATCH expression decides what is being benchmarked, so it is pinned by test.

use annlite_fts5::query::{match_expression, match_expression_with, TermSplit};

#[test]
fn terms_are_or_ed_so_partial_matches_still_rank() {
    // Handing FTS5 `alpha beta` would mean *both* terms; the baseline wants documents
    // containing either, ranked by BM25.
    assert_eq!(match_expression("alpha beta").unwrap(), "\"alpha\" OR \"beta\"");
    assert_eq!(match_expression("solo").unwrap(), "\"solo\"");
}

#[test]
fn query_operators_in_the_text_are_quoted_into_literals() {
    // A vocabulary word that happens to spell an FTS5 operator must not become one.
    assert_eq!(match_expression("near and or").unwrap(), "\"near\" OR \"and\" OR \"or\"");
    assert_eq!(match_expression("wild*").unwrap(), "\"wild*\"");
    assert_eq!(match_expression("say \"hi\"").unwrap(), "\"say\" OR \"\"\"hi\"\"\"");
}

#[test]
fn whitespace_variation_does_not_change_the_query() {
    assert_eq!(match_expression("  a\tb \n c ").unwrap(), match_expression("a b c").unwrap());
}

#[test]
fn an_empty_query_has_no_match_expression() {
    assert!(match_expression("").is_none());
    assert!(match_expression("   ").is_none());
}

// --- prose queries, as the code corpus asks them -----------------------------------
//
// A docstring is a sentence. Splitting it on whitespace quotes chunks like
// `find_module().`, and a quoted string in FTS5 is a *phrase*, not a literal — so the
// query silently starts demanding that `find` and `module` be adjacent. The choice of
// split is therefore not cosmetic; it changes which documents are candidates at all.

#[test]
fn prose_is_split_on_non_alphanumerics() {
    assert_eq!(
        match_expression_with("Load a module, given information.", TermSplit::Alphanumeric)
            .unwrap(),
        "\"Load\" OR \"a\" OR \"module\" OR \"given\" OR \"information\""
    );
    // Identifiers, dotted calls and markup all break into their alphanumeric runs,
    // which is how the FTS5 `unicode61` tokenizer will have indexed them too, and
    // each run is then free to match on its own rather than as part of a phrase.
    assert_eq!(
        match_expression_with("`find_module()` -> self.x", TermSplit::Alphanumeric).unwrap(),
        "\"find\" OR \"module\" OR \"self\" OR \"x\""
    );
    // Digits are terms; a version number is three of them, not one.
    assert_eq!(
        match_expression_with("since 3.11", TermSplit::Alphanumeric).unwrap(),
        "\"since\" OR \"3\" OR \"11\""
    );
}

#[test]
fn punctuation_only_queries_have_no_match_expression() {
    // Under the whitespace split these are terms, so the two splits must disagree
    // here: a query of pure punctuation has nothing to search for.
    assert!(match_expression_with("--- ***", TermSplit::Alphanumeric).is_none());
    assert!(match_expression_with("--- ***", TermSplit::Whitespace).is_some());
}

#[test]
fn the_word_corpora_are_unaffected_by_the_split() {
    // The generated vocabulary is pure [a-z]+ separated by single spaces, so the two
    // splits coincide there. This is what lets the published word-corpus numbers stay
    // comparable across this change.
    for q in ["proof", "reevaluating abhorring", "a b c"] {
        assert_eq!(
            match_expression_with(q, TermSplit::Alphanumeric),
            match_expression_with(q, TermSplit::Whitespace),
        );
    }
}

#[test]
fn repeated_terms_are_kept() {
    // Deduplicating would change BM25's view of the query and silently diverge from
    // tools/analyze/code_eval.py, which does not deduplicate either.
    assert_eq!(
        match_expression_with("the the", TermSplit::Alphanumeric).unwrap(),
        "\"the\" OR \"the\""
    );
}
