//! The MATCH expression decides what is being benchmarked, so it is pinned by test.

use annlite_fts5::query::match_expression;

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
