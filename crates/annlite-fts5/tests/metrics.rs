//! Hand-worked checks of the ranking metrics.
//!
//! These definitions are the yardstick the ANN indexes will be held to, so they are
//! checked against numbers computed by hand rather than against the implementation's
//! own output.

use annlite_fts5::metrics::{known_item_quality, percentile, Dist};

#[test]
fn success_and_mrr_match_a_worked_example() {
    // Five known-item queries whose gold document came back at ranks 1, 2, 11, 100 and
    // not at all.
    let ranks = vec![Some(1), Some(2), Some(11), Some(100), None];
    let q = known_item_quality(&ranks);

    // success@1: only the first query. @10: ranks 1 and 2. @100: those plus 11 and 100.
    assert_eq!(q.success_at, vec![(1, 1.0 / 5.0), (10, 2.0 / 5.0), (100, 4.0 / 5.0)]);

    // MRR@1  = (1/1)                       / 5 = 0.2
    // MRR@10 = (1/1 + 1/2)                 / 5 = 0.3
    // MRR@100= (1/1 + 1/2 + 1/11 + 1/100)  / 5
    let mrr100 = (1.0 + 0.5 + 1.0 / 11.0 + 0.01) / 5.0;
    assert!((q.mrr_at[0].1 - 0.2).abs() < 1e-12);
    assert!((q.mrr_at[1].1 - 0.3).abs() < 1e-12);
    assert!((q.mrr_at[2].1 - mrr100).abs() < 1e-12);

    // The unretrieved query stays in the denominator; dropping it would turn 4/5 into
    // 4/4 and flatter the system.
    assert_eq!(q.n_unretrieved, 1);
    assert_eq!(q.n_queries, 5);
}

#[test]
fn a_perfect_run_scores_one_everywhere() {
    let q = known_item_quality(&vec![Some(1); 7]);
    for (_, v) in q.success_at.iter().chain(q.mrr_at.iter()) {
        assert_eq!(*v, 1.0);
    }
}

#[test]
fn percentiles_are_nearest_rank_not_interpolated() {
    let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    assert_eq!(percentile(&v, 0.50), 50.0);
    assert_eq!(percentile(&v, 0.95), 95.0);
    assert_eq!(percentile(&v, 0.99), 99.0);
    // Every reported percentile is a value that was actually observed.
    let d = Dist::of(&[3.0, 1.0, 2.0]);
    assert_eq!((d.min, d.median, d.max), (1.0, 2.0, 3.0));
    assert_eq!(d.p95, 3.0);
    assert!((d.mean - 2.0).abs() < 1e-12);
}

#[test]
fn empty_samples_do_not_panic() {
    let d = Dist::of(&[]);
    assert_eq!(d.n, 0);
    assert!(percentile(&[], 0.5).is_nan());
}
