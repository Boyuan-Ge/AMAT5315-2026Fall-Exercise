use ising::analysis::{
    autocorrelation_direct, autocorrelation_fft, blocked_standard_error, naive_standard_error,
    parabolic_peak, susceptibility,
};

#[test]
fn susceptibility_uses_signed_second_moment_and_absolute_first_moment() {
    let values = [-1.0, -0.5, 0.5, 1.0];
    let chi = susceptibility(4, 2.0, &values).unwrap();
    assert!((chi - 0.5).abs() < 1e-12);
}

#[test]
fn five_point_parabola_recovers_a_known_vertex() {
    let points: Vec<_> = [2.1_f64, 2.2, 2.3, 2.4, 2.5]
        .into_iter()
        .map(|t| (t, 10.0 - 4.0 * (t - 2.34).powi(2)))
        .collect();
    assert!((parabolic_peak(&points).unwrap() - 2.34).abs() < 1e-10);
}

#[test]
fn fft_autocorrelation_matches_direct_estimator() {
    let values = [1.0, 0.0, 2.0, -1.0, 3.0, 0.5, -0.5, 1.5];
    let direct = autocorrelation_direct(&values).unwrap();
    let fast = autocorrelation_fft(&values).unwrap();
    assert_eq!(direct.len(), fast.len());
    for (left, right) in direct.iter().zip(fast) {
        assert!((left - right).abs() < 1e-10);
    }
}

#[test]
fn blocking_exposes_repeated_sample_correlation() {
    let values: Vec<f64> = (0..50)
        .flat_map(|value| std::iter::repeat_n(value as f64, 20))
        .collect();
    let naive = naive_standard_error(&values).unwrap();
    let blocked = blocked_standard_error(&values, 50).unwrap();
    assert!(blocked / naive > 4.0);
}
