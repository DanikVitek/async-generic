macro_rules! local_assert_snapshot {
    ($value:expr) => {
        insta::with_settings!({prepend_module_to_snapshot => false}, {
            insta::assert_snapshot!($value);
        });
    };
}

macro_rules! test_expand {
    ($target_fn:expr, $args:expr) => {
        test_expand!($target_fn, $args => formatted);
    };
    ($target_fn:expr, $args:expr => $formatted: ident) => {
        let $formatted = format_expand($target_fn, $args);
        $crate::test_helpers::local_assert_snapshot!($formatted);
    };
}

pub(crate) use local_assert_snapshot;
pub(crate) use test_expand;
