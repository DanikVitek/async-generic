macro_rules! local_assert_snapshot {
    ($value:expr) => {
        insta::with_settings!({prepend_module_to_snapshot => false}, {
            insta::assert_snapshot!($value);
        });
    };
}

pub(crate) use local_assert_snapshot;