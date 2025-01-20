use async_trait::async_trait;
use async_generic::async_generic;

#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("src/tests/pass/fn-with-type-params.rs");
    t.pass("src/tests/pass/fn-with-types.rs");
    t.pass("src/tests/pass/generic-fn.rs");
    t.pass("src/tests/pass/generic-fn-with-visibility.rs");
    t.pass("src/tests/pass/struct-method-generic.rs");
    t.pass("src/tests/pass/trait-fn.rs");
    // TODO: test that macro does not mess with the internal declarations of
    // functions

    t.compile_fail("src/tests/fail/misuse-of-underscore-async.rs");
    t.compile_fail("src/tests/fail/no-async-fn.rs");
    t.compile_fail("src/tests/fail/no-macro-args.rs");
    t.compile_fail("src/tests/fail/no-struct.rs");
}

#[async_generic(async_trait)]
trait NoDefaultImpl {
    #[async_generic(async_signature(&self) -> String where Self: Send + Sync)]
    fn do_stuff(&self) -> String;
}
