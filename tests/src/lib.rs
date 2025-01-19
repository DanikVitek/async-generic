use std::io::{Read, Result};
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
    t.compile_fail("src/tests/fail/no-impl.rs");
    t.compile_fail("src/tests/fail/no-macro-args.rs");
    t.compile_fail("src/tests/fail/no-struct.rs");
    t.compile_fail("src/tests/fail/no-trait.rs");
}

#[async_generic(async_signature<R: AsyncRead>(reader: &mut R) -> Result<u8>)]
fn do_stuff<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    if _async {
        reader.read_exact(&mut buf).await?;
    } else {
        reader.read_exact(&mut buf)?;
    }
    Ok(buf[0])
}

#[async_trait]
trait AsyncRead: Unpin + Send {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    async fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
}

#[async_trait]
impl<R: async_std::io::ReadExt + Unpin + Send> AsyncRead for R {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        async_std::io::ReadExt::read(self, buf).await
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        async_std::io::ReadExt::read_exact(self, buf).await
    }
}
