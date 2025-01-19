use std::io::{Read, Result};

use async_generic::async_generic;
use async_trait::async_trait;

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

#[async_std::main]
async fn main() {
    let mut reader: &[u8] = &[0, 1, 2, 3];
    println!("sync => {:?}", do_stuff(&mut reader));
    println!("async => {:?}", do_stuff_async(&mut reader).await);
}
