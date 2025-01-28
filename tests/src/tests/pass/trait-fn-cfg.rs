use std::io::{Result, Write};

use async_trait::async_trait;
use async_generic::async_generic;

#[async_generic(
    #[async_trait]
    async_variant
)]
trait Read {
    #[async_generic(
        #[cfg(feature = "async")]
        async_signature(&mut self) -> Result<u8>
    )]
    fn read(&mut self) -> Result<u8>;

    #[cfg_attr(feature = "async", track_caller)]
    #[async_generic(
        #[cfg(feature = "async")]
        async_signature(&mut self, value: u8) -> Result<u8>
    )]
    fn write(&mut self, value: u8) -> Result<()>;
}

#[async_generic(
    #[async_trait]
    async_variant(copy_sync)
)]
trait Serialize {
    #[async_generic(
        async_signature<W: AsyncWrite>(&self, writer: &mut W) -> Result<()>
    )]
    fn serialize<W: Write>(&self, writer: &mut W) -> Result<()>;

    fn u8_slice(slice: &[Self]) -> Option<&[u8]>
    where
        Self: Sized,
    {
        let _ = slice;
        None
    }
}

#[async_trait]
trait AsyncWrite: Unpin + Send {
    async fn write_all(&mut self, buf: &[u8]) -> Result<()>;
}

#[async_generic(
    #[async_trait]
    async_variant(copy_sync)
)]
impl Serialize for u8 {
    #[async_generic(
        async_signature<W: AsyncWrite>(&self, writer: &mut W) -> Result<()>
    )]
    fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
        if _sync {
            writer.write_all(&[*self])?;
        } else {
            writer.write_all(&[*self]).await?;
        }
        Ok(())
    }

    fn u8_slice(slice: &[Self]) -> Option<&[u8]> {
        Some(slice)
    }
}

fn main() {}
