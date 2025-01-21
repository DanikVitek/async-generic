use std::io::Result;

use async_trait::async_trait;
use async_generic::async_generic;

#[async_generic(async_trait)]
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

fn main() {}
