use std::io::Result;

use async_trait::async_trait;
use async_generic::async_generic;

#[async_generic(async_trait)]
trait Read {
    #[cfg_attr(feature = "async", async_generic)]
    fn read(&mut self) -> Result<u8>;

    #[cfg_attr(feature = "async", async_generic, track_caller)]
    fn write(&mut self, value: u8) -> Result<()>;
}

fn main() {}
