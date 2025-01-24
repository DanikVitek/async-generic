use async_generic::async_generic;
use async_trait::async_trait;

#[async_generic(
    #[async_trait]
)]
trait A {}

fn main() {}