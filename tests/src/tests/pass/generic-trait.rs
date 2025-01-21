use async_trait::async_trait;
use async_generic::async_generic;

#[async_generic(
    #[async_trait]
    async_trait: Send
)]
trait Deserialize {
    #[async_generic]
    fn deserialize(&self) -> u8;

    fn sync_fn(&self) -> u8;

    async fn async_fn(&self) -> u8;
}

struct Foo;

#[async_generic(
    #[async_trait]
    async_trait
)]
impl Deserialize for Foo {
    #[async_generic]
    fn deserialize(&self) -> u8 {
        if _sync {
            self.sync_fn()
        } else {
            self.async_fn().await
        }
    }

    fn sync_fn(&self) -> u8 {
        0
    }

    async fn async_fn(&self) -> u8 {
        1
    }
}

#[async_std::main]
async fn main() {
    let foo = Foo;
    println!("sync => {}", Deserialize::deserialize(&foo));
    println!("async => {}", DeserializeAsync::deserialize(&foo).await);
}
