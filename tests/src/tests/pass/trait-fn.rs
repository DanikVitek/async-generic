use async_generic::async_generic;
use async_trait::async_trait;

#[async_trait]
trait NoDefaultImpl {
    #[async_generic(async_signature(&self) where Self: Send + Sync)]
    fn do_stuff(&self) -> String;
}

struct StructA {}

#[async_trait]
impl NoDefaultImpl for StructA {
    #[async_generic(async_signature(&self) where Self: Send + Sync)]
    fn do_stuff(&self) -> String {
        if _async {
            self.my_async_stuff().await
        } else {
            "not async".to_owned()
        }
    }
}

impl StructA {
    async fn my_async_stuff(&self) -> String {
        "async".to_owned()
    }
}

#[async_trait]
trait DefaultImpl {
    #[async_generic(async_signature(&self) where Self: Send + Sync)]
    fn do_stuff(&self) -> String {
        if _async {
            self.my_async_stuff().await
        } else {
            "not async".to_owned()
        }
    }

    async fn my_async_stuff(&self) -> String
    where
        Self: Send + Sync
    {
        "async".to_owned()
    }
}

struct StructB {}

#[async_trait]
impl DefaultImpl for StructB {}

#[async_std::main]
async fn main() {
    let a = StructA {};
    let b = StructB {};

    println!("sync => {}", a.do_stuff());
    println!("async => {}", a.do_stuff_async().await);

    println!("sync => {}", b.do_stuff());
    println!("async => {}", b.do_stuff_async().await);
}