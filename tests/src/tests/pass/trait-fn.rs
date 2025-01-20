use core::fmt::{Debug, Display};

use async_generic::async_generic;
use async_trait::async_trait;

#[async_generic(async_trait)]
trait NoDefaultImpl {
    #[async_generic(async_signature(&self) -> String where Self: Send + Sync)]
    fn do_stuff(&self) -> String;
}

struct StructA;

#[async_generic(async_trait)]
impl NoDefaultImpl for StructA {
    #[async_generic(async_signature(&self) -> String where Self: Send + Sync)]
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

#[async_generic(async_trait)]
trait DefaultImpl {
    #[async_generic(async_signature(&self) -> String where Self: Send + Sync)]
    fn do_stuff(&self) -> String {
        if _async {
            self.my_async_stuff().await
        } else {
            "not async".to_owned()
        }
    }

    async fn my_async_stuff(&self) -> String
    where
        Self: Send + Sync,
    {
        "async".to_owned()
    }
}

struct StructB;

#[async_trait]
impl DefaultImpl for StructB {}

#[async_generic(async_trait)]
trait DefaultImplGenericParam {
    #[async_generic(
        async_signature<T>(&self, inp: &T) -> String
        where
            Self: Send + Sync,
            T: Debug + Send + Sync + ?Sized,
    )]
    fn do_stuff<T: Display + ?Sized>(&self, inp: &T) -> String {
        if _async {
            self.my_async_stuff(inp).await
        } else {
            inp.to_string()
        }
    }

    async fn my_async_stuff<T>(&self, inp: &T) -> String
    where
        Self: Send + Sync,
        T: Debug + Send + Sync + ?Sized,
    {
        format!("{inp:?}")
    }
}

struct StructC;

#[async_trait]
impl DefaultImplGenericParam for StructC {}

#[async_std::main]
async fn main() {
    let a = StructA;
    let b = StructB;
    let c = StructC;

    println!("sync => {}", a.do_stuff());
    println!("async => {}", a.do_stuff_async().await);

    println!("sync => {}", b.do_stuff());
    println!("async => {}", b.do_stuff_async().await);

    println!("sync => {}", c.do_stuff("123"));
    println!("async => {}", c.do_stuff_async("123").await);
}
