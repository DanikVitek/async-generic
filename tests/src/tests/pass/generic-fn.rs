use async_generic::async_generic;

#[async_generic]
fn do_stuff() -> String {
    if _async {
        my_async_stuff().await
    } else {
        "not async".to_owned()
    }
}

#[async_generic(
    /// this is documentation for sync version
    sync_signature,
    /// this is documentation for async version
    async_signature,
)]
fn doc_do_stuff() -> String {
    if _async {
        do_stuff_async().await
    } else {
        do_stuff()
    }
}

async fn my_async_stuff() -> String {
    "async".to_owned()
}

#[async_std::main]
async fn main() {
    println!("sync => {}", do_stuff());
    println!("async => {}", do_stuff_async().await);
}
