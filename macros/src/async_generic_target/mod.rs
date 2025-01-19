use proc_macro2::Span;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    Error, ItemTrait,
};

use crate::async_generic_target::r#fn::TargetItemFn;

pub mod r#fn;
pub mod r#trait;

pub mod state {
    pub struct Initial;
    pub struct Final;
}

pub enum TargetItem {
    Fn(TargetItemFn),
    Trait(ItemTrait),
}

impl Parse for TargetItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let target_item = {
            use crate::util::InspectExt;

            let fork = input.fork();
            InspectExt::inspect(fork.parse().map(TargetItem::Fn), |_| {
                input.advance_to(&fork)
            })
            .or_else(|err1| {
                let fork = input.fork();
                InspectExt::inspect(fork.parse().map(TargetItem::Trait), |_| {
                    input.advance_to(&fork)
                })
                .or_else(|err2| {
                    let mut err = Error::new(
                        Span::call_site(),
                        "async_generic can only be used with traits or functions",
                    );
                    err.extend([err1, err2]);
                    Err(err)
                })
            })?
        };

        Ok(target_item)
    }
}
