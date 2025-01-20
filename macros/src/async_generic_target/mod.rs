use proc_macro2::Span;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    punctuated::Punctuated,
    Error, ItemImpl, ItemTrait, Meta, Result, Token,
};

use self::r#fn::TargetItemFn;

pub mod r#fn;
pub mod r#impl;
pub mod r#trait;

pub mod state {
    pub struct Initial;
    pub struct Final;
}

pub enum TargetItem {
    Fn(TargetItemFn),
    Trait(ItemTrait),
    Impl(ItemImpl),
}

impl Parse for TargetItem {
    fn parse(input: ParseStream) -> Result<Self> {
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
                    let fork = input.fork();
                    InspectExt::inspect(fork.parse().map(TargetItem::Impl), |_| {
                        input.advance_to(&fork)
                    })
                    .map_err(|err3| {
                        let mut err = Error::new(
                            Span::call_site(),
                            "async_generic can only be used with functions, traits or impls",
                        );
                        err.extend([err1, err2, err3]);
                        err
                    })
                })
            })?
        };

        Ok(target_item)
    }
}

#[derive(Default)]
pub struct LaterAttributes {
    attrs: Punctuated<Meta, Token![,]>,
}

impl Parse for LaterAttributes {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            attrs: Punctuated::parse_terminated(input)?,
        })
    }
}
