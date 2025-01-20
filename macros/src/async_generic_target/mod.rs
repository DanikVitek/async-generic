use proc_macro2::Span;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    Error, Result,
};

use self::{r#fn::TargetItemFn, trait_part::TargetTraitPart};

pub mod r#fn;
pub mod trait_part;

pub mod state {
    pub struct Initial;
    pub struct Final;
}

pub enum TargetItem {
    Fn(TargetItemFn),
    TraitPart(TargetTraitPart),
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
                InspectExt::inspect(fork.parse().map(TargetItem::TraitPart), |_| {
                    input.advance_to(&fork)
                })
                .map_err(|err2| {
                    let mut err = Error::new(
                        Span::call_site(),
                        "async_generic can only be used with functions, traits or impls",
                    );
                    err.extend([err1, err2]);
                    err
                })
            })?
        };

        Ok(target_item)
    }
}
