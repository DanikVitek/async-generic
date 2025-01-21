use proc_macro2::Span;
use syn::{
    bracketed,
    parse::{discouraged::Speculative, Parse, ParseStream},
    AttrStyle, Attribute, Error, Result, Token,
};

use self::{r#fn::TargetItemFn, trait_part::TargetTraitPart};

pub mod r#fn;
pub mod trait_part;

pub mod state {
    pub struct Initial;
    pub struct Final;
}

const ERROR_TARGET: &str = "`async_generic` can only be used on `fn`, `trait` or `impl`";

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
                    let mut err = Error::new(Span::call_site(), ERROR_TARGET);
                    err.extend([err1, err2]);
                    err
                })
            })?
        };

        Ok(target_item)
    }
}

pub(self) fn parse_attrs(input: ParseStream) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();
    while input.peek(Token![#]) {
        let content;
        attrs.push(Attribute {
            pound_token: input.parse()?,
            style: match input.parse::<Option<Token![!]>>()? {
                None => AttrStyle::Outer,
                Some(not_token) => AttrStyle::Inner(not_token),
            },
            bracket_token: bracketed!(content in input),
            meta: content.parse()?,
        });
    }
    Ok(attrs)
}
