use proc_macro2::Span;
use syn::{
    bracketed,
    parse::{discouraged::Speculative, End, Parse, ParseStream},
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

trait CanSetAttrs {
    fn set_attrs(&mut self, attrs: Vec<Attribute>);
}

fn parse_in_order<A, B, Args>(input: ParseStream, attrs: Vec<Attribute>) -> Result<Args>
where
    A: Parse + CanSetAttrs,
    B: Parse + CanSetAttrs,
    Args: From<A> + From<B> + From<(A, B)>,
{
    let mut async_signature: A = input.parse()?;
    async_signature.set_attrs(attrs);

    let lookahead = input.lookahead1();
    if !lookahead.peek(Token![;]) {
        if lookahead.peek(End) {
            return Ok(Args::from(async_signature));
        }
        return Err(lookahead.error());
    }
    let _: Token![;] = input.parse()?;

    if input.is_empty() {
        return Ok(Args::from(async_signature));
    }

    let sync_signature = input.parse()?;

    let lookahead = input.lookahead1();
    if !lookahead.peek(Token![;]) && !lookahead.peek(End) {
        return Err(lookahead.error());
    }
    let _: Option<Token![;]> = input.parse()?;

    Ok(Args::from((async_signature, sync_signature)))
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
