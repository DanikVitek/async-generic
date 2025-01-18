#![deny(warnings)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg, doc_cfg_hide))]

use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parenthesized,
    parse::{discouraged::Speculative, Parse, ParseStream, Result},
    parse_macro_input,
    punctuated::Punctuated,
    Block, Error, FnArg, Generics, ImplItemFn, ItemFn, Signature, Token, TraitItemFn,
};

use crate::async_generic_fn::{
    state::{Async, Sync},
    AsyncGenericFn,
};

mod async_generic_fn;
mod desugar_if_async;
mod util;

#[proc_macro_attribute]
pub fn async_generic(args: TokenStream, input: TokenStream) -> TokenStream {
    let async_signature: Option<AsyncSignature> = if args.is_empty() {
        None
    } else {
        Some(parse_macro_input!(args as AsyncSignature))
    };

    let target_fn = parse_macro_input!(input as TargetFn);

    let sync_fn = AsyncGenericFn::<Sync>::new(&target_fn);
    let async_fn = AsyncGenericFn::<Async>::new(&target_fn, async_signature);

    (quote! {
        #sync_fn
        #async_fn
    })
    .into()
}

struct AsyncSignature {
    inputs: Punctuated<FnArg, Token![,]>,
    generics: Generics,
}

impl Parse for AsyncSignature {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident: Ident = input.parse()?;
        if ident.to_string() != "async_signature" {
            return Err(Error::new(
                ident.span(),
                "async_generic can only take a async_signature argument",
            ));
        }

        let mut generics: Generics = input.parse()?;

        let content;
        parenthesized!(content in input);
        let inputs = Punctuated::parse_terminated(&content)?;

        generics.where_clause = if input.is_empty() {
            None
        } else {
            input.parse()?
        };

        Ok(AsyncSignature { inputs, generics })
    }
}

impl ToTokens for AsyncSignature {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            inputs, generics, ..
        } = self;
        let where_clause = self.generics.where_clause.as_ref();

        tokens.extend(quote! {
            #generics(#inputs)
            #where_clause
        });
    }
}

enum TargetFn {
    FreeStanding(ItemFn),
    Trait(TraitItemFn),
    Impl(ImplItemFn),
}

impl TargetFn {
    fn sig(&self) -> &Signature {
        match self {
            Self::FreeStanding(f) => &f.sig,
            Self::Trait(f) => &f.sig,
            Self::Impl(f) => &f.sig,
        }
    }

    fn block(&self) -> Option<&Block> {
        match self {
            Self::FreeStanding(f) => Some(&f.block),
            Self::Trait(f) => f.default.as_ref(),
            Self::Impl(f) => Some(&f.block),
        }
    }
}

impl Parse for TargetFn {
    fn parse(input: ParseStream) -> Result<Self> {
        let target_fn = {
            let fork = input.fork();
            fork.parse()
                .map(TargetFn::FreeStanding)
                .inspect(|_| input.advance_to(&fork))
                .or_else(|err1| {
                    let fork = input.fork();
                    fork.parse()
                        .map(TargetFn::Trait)
                        .inspect(|_| input.advance_to(&fork))
                        .or_else(|err2| {
                            let fork = input.fork();
                            fork.parse()
                                .map(TargetFn::Impl)
                                .inspect(|_| input.advance_to(&fork))
                                .or_else(|err3| {
                                    let mut err = Error::new(
                                        Span::call_site(),
                                        "async_generic can only be used with functions",
                                    );
                                    err.extend([err1, err2, err3]);
                                    Err(err)
                                })
                        })
                })?
        };

        if let Some(r#async) = &target_fn.sig().asyncness {
            Err(Error::new(
                r#async.span,
                "an async_generic function should not be declared as async",
            ))
        } else {
            Ok(target_fn)
        }
    }
}
