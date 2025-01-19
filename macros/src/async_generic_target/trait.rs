use std::marker::PhantomData;

use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    AttrStyle, Attribute, ItemTrait, Meta, MetaList, Result, Token, TraitItem,
};

use crate::async_generic_target::{
    r#fn::{AsyncGenericFn, TargetItemFn},
    state,
};

pub fn expand(target: ItemTrait, later_attributes: LaterAttributes) -> TokenStream2 {
    AsyncGenericTrait::new(target, later_attributes)
        .rewrite()
        .map(|r#trait| r#trait.into_token_stream())
        .unwrap_or_else(|err| err.into_compile_error())
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

pub struct AsyncGenericTrait<S> {
    target: ItemTrait,
    later_attributes: LaterAttributes,
    _state: PhantomData<S>,
}

impl AsyncGenericTrait<state::Initial> {
    pub fn new(target: ItemTrait, later_attributes: LaterAttributes) -> Self {
        Self {
            target,
            later_attributes,
            _state: PhantomData,
        }
    }
}

impl AsyncGenericTrait<state::Initial> {
    pub fn rewrite(mut self) -> Result<AsyncGenericTrait<state::Final>> {
        let acc = Vec::with_capacity(self.target.items.len());
        self.target.items = self.target.items.into_iter().try_fold(
            acc,
            |mut acc, item| -> Result<Vec<TraitItem>> {
                match item {
                    TraitItem::Fn(mut trait_item_fn) => {
                        if let Some(i) = trait_item_fn.attrs.iter().position(|attr| {
                            matches!(attr.style, AttrStyle::Outer)
                                && attr.path().is_ident("async_generic")
                        }) {
                            match &trait_item_fn.attrs[i].meta {
                                Meta::Path(_) => {
                                    trait_item_fn.attrs.remove(i).meta;
                                    let (
                                        AsyncGenericFn {
                                            target: TargetItemFn::Trait(sync_fn),
                                            ..
                                        },
                                        AsyncGenericFn {
                                            target: TargetItemFn::Trait(async_fn),
                                            ..
                                        },
                                    ) = super::r#fn::transform(
                                        TargetItemFn::Trait(trait_item_fn),
                                        None,
                                    )
                                    else {
                                        unreachable!()
                                    };
                                    acc.extend([sync_fn, async_fn].map(TraitItem::Fn));
                                }
                                Meta::List(_) => {
                                    let meta = trait_item_fn.attrs.remove(i).meta;
                                    let Meta::List(MetaList { tokens: args, .. }) = meta else {
                                        unreachable!();
                                    };
                                    let async_signature = parse2(args)?;
                                    let (
                                        AsyncGenericFn {
                                            target: TargetItemFn::Trait(sync_fn),
                                            ..
                                        },
                                        AsyncGenericFn {
                                            target: TargetItemFn::Trait(async_fn),
                                            ..
                                        },
                                    ) = super::r#fn::transform(
                                        TargetItemFn::Trait(trait_item_fn),
                                        Some(async_signature),
                                    )
                                    else {
                                        unreachable!()
                                    };
                                    acc.extend([sync_fn, async_fn].map(TraitItem::Fn));
                                }
                                Meta::NameValue(_) => acc.extend([TraitItem::Fn(trait_item_fn)]),
                            }
                        } else {
                            acc.extend([TraitItem::Fn(trait_item_fn)])
                        }
                    }
                    trait_item => acc.extend([trait_item]),
                }
                Ok(acc)
            },
        )?;
        self.target.attrs.extend(
            std::mem::take(&mut self.later_attributes.attrs)
                .into_iter()
                .map(|meta| Attribute {
                    pound_token: Default::default(),
                    style: AttrStyle::Outer,
                    bracket_token: Default::default(),
                    meta,
                }),
        );

        Ok(AsyncGenericTrait {
            target: self.target,
            later_attributes: self.later_attributes,
            _state: PhantomData,
        })
    }
}

impl ToTokens for AsyncGenericTrait<state::Final> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.target.to_tokens(tokens);
    }
}
