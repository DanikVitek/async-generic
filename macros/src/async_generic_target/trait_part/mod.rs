use core::marker::PhantomData;

use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    AttrStyle, Attribute, ItemImpl, ItemTrait, Meta, MetaList, Token,
};

use super::{
    r#fn::{AsyncGenericFn, TargetItemFn},
    state,
};

pub mod r#impl;
pub mod r#trait;

pub fn expand(target: TargetTraitPart, later_attributes: LaterAttributes) -> TokenStream2 {
    fn expand<T: TraitPart>(target: T, later_attributes: LaterAttributes) -> TokenStream2 {
        AsyncGenericTraitPart::new(target, later_attributes)
            .rewrite()
            .map(|r#trait| r#trait.into_token_stream())
            .unwrap_or_else(|err| err.into_compile_error())
    }
    match target {
        TargetTraitPart::Trait(item) => expand(item, later_attributes),
        TargetTraitPart::Impl(item) => expand(item, later_attributes),
    }
}

pub enum TargetTraitPart {
    Trait(ItemTrait),
    Impl(ItemImpl),
}

impl Parse for TargetTraitPart {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let target_item = {
            use crate::util::InspectExt;

            let fork = input.fork();
            InspectExt::inspect(fork.parse().map(TargetTraitPart::Trait), |_| {
                input.advance_to(&fork)
            })
            .or_else(|mut err1| {
                let fork = input.fork();
                InspectExt::inspect(fork.parse().map(TargetTraitPart::Impl), |_| {
                    input.advance_to(&fork)
                })
                .map_err(|err2| {
                    err1.extend(Some(err2));
                    err1
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
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            attrs: Punctuated::parse_terminated(input)?,
        })
    }
}

pub trait TraitPart: ToTokens {
    type Item: TraitPartItem + From<<Self::Item as TraitPartItem>::ItemFn>;

    fn items(&self) -> &[Self::Item];

    fn try_replace_items<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(Vec<Self::Item>) -> Result<Vec<Self::Item>, E>;

    fn extend_attrs(&mut self, iter: impl IntoIterator<Item = Attribute>);
}

pub trait TraitPartItem {
    type ItemFn: Clone + HasAttributes + Into<TargetItemFn> + TryFrom<TargetItemFn>;

    fn as_item_fn(&self) -> Option<&Self::ItemFn>;
    fn to_item_fn(self) -> Result<Self::ItemFn, Self>
    where
        Self: Sized;
}

pub trait HasAttributes {
    fn attrs(&self) -> &[Attribute];
    fn remove_attr(&mut self, i: usize) -> Attribute;
}

struct AsyncGenericTraitPart<T, S> {
    target: T,
    later_attributes: LaterAttributes,
    _state: PhantomData<S>,
}

impl<T> AsyncGenericTraitPart<T, state::Initial> {
    pub fn new(target: T, later_attributes: LaterAttributes) -> Self {
        Self {
            target,
            later_attributes,
            _state: PhantomData,
        }
    }
}

impl<T> AsyncGenericTraitPart<T, state::Initial>
where
    T: TraitPart,
{
    pub fn rewrite(mut self) -> syn::Result<AsyncGenericTraitPart<T, state::Final>> {
        self.target.try_replace_items(|items| {
            items
                .into_iter()
                .try_fold(vec![], |mut acc, item| -> syn::Result<Vec<T::Item>> {
                    match item.to_item_fn() {
                        Err(trait_item) => acc.push(trait_item),
                        Ok(mut trait_item_fn) => {
                            let suitable_attr = trait_item_fn.attrs().iter().position(|attr| {
                                matches!(attr.style, AttrStyle::Outer)
                                    && attr.path().is_ident("async_generic")
                            });

                            let Some(i) = suitable_attr else {
                                acc.push(From::from(trait_item_fn));
                                return Ok(acc);
                            };
                            let async_signature = match &trait_item_fn.attrs()[i].meta {
                                Meta::Path(_) => {
                                    trait_item_fn.remove_attr(i).meta;
                                    None
                                }
                                Meta::List(_) => {
                                    let meta = trait_item_fn.remove_attr(i).meta;
                                    let Meta::List(MetaList { tokens: args, .. }) = meta else {
                                        unreachable!();
                                    };
                                    Some(parse2(args)?)
                                }
                                Meta::NameValue(_) => {
                                    acc.push(T::Item::from(trait_item_fn));
                                    return Ok(acc);
                                }
                            };
                            let (
                                AsyncGenericFn {
                                    target: sync_fn, ..
                                },
                                AsyncGenericFn {
                                    target: async_fn, ..
                                },
                            ) = super::r#fn::transform(trait_item_fn.clone(), async_signature);
                            acc.extend([sync_fn, async_fn].map(|f| {
                                T::Item::from(
                                    <T::Item as TraitPartItem>::ItemFn::try_from(f)
                                        .unwrap_or_else(|_| unreachable!()),
                                )
                            }));
                        }
                    }
                    Ok(acc)
                })
        })?;
        self.target.extend_attrs(
            std::mem::take(&mut self.later_attributes.attrs)
                .into_iter()
                .map(|meta| Attribute {
                    pound_token: Default::default(),
                    style: AttrStyle::Outer,
                    bracket_token: Default::default(),
                    meta,
                }),
        );

        Ok(AsyncGenericTraitPart {
            target: self.target,
            later_attributes: self.later_attributes,
            _state: PhantomData,
        })
    }
}

impl<T> ToTokens for AsyncGenericTraitPart<T, state::Final>
where
    T: TraitPart,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.target.to_tokens(tokens);
    }
}
