use core::marker::PhantomData;

use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;
use syn::{parse2, AttrStyle, Attribute, Meta, MetaList};

use super::{
    r#fn::{AsyncGenericFn, TargetItemFn},
    state, LaterAttributes,
};

pub mod r#impl;
pub mod r#trait;

pub fn expand<T: TraitPart>(target: T, later_attributes: LaterAttributes) -> TokenStream2 {
    AsyncGenericTraitPart::<T, _>::new(target, later_attributes)
        .rewrite()
        .map(|r#trait| r#trait.into_token_stream())
        .unwrap_or_else(|err| err.into_compile_error())
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
    type ItemFn: HasAttributes + Into<TargetItemFn> + TryFrom<TargetItemFn>;

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
        let count = self
            .target
            .items()
            .iter()
            .map(|item| match item.as_item_fn() {
                Some(trait_item_fn) => {
                    if trait_item_fn.attrs().iter().any(|attr| {
                        matches!(attr.style, AttrStyle::Outer)
                            && attr.path().is_ident("async_generic")
                            && matches!(attr.meta, Meta::Path(_) | Meta::List(_))
                    }) {
                        2
                    } else {
                        1
                    }
                }
                None => 1,
            })
            .sum();
        let acc = Vec::with_capacity(count);
        self.target.try_replace_items(|items| {
            items
                .into_iter()
                .try_fold(acc, |mut acc, item| -> syn::Result<Vec<T::Item>> {
                    match item.to_item_fn() {
                        Ok(mut trait_item_fn) => {
                            match trait_item_fn.attrs().iter().position(|attr| {
                                matches!(attr.style, AttrStyle::Outer)
                                    && attr.path().is_ident("async_generic")
                            }) {
                                None => acc.push(From::from(trait_item_fn)),
                                Some(i) => match &trait_item_fn.attrs()[i].meta {
                                    Meta::Path(_) => {
                                        trait_item_fn.remove_attr(i).meta;
                                        let (
                                            AsyncGenericFn {
                                                target: sync_fn, ..
                                            },
                                            AsyncGenericFn {
                                                target: async_fn, ..
                                            },
                                        ) = super::r#fn::transform(trait_item_fn, None);
                                        acc.extend([sync_fn, async_fn].map(|r#fn| {
                                            From::from(
                                                TryFrom::try_from(r#fn)
                                                    .unwrap_or_else(|_| unreachable!()),
                                            )
                                        }));
                                    }
                                    Meta::List(_) => {
                                        let meta = trait_item_fn.remove_attr(i).meta;
                                        let Meta::List(MetaList { tokens: args, .. }) = meta else {
                                            unreachable!();
                                        };
                                        let async_signature = parse2(args)?;
                                        let (
                                            AsyncGenericFn {
                                                target: sync_fn, ..
                                            },
                                            AsyncGenericFn {
                                                target: async_fn, ..
                                            },
                                        ) = super::r#fn::transform(
                                            trait_item_fn,
                                            Some(async_signature),
                                        );
                                        acc.extend([sync_fn, async_fn].map(|r#fn| {
                                            From::from(
                                                TryFrom::try_from(r#fn)
                                                    .unwrap_or_else(|_| unreachable!()),
                                            )
                                        }));
                                    }
                                    Meta::NameValue(_) => acc.push(From::from(
                                        TryFrom::try_from(trait_item_fn)
                                            .unwrap_or_else(|_| unreachable!()),
                                    )),
                                },
                            }
                        }
                        Err(trait_item) => acc.push(trait_item),
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
