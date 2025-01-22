use core::marker::PhantomData;

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::ToTokens;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    AttrStyle, Attribute, Error, Generics, ItemImpl, ItemTrait, Meta, MetaList, Path, Token,
    TypeParamBound,
};

use super::{
    parse_attrs, r#fn,
    r#fn::{AsyncGenericFn, AsyncSignature, TargetItemFn},
    state,
};
use crate::util::LetExt;

pub mod r#impl;
mod kind;
pub mod r#trait;

pub mod kw {
    syn::custom_keyword!(async_trait);
    syn::custom_keyword!(sync_trait);
}

const ERROR_PARSE_ARGS: &str =
    "`async_generic` on `trait` or `impl` can only take an `sync_trait` or `async_trait` argument";

pub fn expand(target: TargetTraitPart, args: AsyncGenericArgs) -> TokenStream2 {
    fn expand<T: TraitPart>(
        target: T,
        AsyncGenericArgs {
            sync_trait,
            async_trait,
        }: AsyncGenericArgs,
    ) -> TokenStream2 {
        match async_trait {
            None => AsyncGenericTraitPart::new(target, kind::Sync::<true>(sync_trait))
                .rewrite()
                .map(|res| res.into_token_stream())
                .unwrap_or_else(|err| err.into_compile_error()),
            Some(async_trait) => {
                let sync_trait =
                    AsyncGenericTraitPart::new(target.clone(), kind::Sync::<false>(sync_trait))
                        .rewrite()
                        .map(|res| res.into_token_stream())
                        .unwrap_or_else(|err| err.into_compile_error());

                let async_trait = AsyncGenericTraitPart::new(target, kind::Async(async_trait))
                    .rewrite()
                    .map(|res| res.into_token_stream())
                    .unwrap_or_else(|err| err.into_compile_error());

                let mut tt = TokenStream2::new();
                tt.extend([sync_trait, async_trait]);
                tt
            }
        }
    }
    match target {
        TargetTraitPart::Trait(item) => expand(item, args),
        TargetTraitPart::Impl(item) => expand(item, args),
    }
}

pub enum TargetTraitPart {
    Trait(ItemTrait),
    Impl(ItemImpl),
}

pub struct AsyncGenericArgs {
    sync_trait: Option<SyncTrait>,
    async_trait: Option<AsyncTrait>,
}

pub struct SyncTrait {
    attrs: Vec<Attribute>,
    _sync_trait_token: kw::sync_trait,
}

pub struct AsyncTrait {
    attrs: Vec<Attribute>,
    _async_trait_token: kw::async_trait,
    generics: Generics,
    supertraits: Option<(Token![:], Punctuated<TypeParamBound, Token![+]>)>,
}

impl Parse for AsyncGenericArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        if input.peek(kw::sync_trait) {
            let mut sync_trait: SyncTrait = input.parse()?;
            sync_trait.attrs.extend(attrs);
            let _: Token![,] = if !input.is_empty() {
                input.parse()?
            } else {
                return Ok(Self {
                    sync_trait: Some(sync_trait),
                    async_trait: None,
                });
            };
            Ok(Self {
                sync_trait: Some(sync_trait),
                async_trait: if input.is_empty() {
                    None
                } else {
                    Some(input.parse()?)
                },
            })
        } else if input.peek(kw::async_trait) {
            let mut async_trait: AsyncTrait = input.parse()?;
            async_trait.attrs.extend(attrs);
            let _: Token![,] = match async_trait
                .generics
                .where_clause
                .as_mut()
                .map(|where_clause| where_clause.predicates.pop_punct())
                .flatten()
            {
                None if input.is_empty() => {
                    return Ok(Self {
                        sync_trait: None,
                        async_trait: Some(async_trait),
                    })
                }
                None => input.parse()?,
                Some(comma_token) => comma_token,
            };
            Ok(Self {
                async_trait: Some(async_trait),
                sync_trait: if input.is_empty() {
                    None
                } else {
                    Some(input.parse()?)
                },
            })
        } else if !input.is_empty() || !attrs.is_empty() {
            Err(input.error(ERROR_PARSE_ARGS))
        } else {
            Ok(Self {
                sync_trait: None,
                async_trait: None,
            })
        }
    }
}

impl Parse for SyncTrait {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let sync_trait_token = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;
        Ok(Self {
            attrs,
            _sync_trait_token: sync_trait_token,
        })
    }
}

impl Parse for AsyncTrait {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let async_trait_token = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;
        let mut generics: Generics = input.parse()?;

        let supertraits = if input.peek(Token![:]) {
            Some((input.parse()?, Punctuated::parse_terminated(input)?))
        } else {
            None
        };
        generics.where_clause = if input.peek(Token![where]) {
            Some(input.parse()?)
        } else {
            None
        };

        Ok(Self {
            attrs,
            _async_trait_token: async_trait_token,
            generics,
            supertraits,
        })
    }
}

impl Parse for TargetTraitPart {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let target_item = {
            use crate::util::InspectExt;

            let fork = input.fork();
            InspectExt::inspect(fork.parse().map(TargetTraitPart::Trait), |_| {
                input.advance_to(&fork)
            })
            .or_else(|mut err1| -> syn::Result<_> {
                let fork = input.fork();
                let item_impl = match fork.parse::<ItemImpl>() {
                    Ok(item_impl) if item_impl.trait_.is_none() => {
                        err1.extend(Some(Error::new(
                            item_impl.impl_token.span(),
                            "expected `impl` for a `trait`",
                        )));
                        Err(err1)
                    }
                    Ok(item_impl) => Ok(item_impl),
                    Err(err2) => {
                        err1.extend(Some(err2));
                        Err(err1)
                    }
                }?;
                input.advance_to(&fork);
                Ok(TargetTraitPart::Impl(item_impl))
            })?
        };

        Ok(target_item)
    }
}

pub trait TraitPart: Clone + ToTokens {
    type Item: TraitPartItem + From<<Self::Item as TraitPartItem>::ItemFn>;

    fn update_ident(&mut self, f: impl FnOnce(Ident) -> Ident);

    fn items(&self) -> &[Self::Item];

    fn items_mut(&mut self) -> &mut Vec<Self::Item>;

    fn set_items(&mut self, items: Vec<Self::Item>);

    fn try_update_items<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(Vec<Self::Item>) -> Result<Vec<Self::Item>, E>,
    {
        let items = f(std::mem::take(self.items_mut()))?;
        self.set_items(items);
        Ok(())
    }

    fn extend_attrs(&mut self, iter: impl IntoIterator<Item = Attribute>);

    fn set_colon_token(&mut self, colon_token: Token![:]);
    fn set_supertraits(&mut self, supertraits: Punctuated<TypeParamBound, Token![+]>);
    fn set_generics(&mut self, generics: Generics);
}

pub trait TraitPartItem {
    type ItemFn: Clone + HasAttributes + HasAsyncness + Into<TargetItemFn> + TryFrom<TargetItemFn>;

    fn as_item_fn(&self) -> Option<&Self::ItemFn>;
    fn to_item_fn(self) -> Result<Self::ItemFn, Self>
    where
        Self: Sized;
}

pub trait HasAttributes {
    fn attrs(&self) -> &[Attribute];
    fn remove_attr(&mut self, i: usize) -> Attribute;
}

pub trait HasAsyncness {
    fn is_async(&self) -> bool;
}

struct AsyncGenericTraitPart<T, A, S> {
    target: T,
    kind: A,
    _state: PhantomData<S>,
}

impl<T, A> AsyncGenericTraitPart<T, A, state::Initial> {
    pub fn new(target: T, kind: A) -> Self {
        Self {
            target,
            kind,
            _state: PhantomData,
        }
    }
}

trait CanBeRewritten {
    type Output;

    fn rewrite(self) -> syn::Result<Self::Output>;
}

impl<T> CanBeRewritten for AsyncGenericTraitPart<T, kind::Sync<true>, state::Initial>
where
    T: TraitPart,
{
    type Output = AsyncGenericTraitPart<T, kind::Sync<true>, state::Final>;

    fn rewrite(mut self) -> syn::Result<Self::Output> {
        self.target.try_update_items(|items| {
            items
                .into_iter()
                .try_fold(vec![], |mut acc, item| -> syn::Result<Vec<T::Item>> {
                    match item.to_item_fn() {
                        Err(trait_item) => acc.push(trait_item),
                        Ok(mut trait_item_fn) => {
                            let suitable_attr =
                                trait_item_fn.attrs().iter().position(attr_is_suitable);

                            let Some(i) = suitable_attr else {
                                acc.push(From::from(trait_item_fn));
                                return Ok(acc);
                            };
                            let async_signature = match take_async_signature(&mut trait_item_fn, i)
                            {
                                Ok(result) => result?,
                                Err(_) => {
                                    // Not remove to let the second pass handle the error
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
                            ) = super::r#fn::split::<false>(
                                trait_item_fn.clone(),
                                r#fn::AsyncGenericArgs(async_signature),
                            );
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
            self.kind
                .0
                .take()
                .into_iter()
                .flat_map(|sync_trait| sync_trait.attrs),
        );

        Ok(AsyncGenericTraitPart {
            target: self.target,
            kind: self.kind,
            _state: PhantomData,
        })
    }
}

impl<T> CanBeRewritten for AsyncGenericTraitPart<T, kind::Sync<false>, state::Initial>
where
    T: TraitPart,
{
    type Output = AsyncGenericTraitPart<T, kind::Sync<false>, state::Final>;

    fn rewrite(mut self) -> syn::Result<Self::Output> {
        self.target.try_update_items(|items| {
            items
                .into_iter()
                .try_fold(vec![], |mut acc, item| -> syn::Result<Vec<T::Item>> {
                    match item.to_item_fn() {
                        Err(trait_item) => acc.push(trait_item),
                        Ok(mut trait_item_fn) => {
                            let suitable_attr =
                                trait_item_fn.attrs().iter().position(attr_is_suitable);

                            let Some(i) = suitable_attr else {
                                if !trait_item_fn.is_async() {
                                    acc.push(T::Item::from(trait_item_fn));
                                }
                                return Ok(acc);
                            };
                            let _async_signature: Option<AsyncSignature> =
                                match take_async_signature(&mut trait_item_fn, i) {
                                    Ok(result) => result?,
                                    Err(_) => {
                                        // Not remove to let the second pass handle the error
                                        if !trait_item_fn.is_async() {
                                            acc.push(T::Item::from(trait_item_fn));
                                        }
                                        return Ok(acc);
                                    }
                                };
                            let AsyncGenericFn {
                                target: sync_fn, ..
                            } = AsyncGenericFn::<r#fn::kind::Sync, state::Initial>::new(
                                trait_item_fn.into(),
                            )
                            .rewrite();

                            acc.push(T::Item::from(
                                <T::Item as TraitPartItem>::ItemFn::try_from(sync_fn)
                                    .unwrap_or_else(|_| unreachable!()),
                            ));
                        }
                    }
                    Ok(acc)
                })
        })?;
        self.target.extend_attrs(
            self.kind
                .0
                .take()
                .into_iter()
                .flat_map(|sync_trait| sync_trait.attrs),
        );

        Ok(AsyncGenericTraitPart {
            target: self.target,
            kind: self.kind,
            _state: PhantomData,
        })
    }
}

impl<T> CanBeRewritten for AsyncGenericTraitPart<T, kind::Async, state::Initial>
where
    T: TraitPart,
{
    type Output = AsyncGenericTraitPart<T, kind::Async, state::Final>;

    fn rewrite(mut self) -> syn::Result<Self::Output> {
        self.target.try_update_items(|items| {
            items
                .into_iter()
                .try_fold(vec![], |mut acc, item| -> syn::Result<Vec<T::Item>> {
                    match item.to_item_fn() {
                        Err(trait_item) => acc.push(trait_item),
                        Ok(mut trait_item_fn) => {
                            let suitable_attr =
                                trait_item_fn.attrs().iter().position(attr_is_suitable);

                            let Some(i) = suitable_attr else {
                                if trait_item_fn.is_async() {
                                    acc.push(T::Item::from(trait_item_fn));
                                }
                                return Ok(acc);
                            };
                            let async_signature = match take_async_signature(&mut trait_item_fn, i)
                            {
                                Ok(result) => result?,
                                Err(_) => {
                                    // Not remove to let the second pass handle the error
                                    if trait_item_fn.is_async() {
                                        acc.push(T::Item::from(trait_item_fn));
                                    }
                                    return Ok(acc);
                                }
                            };

                            let AsyncGenericFn {
                                target: async_fn, ..
                            } = AsyncGenericFn::<r#fn::kind::Async<true>, state::Initial>::new(
                                trait_item_fn.into(),
                                async_signature,
                            )
                            .rewrite();

                            acc.push(T::Item::from(
                                <T::Item as TraitPartItem>::ItemFn::try_from(async_fn)
                                    .unwrap_or_else(|_| unreachable!()),
                            ));
                        }
                    }
                    Ok(acc)
                })
        })?;

        self.target
            .extend_attrs(std::mem::take(&mut self.kind.0.attrs));
        if let Some((colon_token, supertraits)) = self.kind.0.supertraits.take() {
            self.target.set_colon_token(colon_token);
            self.target.set_supertraits(supertraits);
        }
        self.target
            .set_generics(std::mem::take(&mut self.kind.0.generics));
        self.target
            .update_ident(|ident| Ident::new(&format!("{ident}Async"), ident.span()));

        Ok(AsyncGenericTraitPart {
            target: self.target,
            kind: self.kind,
            _state: PhantomData,
        })
    }
}

fn attr_is_suitable(attr: &Attribute) -> bool {
    matches!(attr.style, AttrStyle::Outer) && path_is_async_generic(attr.path())
}

fn path_is_async_generic(path: &Path) -> bool {
    path.segments.iter().r#let(|mut segments| {
        [1, 2].contains(&segments.len())
            && segments
                .all(|segment| segment.ident == "async_generic" && segment.arguments.is_empty())
    })
}

fn take_async_signature<T: HasAttributes>(
    trait_item_fn: &mut T,
    i: usize,
) -> Result<syn::Result<Option<AsyncSignature>>, ()> {
    match &trait_item_fn.attrs()[i].meta {
        Meta::Path(_) => {
            trait_item_fn.remove_attr(i).meta;
            Ok(Ok(None))
        }
        Meta::List(_) => {
            let meta = trait_item_fn.remove_attr(i).meta;
            let Meta::List(MetaList { tokens: args, .. }) = meta else {
                unreachable!();
            };
            Ok(parse2(args).map(Some))
        }
        Meta::NameValue(_) => Err(()),
    }
}

impl<T, A> ToTokens for AsyncGenericTraitPart<T, A, state::Final>
where
    T: TraitPart,
{
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.target.to_tokens(tokens);
    }
}
