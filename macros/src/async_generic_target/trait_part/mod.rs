use core::marker::PhantomData;

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::ToTokens;
use syn::{
    parenthesized,
    parse::{discouraged::Speculative, End, Parse, ParseStream},
    parse2,
    punctuated::Punctuated,
    spanned::Spanned,
    token::Paren,
    AttrStyle, Attribute, Error, Generics, ItemImpl, ItemTrait, Meta, MetaList, Path, Token,
    TypeParamBound,
};

use super::{parse_attrs, parse_in_order, r#fn, r#fn::{AsyncGenericFn, TargetItemFn}, state, CanSetAttrs};
use crate::util::LetExt;

pub mod r#impl;
mod kind;
pub mod r#trait;

pub mod kw {
    use syn::custom_keyword;

    custom_keyword!(async_trait);
    custom_keyword!(sync_trait);
    custom_keyword!(copy_sync);
}

const ERROR_PARSE_ARGS: &str =
    "`async_generic` on `trait` or `impl` can only take an `sync_trait` or `async_trait` argument";
const ERROR_UNATTAINED_ATTRIBUTES: &str =
    "attributes must be placed on `sync_trait` and/or `async_trait`";

#[inline]
pub fn expand(target: impl Into<TargetTraitPart>, args: AsyncGenericArgs) -> TokenStream2 {
    fn expand(target: TargetTraitPart, args: AsyncGenericArgs) -> TokenStream2 {
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
    expand(target.into(), args)
}

pub enum TargetTraitPart {
    Trait(ItemTrait),
    Impl(ItemImpl),
}

impl From<ItemTrait> for TargetTraitPart {
    fn from(value: ItemTrait) -> Self {
        Self::Trait(value)
    }
}

impl From<ItemImpl> for TargetTraitPart {
    fn from(value: ItemImpl) -> Self {
        Self::Impl(value)
    }
}

#[derive(Default)]
pub struct AsyncGenericArgs {
    sync_trait: Option<SyncTrait>,
    async_trait: Option<AsyncTrait>,
}

pub struct SyncTrait {
    attrs: Vec<Attribute>,
}

pub struct AsyncTrait {
    attrs: Vec<Attribute>,
    options: Option<Options>,
    generics: Generics,
    supertraits: Option<(Token![:], Punctuated<TypeParamBound, Token![+]>)>,
}

#[derive(Clone, Copy, Default)]
pub struct Options {
    copy_sync: bool,
}

impl Parse for AsyncGenericArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let fork1 = input.fork();
        let attrs = parse_attrs(input)?;
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::sync_trait) {
            parse_in_order::<SyncTrait, AsyncTrait, _>(input, attrs)
        } else if lookahead.peek(kw::async_trait) {
            parse_in_order::<AsyncTrait, SyncTrait, _>(input, attrs)
        } else if !lookahead.peek(End) || !attrs.is_empty() {
            let mut err = lookahead.error();
            err.extend(fork1.error(ERROR_UNATTAINED_ATTRIBUTES));
            Err(err)
        } else {
            Ok(Self::default())
        }
    }
}

impl CanSetAttrs for SyncTrait {
    fn set_attrs(&mut self, attrs: Vec<Attribute>) {
        self.attrs = attrs;
    }
}

impl CanSetAttrs for AsyncTrait {
    fn set_attrs(&mut self, attrs: Vec<Attribute>) {
        self.attrs = attrs;
    }
}

impl From<SyncTrait> for AsyncGenericArgs {
    fn from(sync_trait: SyncTrait) -> Self {
        Self {
            sync_trait: Some(sync_trait),
            async_trait: None,
        }
    }
}

impl From<AsyncTrait> for AsyncGenericArgs {
    fn from(async_trait: AsyncTrait) -> Self {
        Self {
            sync_trait: None,
            async_trait: Some(async_trait),
        }
    }
}

impl From<(SyncTrait, AsyncTrait)> for AsyncGenericArgs {
    fn from((sync_trait, async_trait): (SyncTrait, AsyncTrait)) -> Self {
        Self {
            sync_trait: Some(sync_trait),
            async_trait: Some(async_trait),
        }
    }
}

impl From<(AsyncTrait, SyncTrait)> for AsyncGenericArgs {
    fn from((async_trait, sync_trait): (AsyncTrait, SyncTrait)) -> Self {
        Self {
            sync_trait: Some(sync_trait),
            async_trait: Some(async_trait),
        }
    }
}

impl Parse for SyncTrait {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let _: kw::sync_trait = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;
        Ok(Self { attrs })
    }
}

impl Parse for AsyncTrait {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let _: kw::async_trait = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;

        let options = if input.peek(Paren) {
            Some(input.parse()?)
        } else {
            None
        };

        let mut generics: Generics = input.parse()?;

        let supertraits = if input.peek(Token![:]) {
            Some((input.parse()?, Punctuated::parse_separated_nonempty(input)?))
        } else {
            None
        };
        generics.where_clause = input.parse()?;

        Ok(Self {
            attrs,
            options,
            generics,
            supertraits,
        })
    }
}

impl Parse for Options {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        let _ = parenthesized!(content in input);
        let options = Punctuated::<Ident, Token![,]>::parse_terminated(&content)?;
        let idents = options.iter().collect::<Vec<_>>();
        if idents.is_empty() {
            return Ok(Self::default());
        }
        if idents.len() > 1 {
            return Err(Error::new(
                options.span(),
                "expected at most one option, found multiple",
            ));
        }
        let copy_sync;
        if idents.iter().any(|ident| **ident == "copy_sync") {
            copy_sync = true;
        } else {
            return Err(Error::new(
                idents[0].span(),
                "expected `copy_sync` as the only option",
            ));
        }
        Ok(Self { copy_sync })
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

    fn items_mut(&mut self) -> &mut Vec<Self::Item>;

    fn set_items(&mut self, items: Vec<Self::Item>);

    fn try_update_items<F, E>(&mut self, f: F) -> Result<(), E>
    where
        F: FnOnce(Vec<Self::Item>) -> Result<Vec<Self::Item>, E>,
    {
        let items = f(core::mem::take(self.items_mut()))?;
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
                            let async_generic_args =
                                match take_async_generic_args(&mut trait_item_fn, i) {
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
                                async_generic_args,
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
                            let async_generic_args: r#fn::AsyncGenericArgs =
                                match take_async_generic_args(&mut trait_item_fn, i) {
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
                                async_generic_args.sync_signature,
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
                                if trait_item_fn.is_async()
                                    || matches!(self.kind.0.options, Some(options) if options.copy_sync)
                                {
                                    acc.push(T::Item::from(trait_item_fn));
                                }
                                return Ok(acc);
                            };
                            let async_generic_args =
                                match take_async_generic_args(&mut trait_item_fn, i) {
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
                                async_generic_args.async_signature,
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
            .extend_attrs(core::mem::take(&mut self.kind.0.attrs));
        if let Some((colon_token, supertraits)) = self.kind.0.supertraits.take() {
            self.target.set_colon_token(colon_token);
            self.target.set_supertraits(supertraits);
        }
        self.target
            .set_generics(core::mem::take(&mut self.kind.0.generics));
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

fn take_async_generic_args<T: HasAttributes>(
    trait_item_fn: &mut T,
    i: usize,
) -> Result<syn::Result<r#fn::AsyncGenericArgs>, ()> {
    match &trait_item_fn.attrs()[i].meta {
        Meta::Path(_) => {
            trait_item_fn.remove_attr(i).meta;
            Ok(Ok(r#fn::AsyncGenericArgs::default()))
        }
        Meta::List(_) => {
            let meta = trait_item_fn.remove_attr(i).meta;
            let Meta::List(MetaList { tokens: args, .. }) = meta else {
                unreachable!();
            };
            Ok(parse2(args))
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_str_eq;
    use syn::parse_quote;

    use super::*;
    use crate::test_helpers::test_expand;

    fn format_expand(target_fn: impl Into<TargetTraitPart>, args: AsyncGenericArgs) -> String {
        let expanded = expand(target_fn, args);
        prettyplease::unparse(&parse2(expanded).unwrap())
    }

    #[test]
    fn test_expand_trait_nop() {
        let target: ItemTrait = parse_quote! {
            trait Foo {}
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait
        };

        let formatted2 = format_expand(target.clone(), args);

        assert_str_eq!(formatted1, formatted2);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_impl_nop() {
        let target: ItemImpl = parse_quote! {
            impl Foo for A {}
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait
        };

        let formatted2 = format_expand(target.clone(), args);

        assert_str_eq!(formatted1, formatted2);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_trait_split_async_generic_fns() {
        let target: ItemTrait = parse_quote! {
            trait Foo {
                #[async_generic]
                fn foo() -> u8;
                #[async_generic]
                fn bar() -> u8;
            }
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait
        };

        let formatted2 = format_expand(target.clone(), args);

        assert_str_eq!(formatted1, formatted2);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_impl_split_async_generic_fns() {
        let target: ItemImpl = parse_quote! {
            impl Foo for A {
                #[async_generic]
                fn foo() -> u8 {}
                #[async_generic]
                fn bar() -> u8 {}
            }
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait
        };

        let formatted2 = format_expand(target.clone(), args);

        assert_str_eq!(formatted1, formatted2);

        let args: AsyncGenericArgs = parse_quote! {
            sync_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_trait_split_item_into_sync_and_async() {
        let target: ItemTrait = parse_quote! {
            trait Foo {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait
        };

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            async_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_impl_split_item_into_sync_and_async() {
        let target: ItemImpl = parse_quote! {
            impl Foo for A {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait
        };

        test_expand!(target.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            async_trait;
        };

        let formatted2 = format_expand(target, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_trait_custom_async_bounds() {
        let target: ItemTrait = parse_quote! {
            trait Foo {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait<T>: Send
            where
                Self: Sync,
                T: Sync;
        };

        test_expand!(target.clone(), args => formatted1);
    }

    #[test]
    fn test_expand_trait_custom_async_bounds_existing_generics() {
        let target: ItemTrait = parse_quote! {
            trait Foo<B> {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait<B, T>: Send
            where
                Self: Sync,
                T: Sync;
        };

        test_expand!(target.clone(), args => formatted1);
    }

    #[test]
    fn test_expand_impl_custom_async_bounds() {
        let target: ItemImpl = parse_quote! {
            impl Foo for A {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait<T>: Send
            where
                Self: Sync,
                T: Sync;
        };

        test_expand!(target.clone(), args => formatted1);
    }

    #[test]
    fn test_expand_impl_custom_async_bounds_existing_generics() {
        let target: ItemImpl = parse_quote! {
            impl<B> Foo<B> for A {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_trait<B, T>: Send
            where
                Self: Sync,
                T: Sync;
        };

        test_expand!(target.clone(), args => formatted1);
    }
}
