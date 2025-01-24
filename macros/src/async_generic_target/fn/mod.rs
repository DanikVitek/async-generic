use std::marker::PhantomData;

use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    bracketed, parenthesized,
    parse::{discouraged::Speculative, End, Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    token::{Brace, Bracket},
    visit_mut::{self, VisitMut},
    Attribute, Block, Error, Expr, ExprAsync, ExprBlock, ExprCall, FnArg, Generics, ImplItemFn,
    ItemFn, Pat, PatType, PatWild, Receiver, ReturnType, Signature, Stmt, Token, TraitItemFn,
    Variadic,
};

use self::kind::Kind;
use super::{parse_attrs, state};

pub mod kind;

pub mod kw {
    use syn::custom_keyword;

    custom_keyword!(sync_signature);
    custom_keyword!(async_signature);
    custom_keyword!(async_fn);
    custom_keyword!(pin_box_fut);
    custom_keyword!(impl_fut);
}

const ERROR_PARSE_ARGS: &str =
    "`async_generic` on `fn` can only take a `sync_signature` or an `async_signature` argument";
const ERROR_ASYNC_FN: &str = "an `async_generic` function should not be declared `async`";

#[inline]
pub fn split<const PRESERVE_IDENT: bool>(
    target_fn: impl Into<TargetItemFn>,
    args: AsyncGenericArgs,
) -> (
    AsyncGenericFn<kind::Sync, state::Final>,
    AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Final>,
) {
    fn split<const PRESERVE_IDENT: bool>(
        target_fn: TargetItemFn,
        sync_signature: Option<SyncSignature>,
        async_signature: Option<AsyncSignature>,
    ) -> (
        AsyncGenericFn<kind::Sync, state::Final>,
        AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Final>,
    ) {
        (
            AsyncGenericFn::<kind::Sync, _>::new(target_fn.clone(), sync_signature).rewrite(),
            AsyncGenericFn::<kind::Async<PRESERVE_IDENT>, _>::new(target_fn, async_signature)
                .rewrite(),
        )
    }

    split(target_fn.into(), args.sync_signature, args.async_signature)
}

#[inline]
pub fn expand(target_fn: impl Into<TargetItemFn>, args: AsyncGenericArgs) -> TokenStream2 {
    fn expand(target_fn: TargetItemFn, args: AsyncGenericArgs) -> TokenStream2 {
        let (sync_fn, async_fn) = split::<false>(target_fn, args);

        quote! {
            #sync_fn
            #async_fn
        }
    }
    expand(target_fn.into(), args)
}

#[derive(Default)]
pub struct AsyncGenericArgs {
    pub sync_signature: Option<SyncSignature>,
    pub async_signature: Option<AsyncSignature>,
}

pub struct SyncSignature {
    attrs: Vec<Attribute>,
}

pub struct AsyncSignature {
    attrs: Vec<Attribute>,
    interface_kind: InterfaceKind,
    params: Option<AsyncSignatureParams>,
}

#[derive(Clone, Copy, Default)]
pub enum InterfaceKind {
    #[default]
    AsyncFn,
    PinBoxFut,
    ImplFut,
}

struct AsyncSignatureParams {
    generics: Generics,
    inputs: Punctuated<FnArg, Token![,]>,
    variadic: Option<Variadic>,
    output: ReturnType,
}

#[derive(Clone)]
pub enum TargetItemFn {
    FreeStanding(ItemFn),
    Trait(TraitItemFn),
    Impl(ImplItemFn),
}

pub struct AsyncGenericFn<A, S> {
    pub(super) target: TargetItemFn,
    kind: A,
    _state: PhantomData<S>,
}

impl TargetItemFn {
    fn sig(&self) -> &Signature {
        match self {
            Self::FreeStanding(f) => &f.sig,
            Self::Trait(f) => &f.sig,
            Self::Impl(f) => &f.sig,
        }
    }
}

impl From<ItemFn> for TargetItemFn {
    fn from(f: ItemFn) -> Self {
        Self::FreeStanding(f)
    }
}

impl From<TraitItemFn> for TargetItemFn {
    fn from(f: TraitItemFn) -> Self {
        Self::Trait(f)
    }
}

impl From<ImplItemFn> for TargetItemFn {
    fn from(f: ImplItemFn) -> Self {
        Self::Impl(f)
    }
}

impl TryFrom<TargetItemFn> for ItemFn {
    type Error = TargetItemFn;

    fn try_from(value: TargetItemFn) -> Result<Self, Self::Error> {
        match value {
            TargetItemFn::FreeStanding(f) => Ok(f),
            _ => Err(value),
        }
    }
}

impl TryFrom<TargetItemFn> for TraitItemFn {
    type Error = TargetItemFn;

    fn try_from(value: TargetItemFn) -> Result<Self, Self::Error> {
        match value {
            TargetItemFn::Trait(f) => Ok(f),
            _ => Err(value),
        }
    }
}

impl TryFrom<TargetItemFn> for ImplItemFn {
    type Error = TargetItemFn;

    fn try_from(value: TargetItemFn) -> Result<Self, Self::Error> {
        match value {
            TargetItemFn::Impl(f) => Ok(f),
            _ => Err(value),
        }
    }
}

impl Parse for TargetItemFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let target_fn = {
            use crate::util::InspectExt;

            let fork = input.fork();
            InspectExt::inspect(fork.parse().map(TargetItemFn::FreeStanding), |_| {
                input.advance_to(&fork)
            })
            .or_else(|mut err1| {
                let fork = input.fork();
                InspectExt::inspect(fork.parse().map(TargetItemFn::Trait), |_| {
                    input.advance_to(&fork)
                })
                .or_else(|err2| {
                    let fork = input.fork();
                    InspectExt::inspect(fork.parse().map(TargetItemFn::Impl), |_| {
                        input.advance_to(&fork)
                    })
                    .map_err(|err3| {
                        err1.extend([err2, err3]);
                        err1
                    })
                })
            })?
        };

        if let Some(r#async) = &target_fn.sig().asyncness {
            Err(Error::new(r#async.span, ERROR_ASYNC_FN))
        } else {
            Ok(target_fn)
        }
    }
}

impl ToTokens for TargetItemFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match &self {
            Self::FreeStanding(f) => f.to_tokens(tokens),
            Self::Trait(f) => f.to_tokens(tokens),
            Self::Impl(f) => f.to_tokens(tokens),
        }
    }
}

impl Parse for AsyncGenericArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::sync_signature) {
            parse_in_order::<SyncSignature, AsyncSignature>(input, attrs)
        } else if lookahead.peek(kw::async_signature) {
            parse_in_order::<AsyncSignature, SyncSignature>(input, attrs)
        } else if !input.is_empty() || !attrs.is_empty() {
            Err(lookahead.error())
        } else {
            Ok(Self::default())
        }
    }
}

trait CanSetAttrs {
    fn set_attrs(&mut self, attrs: Vec<Attribute>);
}

impl CanSetAttrs for SyncSignature {
    fn set_attrs(&mut self, attrs: Vec<Attribute>) {
        self.attrs = attrs;
    }
}

impl CanSetAttrs for AsyncSignature {
    fn set_attrs(&mut self, attrs: Vec<Attribute>) {
        self.attrs = attrs;
    }
}

impl From<SyncSignature> for AsyncGenericArgs {
    fn from(sync_signature: SyncSignature) -> Self {
        Self {
            sync_signature: Some(sync_signature),
            async_signature: None,
        }
    }
}

impl From<AsyncSignature> for AsyncGenericArgs {
    fn from(async_signature: AsyncSignature) -> Self {
        Self {
            sync_signature: None,
            async_signature: Some(async_signature),
        }
    }
}

impl From<(SyncSignature, AsyncSignature)> for AsyncGenericArgs {
    fn from((sync_signature, async_signature): (SyncSignature, AsyncSignature)) -> Self {
        Self {
            sync_signature: Some(sync_signature),
            async_signature: Some(async_signature),
        }
    }
}

impl From<(AsyncSignature, SyncSignature)> for AsyncGenericArgs {
    fn from((async_signature, sync_signature): (AsyncSignature, SyncSignature)) -> Self {
        Self {
            sync_signature: Some(sync_signature),
            async_signature: Some(async_signature),
        }
    }
}

fn parse_in_order<A, B>(
    input: ParseStream,
    attrs: Vec<Attribute>,
) -> syn::Result<AsyncGenericArgs>
where
    A: Parse + CanSetAttrs,
    B: Parse + CanSetAttrs,
    AsyncGenericArgs: From<A> + From<B> + From<(A, B)>,
{
    let mut async_signature: A = input.parse()?;
    async_signature.set_attrs(attrs);

    let lookahead = input.lookahead1();
    if !lookahead.peek(Token![;]) {
        if lookahead.peek(End) {
            return Ok(AsyncGenericArgs::from(async_signature));
        }
        return Err(lookahead.error());
    }
    let _: Token![;] = input.parse()?;

    if input.is_empty() {
        return Ok(AsyncGenericArgs::from(async_signature));
    }

    let sync_signature = input.parse()?;

    let lookahead = input.lookahead1();
    if !lookahead.peek(Token![;]) && !lookahead.peek(End) {
        return Err(lookahead.error());
    }
    let _: Option<Token![;]> = input.parse()?;

    Ok(AsyncGenericArgs::from((async_signature, sync_signature)))
}

impl Parse for SyncSignature {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let _: kw::sync_signature = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;

        Ok(SyncSignature { attrs })
    }
}

impl Parse for AsyncSignature {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let _: kw::async_signature = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;
        let interface_kind = if input.peek(Bracket) {
            let content;
            bracketed!(content in input);
            content.parse()?
        } else {
            InterfaceKind::default()
        };

        if input.is_empty() || input.peek(Token![;]) {
            return Ok(AsyncSignature {
                attrs,
                interface_kind,
                params: None,
            });
        }

        Ok(AsyncSignature {
            attrs,
            interface_kind,
            params: Some(input.parse()?),
        })
    }
}

impl Parse for InterfaceKind {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::async_fn) {
            input.parse::<kw::async_fn>().map(|_| Self::AsyncFn)
        } else if lookahead.peek(kw::pin_box_fut) {
            input.parse::<kw::pin_box_fut>().map(|_| Self::PinBoxFut)
        } else if lookahead.peek(kw::impl_fut) {
            input.parse::<kw::impl_fut>().map(|_| Self::ImplFut)
        } else {
            Err(lookahead.error())
        }
    }
}

impl Parse for AsyncSignatureParams {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut generics: Generics = input.parse()?;

        let content;
        parenthesized!(content in input);
        let (inputs, variadic) = parse_fn_args(&content)?;

        let output = input.parse()?;
        generics.where_clause = input.parse()?;

        Ok(Self {
            generics,
            inputs,
            variadic,
            output,
        })
    }
}

fn parse_fn_args(
    input: ParseStream,
) -> syn::Result<(Punctuated<FnArg, Token![,]>, Option<Variadic>)> {
    let mut args = Punctuated::new();
    let mut variadic = None;
    let mut has_receiver = false;

    while !input.is_empty() {
        let attrs = input.call(Attribute::parse_outer)?;

        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            variadic = Some(Variadic {
                attrs,
                pat: None,
                dots,
                comma: if input.is_empty() {
                    None
                } else {
                    Some(input.parse()?)
                },
            });
            break;
        }

        let allow_variadic = true;
        let arg = match parse_fn_arg_or_variadic(input, attrs, allow_variadic)? {
            FnArgOrVariadic::FnArg(arg) => arg,
            FnArgOrVariadic::Variadic(arg) => {
                variadic = Some(Variadic {
                    comma: if input.is_empty() {
                        None
                    } else {
                        Some(input.parse()?)
                    },
                    ..arg
                });
                break;
            }
        };

        match &arg {
            FnArg::Receiver(receiver) if has_receiver => {
                return Err(Error::new(
                    receiver.self_token.span,
                    "unexpected second method receiver",
                ));
            }
            FnArg::Receiver(receiver) if !args.is_empty() => {
                return Err(Error::new(
                    receiver.self_token.span,
                    "unexpected method receiver",
                ));
            }
            FnArg::Receiver(_) => has_receiver = true,
            FnArg::Typed(_) => {}
        }
        args.push_value(arg);

        if input.is_empty() {
            break;
        }

        let comma: Token![,] = input.parse()?;
        args.push_punct(comma);
    }

    Ok((args, variadic))
}

fn parse_fn_arg_or_variadic(
    input: ParseStream,
    attrs: Vec<Attribute>,
    allow_variadic: bool,
) -> syn::Result<FnArgOrVariadic> {
    let ahead = input.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        input.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(FnArg::Receiver(receiver)));
    }

    // Hack to parse pre-2018 syntax in
    // test/ui/rfc-2565-param-attrs/param-attrs-pretty.rs
    // because the rest of the test case is valuable.
    if input.peek(syn::Ident) && input.peek2(Token![<]) {
        let span = input.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVariadic::FnArg(FnArg::Typed(PatType {
            attrs,
            pat: Box::new(Pat::Wild(PatWild {
                attrs: Vec::new(),
                underscore_token: Token![_](span),
            })),
            colon_token: Token![:](span),
            ty: input.parse()?,
        })));
    }

    let pat = Box::new(Pat::parse_single(input)?);
    let colon_token: Token![:] = input.parse()?;

    if allow_variadic {
        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            return Ok(FnArgOrVariadic::Variadic(Variadic {
                attrs,
                pat: Some((pat, colon_token)),
                dots,
                comma: None,
            }));
        }
    }

    Ok(FnArgOrVariadic::FnArg(FnArg::Typed(PatType {
        attrs,
        pat,
        colon_token,
        ty: input.parse()?,
    })))
}

enum FnArgOrVariadic {
    FnArg(FnArg),
    Variadic(Variadic),
}

impl AsyncGenericFn<kind::Sync, state::Initial> {
    pub(crate) const fn new(target: TargetItemFn, sig: Option<SyncSignature>) -> Self {
        Self {
            target,
            kind: kind::Sync(sig),
            _state: PhantomData,
        }
    }
}

impl<const PRESERVE_IDENT: bool> AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Initial> {
    pub(crate) const fn new(target: TargetItemFn, sig: Option<AsyncSignature>) -> Self {
        Self {
            target,
            kind: kind::Async(sig),
            _state: PhantomData,
        }
    }
}

trait CanTransformBlock {
    fn transform_block(&self, initial: Block) -> Block;
}

impl<A: CanCompareToPredicate + CanSurround> CanTransformBlock for A {
    fn transform_block(&self, mut initial: Block) -> Block {
        IfAsyncRewriter::<A>(PhantomData).visit_block_mut(&mut initial);
        self.surround(&mut initial);
        initial
    }
}

pub trait CanSurround {
    fn surround(&self, block: &mut Block);
}

impl CanSurround for kind::Sync {
    fn surround(&self, _block: &mut Block) {}
}

impl<const PRESERVE_IDENT: bool> CanSurround for kind::Async<PRESERVE_IDENT> {
    fn surround(&self, block: &mut Block) {
        if let Some(sig) = self.0.as_ref() {
            fn surround_with_async(stmts: Vec<Stmt>) -> ExprAsync {
                ExprAsync {
                    attrs: vec![],
                    async_token: Default::default(),
                    capture: Some(Default::default()),
                    block: Block {
                        brace_token: Brace(Span::call_site()),
                        stmts,
                    },
                }
            }
            match sig.interface_kind {
                InterfaceKind::AsyncFn => {}
                InterfaceKind::PinBoxFut => {
                    let stmts = core::mem::take(&mut block.stmts);
                    block.stmts = vec![Stmt::Expr(
                        Expr::Call(ExprCall {
                            attrs: vec![],
                            func: parse_quote! { Box::pin },
                            paren_token: Default::default(),
                            args: {
                                let mut p = Punctuated::new();
                                p.push_value(Expr::Async(surround_with_async(stmts)));
                                p
                            },
                        }),
                        None,
                    )];
                }
                InterfaceKind::ImplFut => {
                    let stmts = core::mem::take(&mut block.stmts);
                    block.stmts = vec![Stmt::Expr(Expr::Async(surround_with_async(stmts)), None)];
                }
            }
        }
    }
}

trait CanRewriteBlock {
    fn rewrite_block(
        self,
        node: &mut Expr,
        predicate: AsyncPredicate,
        then_branch: Block,
        else_branch: Option<Expr>,
    );
}

pub enum AsyncPredicate {
    Sync,
    Async,
}

struct IfAsyncRewriter<A>(PhantomData<A>);

impl<A> Clone for IfAsyncRewriter<A> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<A> Copy for IfAsyncRewriter<A> {}

pub trait CanCompareToPredicate {
    fn cmp(predicate: AsyncPredicate) -> bool;
}

impl CanCompareToPredicate for kind::Sync {
    fn cmp(predicate: AsyncPredicate) -> bool {
        matches!(predicate, AsyncPredicate::Sync)
    }
}

impl<const PRESERVE_IDENT: bool> CanCompareToPredicate for kind::Async<PRESERVE_IDENT> {
    fn cmp(predicate: AsyncPredicate) -> bool {
        matches!(predicate, AsyncPredicate::Async)
    }
}

impl<A> CanRewriteBlock for IfAsyncRewriter<A>
where
    A: CanCompareToPredicate,
{
    fn rewrite_block(
        self,
        node: &mut Expr,
        predicate: AsyncPredicate,
        then_branch: Block,
        else_branch: Option<Expr>,
    ) {
        *node = if A::cmp(predicate) {
            Expr::Block(ExprBlock {
                attrs: vec![],
                label: None,
                block: then_branch,
            })
        } else if let Some(else_expr) = else_branch {
            else_expr
        } else {
            parse_quote! {{}}
        }
    }
}

impl<A> VisitMut for IfAsyncRewriter<A>
where
    IfAsyncRewriter<A>: CanRewriteBlock,
{
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        visit_mut::visit_expr_mut(self, node);

        let Expr::If(expr_if) = node else {
            return;
        };
        let Expr::Path(cond) = expr_if.cond.as_ref() else {
            return;
        };
        if !cond.attrs.is_empty() || !cond.qself.is_none() {
            return;
        }
        let Some(ident) = cond.path.get_ident() else {
            return;
        };

        let predicate = if *ident == "_sync" {
            AsyncPredicate::Sync
        } else if ident == "_async" {
            AsyncPredicate::Async
        } else {
            return;
        };
        let then_branch = expr_if.then_branch.clone();
        let else_branch = expr_if.else_branch.as_ref().map(|eb| *eb.1.clone());

        self.rewrite_block(node, predicate, then_branch, else_branch);
    }
}

impl<A> AsyncGenericFn<A, state::Initial>
where
    A: Kind + CanCompareToPredicate + CanSurround,
{
    pub fn rewrite(mut self) -> AsyncGenericFn<A, state::Final> {
        let mut rewrite_sig = |sig: Signature| Signature {
            constness: A::transform_constness(sig.constness),
            asyncness: self.kind.asyncness(),
            ident: A::transform_ident(sig.ident),
            generics: self.kind.transform_generics(sig.generics),
            inputs: self.kind.transform_inputs(sig.inputs),
            variadic: self.kind.transform_variadic(sig.variadic),
            output: self.kind.transform_output(sig.output),
            ..sig
        };

        let target = match self.target {
            TargetItemFn::FreeStanding(f) => TargetItemFn::FreeStanding(ItemFn {
                sig: rewrite_sig(f.sig),
                attrs: self.kind.extend_attrs(f.attrs),
                block: Box::new(self.kind.transform_block(*f.block)),
                ..f
            }),
            TargetItemFn::Trait(f) => TargetItemFn::Trait(TraitItemFn {
                sig: rewrite_sig(f.sig),
                attrs: self.kind.extend_attrs(f.attrs),
                default: f.default.map(|block| self.kind.transform_block(block)),
                ..f
            }),
            TargetItemFn::Impl(f) => TargetItemFn::Impl(ImplItemFn {
                sig: rewrite_sig(f.sig),
                attrs: self.kind.extend_attrs(f.attrs),
                block: self.kind.transform_block(f.block),
                ..f
            }),
        };

        AsyncGenericFn {
            target,
            kind: self.kind,
            _state: PhantomData,
        }
    }
}

impl<A> ToTokens for AsyncGenericFn<A, state::Final> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.target.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_str_eq;

    use super::*;
    use crate::test_helpers::local_assert_snapshot;

    fn format_expand(target_fn: ItemFn, args: AsyncGenericArgs) -> String {
        let expanded = expand(target_fn, args);
        prettyplease::unparse(&parse_quote!(#expanded))
    }

    macro_rules! test_expand {
        ($target_fn:expr, $args:expr) => {
            test_expand!($target_fn, $args => formatted);
        };
        ($target_fn:expr, $args:expr => $formatted: ident) => {
            let $formatted = format_expand($target_fn, $args);
            local_assert_snapshot!($formatted);
        };
    }

    #[test]
    fn test_expand_nop() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target_fn.clone(), args => formatted_default);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature;
        };

        let formatted_sync = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature;
        };

        let formatted_async = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature; async_signature;
        };

        let formatted_sync_async = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature; sync_signature;
        };

        let formatted_async_sync = format_expand(target_fn, args);

        assert_str_eq!(formatted_default, formatted_sync);
        assert_str_eq!(formatted_default, formatted_async);
        assert_str_eq!(formatted_default, formatted_sync_async);
        assert_str_eq!(formatted_default, formatted_async_sync);
    }

    #[test]
    fn test_expand_nop_with_ret_ty() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() -> String {}
        };
        let args: AsyncGenericArgs = parse_quote!();

        test_expand!(target_fn.clone(), args => formatted_default);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature
        };

        let formatted_sync = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature;
        };

        let formatted_sync_comma = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature
        };

        let formatted_async = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature;
        };

        let formatted_async_comma = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature; async_signature
        };

        let formatted_sync_async = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature; sync_signature
        };

        let formatted_async_sync = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            sync_signature; async_signature;
        };

        let formatted_sync_async_comma = format_expand(target_fn.clone(), args);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature; sync_signature;
        };

        let formatted_async_sync_comma = format_expand(target_fn, args);

        assert_str_eq!(formatted_default, formatted_sync);
        assert_str_eq!(formatted_default, formatted_async);
        assert_str_eq!(formatted_default, formatted_sync_async);
        assert_str_eq!(formatted_default, formatted_async_sync);

        assert_str_eq!(formatted_sync, formatted_sync_comma);
        assert_str_eq!(formatted_async, formatted_async_comma);
        assert_str_eq!(formatted_sync_async, formatted_sync_async_comma);
        assert_str_eq!(formatted_async_sync, formatted_async_sync_comma);
    }

    #[test]
    fn test_expand_sync1() {
        let target_fn: ItemFn = parse_quote! {
            /// # Common docs
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            /// # Sync Docs
            sync_signature;
        };

        test_expand!(target_fn, args);
    }

    #[test]
    fn test_expand_async_default() {
        let target_fn: ItemFn = parse_quote! {
            /// # Common docs
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            /// # Async Docs
            async_signature;
        };

        test_expand!(target_fn, args);
    }

    #[test]
    fn test_expand_async_change_signature1() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() -> String {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature();
        };

        test_expand!(target_fn, args);
    }

    #[test]
    fn test_expand_async_change_signature2() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() -> String {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature() -> StringAsync;
        };

        test_expand!(target_fn, args);
    }

    #[test]
    fn test_expand_async_change_signature3() {
        let target_fn: ItemFn = parse_quote! {
            fn foo<R: Read>(reader: &mut R) -> String {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature<R: AsyncRead>(reader: &mut R) -> String;
        };

        test_expand!(target_fn, args);
    }

    #[test]
    fn test_expand_sync_async1() {
        let target_fn: ItemFn = parse_quote! {
            /// # Common docs
            fn foo<T>() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            /// # Sync Docs
            sync_signature;
            /// # Async Docs
            async_signature<T>() -> impl Future<Output=()> + Send where T: Send;
        };

        test_expand!(target_fn.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            /// # Async Docs
            async_signature<T>() -> impl Future<Output=()> + Send where T: Send;
            /// # Sync Docs
            sync_signature;
        };

        let formatted2 = format_expand(target_fn, args);

        assert_str_eq!(formatted1, formatted2);
    }
    
    #[test]
    fn test_expand_async_interface_kind_async_fn() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[async_fn];
        };

        test_expand!(target_fn.clone(), args => formatted1);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[async_fn]
        };
        
        let formatted2 = format_expand(target_fn.clone(), args);
        
        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[async_fn]()
        };
        
        let formatted2 = format_expand(target_fn.clone(), args);
        
        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[async_fn]();
        };
        
        let formatted2 = format_expand(target_fn, args);
        
        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_async_interface_kind_impl_fut() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[impl_fut];
        };

        test_expand!(target_fn.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature[impl_fut]
        };

        let formatted2 = format_expand(target_fn.clone(), args);

        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[impl_fut]()
        };

        let formatted2 = format_expand(target_fn.clone(), args);

        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[impl_fut]();
        };

        let formatted2 = format_expand(target_fn, args);

        assert_str_eq!(formatted1, formatted2);
    }

    #[test]
    fn test_expand_async_interface_kind_pin_box_fut() {
        let target_fn: ItemFn = parse_quote! {
            fn foo() {}
        };
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[pin_box_fut];
        };

        test_expand!(target_fn.clone(), args => formatted1);

        let args: AsyncGenericArgs = parse_quote! {
            async_signature[pin_box_fut]
        };

        let formatted2 = format_expand(target_fn.clone(), args);

        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[pin_box_fut]()
        };

        let formatted2 = format_expand(target_fn.clone(), args);

        assert_str_eq!(formatted1, formatted2);
        
        let args: AsyncGenericArgs = parse_quote! {
            async_signature[pin_box_fut]();
        };

        let formatted2 = format_expand(target_fn, args);

        assert_str_eq!(formatted1, formatted2);
    }
}
