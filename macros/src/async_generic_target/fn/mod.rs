use std::marker::PhantomData;

use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{
    parenthesized,
    parse::{discouraged::Speculative, Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
    Attribute, Block, Error, Expr, ExprBlock, FnArg, Generics, ImplItemFn, ItemFn, ReturnType,
    Signature, Token, TraitItemFn,
};

use self::kind::Kind;
use super::{parse_attrs, state};

pub mod kind;

pub mod kw {
    use syn::custom_keyword;

    custom_keyword!(async_signature);
}

const ERROR_PARSE_ARGS: &str =
    "`async_generic` on `fn` can only take an `async_signature` argument";
const ERROR_ASYNC_FN: &str = "an `async_generic` function should not be declared `async`";

#[inline]
pub fn split<const PRESERVE_IDENT: bool>(
    target_fn: impl Into<TargetItemFn>,
    args: AsyncGenericArgs,
) -> (
    AsyncGenericFn<kind::Sync, state::Final>,
    AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Final>,
) {
    fn transform<const PRESERVE_IDENT: bool>(
        target_fn: TargetItemFn,
        async_signature: Option<AsyncSignature>,
    ) -> (
        AsyncGenericFn<kind::Sync, state::Final>,
        AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Final>,
    ) {
        (
            AsyncGenericFn::<kind::Sync, _>::new(target_fn.clone()).rewrite(),
            AsyncGenericFn::<kind::Async<PRESERVE_IDENT>, _>::new(target_fn, async_signature).rewrite(),
        )
    }

    transform(target_fn.into(), args.0)
}

pub fn expand(target_fn: TargetItemFn, args: AsyncGenericArgs) -> TokenStream2 {
    let (sync_fn, async_fn) = split::<false>(target_fn, args);

    quote! {
        #sync_fn
        #async_fn
    }
}

pub struct AsyncGenericArgs(pub Option<AsyncSignature>);

pub struct AsyncSignature {
    attrs: Vec<Attribute>,
    _async_signature_token: kw::async_signature,
    generics: Generics,
    inputs: Punctuated<FnArg, Token![,]>,
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
        (!input.is_empty())
            .then(|| input.parse())
            .transpose()
            .map(Self)
    }
}

impl Parse for AsyncSignature {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = parse_attrs(input)?;
        let async_signature_token = input
            .parse()
            .map_err(|err| Error::new(err.span(), ERROR_PARSE_ARGS))?;

        let mut generics: Generics = input.parse()?;

        let content;
        parenthesized!(content in input);
        let inputs = Punctuated::parse_terminated(&content)?;

        let output = input.parse()?;

        generics.where_clause = if input.peek(Token![where]) {
            Some(input.parse()?)
        } else {
            None
        };

        Ok(AsyncSignature {
            attrs,
            _async_signature_token: async_signature_token,
            generics,
            inputs,
            output,
        })
    }
}

impl AsyncGenericFn<kind::Sync, state::Initial> {
    pub(crate) const fn new(target: TargetItemFn) -> Self {
        Self {
            target,
            kind: kind::Sync,
            _state: PhantomData,
        }
    }
}

impl<const PRESERVE_IDENT: bool> AsyncGenericFn<kind::Async<PRESERVE_IDENT>, state::Initial> {
    pub(crate) const fn new(target: TargetItemFn, sig: Option<AsyncSignature>) -> Self {
        Self {
            target,
            kind: kind::Async { signature: sig },
            _state: PhantomData,
        }
    }
}

trait CanTransformBlock {
    fn transform_block(initial: Block) -> Block;
}

impl<A: CanCompareToPredicate> CanTransformBlock for A {
    fn transform_block(mut initial: Block) -> Block {
        IfAsyncRewriter::<A>(PhantomData).visit_block_mut(&mut initial);
        initial
    }
}

trait CanRewriteBlock {
    fn rewrite_block(
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

struct IfAsyncRewriter<S>(PhantomData<S>);

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

        Self::rewrite_block(node, predicate, then_branch, else_branch);
    }
}

impl<A> AsyncGenericFn<A, state::Initial>
where
    A: Kind + CanCompareToPredicate,
{
    pub fn rewrite(mut self) -> AsyncGenericFn<A, state::Final> {
        let target = match self.target {
            TargetItemFn::FreeStanding(f) => TargetItemFn::FreeStanding(ItemFn {
                attrs: self.kind.extend_attrs(f.attrs),
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                block: Box::new(A::transform_block(*f.block)),
                ..f
            }),
            TargetItemFn::Trait(f) => TargetItemFn::Trait(TraitItemFn {
                attrs: self.kind.extend_attrs(f.attrs),
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                default: f.default.map(A::transform_block),
                ..f
            }),
            TargetItemFn::Impl(f) => TargetItemFn::Impl(ImplItemFn {
                attrs: self.kind.extend_attrs(f.attrs),
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                block: A::transform_block(f.block),
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
