use std::marker::PhantomData;

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parenthesized,
    parse::{discouraged::Speculative, Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
    Block, Error, Expr, ExprBlock, FnArg, Generics, ImplItemFn, ItemFn, ReturnType, Signature,
    Token, TraitItemFn,
};

use self::kind::Kind;
use super::state;

pub mod kind;

#[inline]
pub fn transform(
    target_fn: impl Into<TargetItemFn>,
    async_signature: Option<AsyncSignature>,
) -> (
    AsyncGenericFn<kind::Sync, state::Final>,
    AsyncGenericFn<kind::Async, state::Final>,
) {
    fn transform(
        target_fn: TargetItemFn,
        async_signature: Option<AsyncSignature>,
    ) -> (
        AsyncGenericFn<kind::Sync, state::Final>,
        AsyncGenericFn<kind::Async, state::Final>,
    ) {
        (
            AsyncGenericFn::<kind::Sync, _>::new(target_fn.clone()).rewrite(),
            AsyncGenericFn::<kind::Async, _>::new(target_fn, async_signature).rewrite(),
        )
    }

    transform(target_fn.into(), async_signature)
}

pub fn expand(target_fn: TargetItemFn, async_signature: Option<AsyncSignature>) -> TokenStream2 {
    let (sync_fn, async_fn) = transform(target_fn, async_signature);

    quote! {
        #sync_fn
        #async_fn
    }
}

#[derive(Clone)]
pub enum TargetItemFn {
    FreeStanding(ItemFn),
    Trait(TraitItemFn),
    Impl(ImplItemFn),
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
            Err(Error::new(
                r#async.span,
                "an async_generic function should not be declared as async",
            ))
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

pub struct AsyncSignature {
    generics: Generics,
    inputs: Punctuated<FnArg, Token![,]>,
    output: ReturnType,
}

impl Parse for AsyncSignature {
    fn parse(input: ParseStream) -> syn::Result<Self> {
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

        let output = input.parse()?;

        generics.where_clause = if input.peek(Token![where]) {
            input.parse()?
        } else {
            None
        };

        Ok(AsyncSignature {
            generics,
            inputs,
            output,
        })
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

pub struct AsyncGenericFn<A, S> {
    pub(super) target: TargetItemFn,
    kind: A,
    _state: PhantomData<S>,
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

impl AsyncGenericFn<kind::Async, state::Initial> {
    pub(crate) const fn new(target: TargetItemFn, sig: Option<AsyncSignature>) -> Self {
        Self {
            target,
            kind: kind::Async(sig),
            _state: PhantomData,
        }
    }
}

trait CanTransformBlock {
    fn transform_block(initial: Option<Block>) -> Option<Block>;
}

impl<A: CanCompareToPredicate> CanTransformBlock for A {
    fn transform_block(initial: Option<Block>) -> Option<Block> {
        initial.map(|mut block| {
            IfAsyncRewriter::<A>(PhantomData).visit_block_mut(&mut block);
            block
        })
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

impl CanCompareToPredicate for kind::Async {
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
        let predicate = match ident.to_string().as_str() {
            "_sync" => AsyncPredicate::Sync,
            "_async" => AsyncPredicate::Async,
            _ => return,
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
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                block: Box::new(A::transform_block(Some(*f.block)).unwrap()),
                ..f
            }),
            TargetItemFn::Trait(f) => TargetItemFn::Trait(TraitItemFn {
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                default: A::transform_block(f.default),
                ..f
            }),
            TargetItemFn::Impl(f) => TargetItemFn::Impl(ImplItemFn {
                sig: Signature {
                    constness: A::transform_constness(f.sig.constness),
                    asyncness: A::asyncness(),
                    ident: A::transform_ident(f.sig.ident),
                    generics: self.kind.transform_generics(f.sig.generics),
                    inputs: self.kind.transform_inputs(f.sig.inputs),
                    output: self.kind.transform_output(f.sig.output),
                    ..f.sig
                },
                block: A::transform_block(Some(f.block)).unwrap(),
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
